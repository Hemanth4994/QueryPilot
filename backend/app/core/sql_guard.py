"""Deterministic PostgreSQL-safe read-only & cross-tenant SQL guard + bounded execution."""

import re

import pandas as pd
from sqlalchemy import text

from ..config import settings
from .database import INTERNAL_TABLES, get_read_engine, tenant_role_name

FORBIDDEN_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
    "COPY",
    "SET",
    "RESET",
    "GRANT",
    "REVOKE",
    "EXEC",
    "EXECUTE",
    "CALL",
    "DO",
    "VACUUM",
    "ANALYZE",
    "CLUSTER",
    "LOCK",
    "MERGE",
    "REINDEX",
    "COMMENT",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "PREPARE",
    "DEALLOCATE",
    "LISTEN",
    "NOTIFY",
    "LOAD",
    "IMPORT",
    "PRAGMA",
    "ATTACH",
    "DETACH",
)

FORBIDDEN_SYSTEM_IDENTIFIERS = INTERNAL_TABLES | frozenset(
    {
        "pg_catalog",
        "information_schema",
        "pg_tables",
        "pg_class",
        "pg_namespace",
        "pg_roles",
        "pg_user",
        "pg_shadow",
        "pg_authid",
        "pg_stat_activity",
        "pg_settings",
        "sqlite_master",
        "sqlite_schema",
        "sqlite_temp_master",
    }
)

FORBIDDEN_FUNCTIONS = frozenset(
    {
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "lo_import",
        "lo_export",
        "dblink",
        "dblink_exec",
        "query_to_xml",
        "query_to_xmlschema",
        "query_to_xml_and_xmlschema",
        "table_to_xml",
        "table_to_xmlschema",
        "table_to_xml_and_xmlschema",
        "schema_to_xml",
        "schema_to_xmlschema",
        "schema_to_xml_and_xmlschema",
        "cursor_to_xml",
        "cursor_to_xmlschema",
        "cursor_to_xml_and_xmlschema",
        "database_to_xml",
        "database_to_xmlschema",
        "database_to_xml_and_xmlschema",
    }
)

ALLOWED_SCHEMAS = frozenset({"public", "main"})

_CLAUSE_STOP_WORDS = frozenset(
    {
        "WHERE",
        "GROUP",
        "HAVING",
        "ORDER",
        "LIMIT",
        "OFFSET",
        "FETCH",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "JOIN",
        "LEFT",
        "RIGHT",
        "INNER",
        "OUTER",
        "FULL",
        "CROSS",
        "NATURAL",
        "ON",
        "USING",
        "WINDOW",
        "FOR",
        "INTO",
        "VALUES",
        "SET",
    }
)

_NON_FUNC_PRECEDING_TOKENS = frozenset(
    {
        "SELECT",
        "FROM",
        "JOIN",
        "WHERE",
        "AND",
        "OR",
        "NOT",
        "IN",
        "EXISTS",
        "ANY",
        "ALL",
        "SOME",
        "AS",
        "ON",
        "USING",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "HAVING",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "WITH",
        "(",
        ",",
        "=",
        "<>",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "+",
        "-",
        "*",
        "/",
        "||",
    }
)

_FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
_SELECT_INTO_PATTERN = re.compile(r"\bSELECT\b[\s\S]*?\bINTO\b", re.IGNORECASE)
_DOLLAR_TAG_PATTERN = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
_TENANT_TABLE_PATTERN = re.compile(r"\bu_(\d+)_[a-z0-9_]+\b", re.IGNORECASE)
_SYSTEM_IDENT_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(FORBIDDEN_SYSTEM_IDENTIFIERS)) + r")\b",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[(),.;]|!=|<>|<=|>=|\|\||[=<>+\-*/]")


def _extract_sql_for_analysis(query: str) -> tuple[str, str, str | None]:
    """Return (keyword_safe_sql, identifier_preserved_sql, parse_error).

    - keyword_safe_sql blanks single-quoted literals, double-quoted identifiers, and comments.
    - identifier_preserved_sql blanks single-quoted literals and comments while unquoting
      double-quoted identifiers so table-ownership checks cannot be bypassed via quoting.
    """
    kw_out: list[str] = []
    ident_out: list[str] = []
    i = 0
    n = len(query)

    while i < n:
        ch = query[i]

        # Line comment: -- ... \n
        if ch == "-" and i + 1 < n and query[i + 1] == "-":
            end = query.find("\n", i + 2)
            kw_out.append(" ")
            ident_out.append(" ")
            i = n if end == -1 else end + 1
            continue

        # Block comment: /* ... */
        if ch == "/" and i + 1 < n and query[i + 1] == "*":
            end = query.find("*/", i + 2)
            if end == -1:
                return "", "", "Security Error: Unclosed SQL block comment is not permitted."
            kw_out.append(" ")
            ident_out.append(" ")
            i = end + 2
            continue

        # Single-quoted string literal: '...'
        if ch == "'":
            i += 1
            closed = False
            while i < n:
                if query[i] == "'":
                    if i + 1 < n and query[i + 1] == "'":
                        i += 2
                        continue
                    closed = True
                    i += 1
                    break
                i += 1
            if not closed:
                return "", "", "Security Error: Unclosed SQL string literal."
            kw_out.append(" ")
            ident_out.append(" ")
            continue

        # Double-quoted identifier: "..."
        if ch == '"':
            i += 1
            ident_chars: list[str] = []
            closed = False
            while i < n:
                if query[i] == '"':
                    if i + 1 < n and query[i + 1] == '"':
                        ident_chars.append('"')
                        i += 2
                        continue
                    closed = True
                    i += 1
                    break
                ident_chars.append(query[i])
                i += 1
            if not closed:
                return "", "", "Security Error: Unclosed SQL quoted identifier."
            kw_out.append(" ")
            ident_out.append(f" {''.join(ident_chars)} ")
            continue

        # Disallow PostgreSQL dollar-quoted blocks (e.g. $$ ... $$)
        if ch == "$":
            match = _DOLLAR_TAG_PATTERN.match(query, i)
            if match:
                return "", "", "Security Error: Dollar-quoted SQL blocks are not permitted."

        kw_out.append(ch)
        ident_out.append(ch)
        i += 1

    return "".join(kw_out), "".join(ident_out), None


def _extract_cte_names(tokens: list[str]) -> set[str]:
    """Extract CTE names defined in top-level WITH clauses."""
    cte_names: set[str] = set()
    if not tokens or tokens[0].upper() != "WITH":
        return cte_names

    idx = 1
    if idx < len(tokens) and tokens[idx].upper() == "RECURSIVE":
        idx += 1

    depth = 0
    expect_cte_name = True
    while idx < len(tokens):
        tok = tokens[idx]
        upper = tok.upper()

        if depth == 0:
            if expect_cte_name:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tok):
                    cte_names.add(tok.lower())
                    expect_cte_name = False
                elif upper == "SELECT":
                    break
            elif upper == "AS" and idx + 1 < len(tokens) and tokens[idx + 1] == "(":
                depth = 1
                idx += 2
                continue
            elif upper == "SELECT":
                break
            elif tok == ",":
                expect_cte_name = True
        else:
            if tok == "(":
                depth += 1
            elif tok == ")":
                depth -= 1
                if depth == 0:
                    if idx + 1 < len(tokens) and tokens[idx + 1] == ",":
                        expect_cte_name = True
                        idx += 2
                        continue
                    break
        idx += 1

    return cte_names


def _extract_referenced_tables(ident_sql: str) -> tuple[list[tuple[str | None, str]], set[str], str | None]:
    """Parse table references (schema, table_name) and CTE names outside function arguments."""
    tokens = _TOKEN_PATTERN.findall(ident_sql)
    cte_names = _extract_cte_names(tokens)
    refs: list[tuple[str | None, str]] = []

    paren_stack: list[bool] = []  # True when inside function call arguments
    in_from_list_at_depth: dict[int, bool] = {}

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        upper = tok.upper()

        # Check forbidden functions like pg_read_file(...) and XML dump functions
        if i + 1 < n and tokens[i + 1] == "(":
            func_name = tok.lower().rsplit(".", 1)[-1]
            if (
                func_name in FORBIDDEN_FUNCTIONS
                or "_to_xml" in func_name
                or "_to_xmlschema" in func_name
            ):
                return [], cte_names, f"Security Error: Forbidden SQL function '{tok}' is not permitted."

        if tok == "(":
            prev_upper = tokens[i - 1].upper() if i > 0 else ""
            is_func = bool(
                i > 0
                and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tokens[i - 1])
                and prev_upper not in _NON_FUNC_PRECEDING_TOKENS
            )
            paren_stack.append(is_func)
            i += 1
            continue

        if tok == ")":
            depth = len(paren_stack)
            in_from_list_at_depth.pop(depth, None)
            if paren_stack:
                paren_stack.pop()
            i += 1
            continue

        inside_func_call = any(paren_stack)
        depth = len(paren_stack)

        if inside_func_call:
            i += 1
            continue

        if upper in ("FROM", "JOIN"):
            if upper == "FROM":
                in_from_list_at_depth[depth] = True
            i += 1
            if i < n and tokens[i].upper() in ("LATERAL", "ONLY"):
                i += 1
            if i < n and tokens[i] != "(":
                schema_name: str | None = None
                table_name = tokens[i]
                if i + 2 < n and tokens[i + 1] == ".":
                    schema_name = table_name.lower()
                    table_name = tokens[i + 2]
                    i += 2
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
                    refs.append((schema_name, table_name.lower()))
                i += 1
                # Skip optional alias (`AS alias` or `alias`)
                if i < n and tokens[i].upper() == "AS":
                    i += 2
                elif (
                    i < n
                    and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tokens[i])
                    and tokens[i].upper() not in _CLAUSE_STOP_WORDS
                ):
                    i += 1
                continue
            continue

        if in_from_list_at_depth.get(depth):
            if tok == ",":
                i += 1
                if i < n and tokens[i].upper() in ("LATERAL", "ONLY"):
                    i += 1
                if i < n and tokens[i] != "(":
                    schema_name = None
                    table_name = tokens[i]
                    if i + 2 < n and tokens[i + 1] == ".":
                        schema_name = table_name.lower()
                        table_name = tokens[i + 2]
                        i += 2
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
                        refs.append((schema_name, table_name.lower()))
                    i += 1
                    if i < n and tokens[i].upper() == "AS":
                        i += 2
                    elif (
                        i < n
                        and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tokens[i])
                        and tokens[i].upper() not in _CLAUSE_STOP_WORDS
                    ):
                        i += 1
                    continue
            elif upper in _CLAUSE_STOP_WORDS or tok == ";":
                in_from_list_at_depth[depth] = False

        i += 1

    return refs, cte_names, None


def validate_read_only_sql(
    query: str,
    user_id: int | None = None,
    allowed_tables: set[str] | None = None,
) -> str | None:
    """Validate read-only SELECT/WITH syntax, internal tables, and cross-tenant ownership."""
    if not query or not query.strip():
        return "Security Error: Only read-only SELECT/WITH queries are permitted."

    kw_sql, ident_sql, parse_error = _extract_sql_for_analysis(query)
    if parse_error is not None:
        return parse_error

    trimmed_kw = kw_sql.strip()
    if not trimmed_kw:
        return "Security Error: Only read-only SELECT/WITH queries are permitted."

    upper = trimmed_kw.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return "Security Error: Only read-only SELECT/WITH queries are permitted."

    # Block stacked/multi-statement queries (allow only one optional trailing semicolon)
    if ";" in trimmed_kw[:-1]:
        return "Security Error: Multiple SQL statements are not allowed."

    keyword_match = _FORBIDDEN_PATTERN.search(trimmed_kw)
    if keyword_match:
        return (
            f"Security Error: Forbidden SQL keyword '{keyword_match.group(1).upper()}' "
            "is not permitted in read-only mode."
        )

    if _SELECT_INTO_PATTERN.search(trimmed_kw):
        return "Security Error: SELECT INTO statements are not permitted."

    # Block internal tables and system catalogs anywhere outside single-quoted string literals
    sys_match = _SYSTEM_IDENT_PATTERN.search(ident_sql)
    if sys_match:
        return (
            f"Security Error: Access to restricted table or catalog '{sys_match.group(1).lower()}' "
            "is not permitted."
        )

    # Block any explicit reference to another tenant's u_<other_id>_* table name
    if user_id is not None:
        for match in _TENANT_TABLE_PATTERN.finditer(ident_sql):
            owner_id = int(match.group(1))
            if owner_id != user_id:
                return (
                    f"Security Error: Cross-tenant access to table '{match.group(0).lower()}' "
                    "is forbidden."
                )

    refs, cte_names, ref_error = _extract_referenced_tables(ident_sql)
    if ref_error is not None:
        return ref_error

    normalized_allowed = {t.lower() for t in allowed_tables} if allowed_tables is not None else None
    for schema_name, table_name in refs:
        if schema_name is not None and schema_name not in ALLOWED_SCHEMAS:
            return f"Security Error: Access to schema '{schema_name}' is not permitted."
        if table_name in cte_names:
            continue
        if table_name in FORBIDDEN_SYSTEM_IDENTIFIERS:
            return f"Security Error: Access to restricted table '{table_name}' is not permitted."
        if user_id is not None:
            tenant_match = _TENANT_TABLE_PATTERN.match(table_name)
            if tenant_match and int(tenant_match.group(1)) != user_id:
                return (
                    f"Security Error: Cross-tenant access to table '{table_name}' is forbidden."
                )
        if normalized_allowed is not None and table_name not in normalized_allowed:
            return f"Security Error: Access to table '{table_name}' is not permitted."

    return None


def _strip_trailing_semicolon(query: str) -> str:
    """Remove a single trailing semicolon so driver execution stays uniform."""
    stripped = query.strip()
    if stripped.endswith(";"):
        return stripped[:-1].strip()
    return stripped


def execute_sql_bounded(
    query: str,
    user_id: int | None = None,
    allowed_tables: set[str] | None = None,
) -> tuple[pd.DataFrame | None, str | None, bool]:
    """Validate and execute read-only SQL inside a bounded, tenant-safe transaction."""
    security_error = validate_read_only_sql(
        query,
        user_id=user_id,
        allowed_tables=allowed_tables,
    )
    if security_error is not None:
        return None, security_error, False

    exec_query = _strip_trailing_semicolon(query)
    max_rows = int(settings.max_query_rows)
    timeout_ms = int(settings.query_timeout_ms)
    read_eng = get_read_engine()
    is_pg = read_eng.dialect.name == "postgresql"

    try:
        with read_eng.connect() as conn:
            if is_pg:
                with conn.begin():
                    conn.execute(text("SET LOCAL TRANSACTION READ ONLY"))
                    if user_id is not None:
                        role_name = tenant_role_name(user_id)
                        conn.execute(text(f"SET LOCAL ROLE {role_name}"))
                    conn.execute(text(f"SET LOCAL statement_timeout = {timeout_ms}"))
                    result = conn.execute(text(exec_query))
                    columns = list(result.keys())
                    rows = result.fetchmany(max_rows + 1)
            else:
                result = conn.execute(text(exec_query))
                columns = list(result.keys())
                rows = result.fetchmany(max_rows + 1)

        truncated = len(rows) > max_rows
        if truncated:
            rows = rows[:max_rows]
        df = pd.DataFrame(rows, columns=columns)
        return df, None, truncated
    except Exception as exc:
        return None, str(exc), False


def execute_sql(
    query: str,
    user_id: int | None = None,
    allowed_tables: set[str] | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Validate and execute read-only SELECT/WITH SQL returning (df, error)."""
    df, error, _ = execute_sql_bounded(
        query,
        user_id=user_id,
        allowed_tables=allowed_tables,
    )
    return df, error
