"""PostgreSQL (and test SQLite) engine management, pooling, seeding, and tenant schema scope."""

from datetime import datetime, timedelta
import logging
from pathlib import Path
import random
import re
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from ..config import settings

logger = logging.getLogger(__name__)

CORE_TABLES = frozenset({"customers", "products", "orders"})
INTERNAL_TABLES = frozenset({"schema_embeddings", "users"})
PROTECTED_TABLES = CORE_TABLES | INTERNAL_TABLES
USER_TABLE_PREFIX_RE = re.compile(r"^u_(\d+)_[a-z0-9_]+$")
_PG_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_engine: Engine | None = None
_read_engine: Engine | None = None
_configured_url: str | None = None
_configured_read_url: str | None = None


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    """Create parent directory for file-backed SQLite URLs when missing."""
    if not database_url.startswith("sqlite:///"):
        return
    raw_path = database_url.removeprefix("sqlite:///")
    if not raw_path or raw_path == ":memory:":
        return
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)


def _create_pair(
    database_url: str,
    read_database_url: str | None = None,
) -> tuple[Engine, Engine]:
    """Create pooled PostgreSQL engine pair (requiring READ_DATABASE_URL) or SQLite pair."""
    if database_url.startswith("sqlite"):
        _ensure_sqlite_parent_dir(database_url)
        connect_args = {"check_same_thread": False}
        write_eng = create_engine(database_url, connect_args=connect_args)
        read_eng = create_engine(database_url, connect_args=connect_args)

        @event.listens_for(read_eng, "connect")
        def _make_read_only(dbapi_connection, connection_record):
            dbapi_connection.execute("PRAGMA query_only = ON")

        return write_eng, read_eng

    read_url = (read_database_url or "").strip()
    if not read_url:
        raise RuntimeError(
            "READ_DATABASE_URL is required when DATABASE_URL is PostgreSQL. "
            "Provision a least-privilege role (infra/init-readonly-role.sql) and set READ_DATABASE_URL."
        )
    if read_url.startswith("sqlite"):
        raise RuntimeError("READ_DATABASE_URL must be a PostgreSQL URL when DATABASE_URL is PostgreSQL.")

    pg_eng = create_engine(
        database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
    )
    read_pg_eng = create_engine(
        read_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
    )
    return pg_eng, read_pg_eng


def configure_database(
    database_url: str | None = None,
    read_database_url: str | None = None,
) -> tuple[Engine, Engine]:
    """Reconfigure active database engines (used at startup and in isolated tests)."""
    global _engine, _read_engine, _configured_url, _configured_read_url
    target_url = database_url or settings.database_url
    target_read_url = (
        read_database_url
        if read_database_url is not None
        else ("" if target_url.startswith("sqlite") else settings.read_database_url)
    )
    if _engine is not None:
        _engine.dispose()
    if _read_engine is not None and _read_engine is not _engine:
        _read_engine.dispose()
    _engine, _read_engine = _create_pair(target_url, read_database_url=target_read_url)
    _configured_url = target_url
    _configured_read_url = target_read_url
    return _engine, _read_engine


def get_engine() -> Engine:
    """Return the active SQLAlchemy write engine."""
    if _engine is None:
        configure_database(settings.database_url)
    assert _engine is not None
    return _engine


def get_read_engine() -> Engine:
    """Return the active SQLAlchemy read execution engine."""
    if _read_engine is None:
        configure_database(settings.database_url)
    assert _read_engine is not None
    return _read_engine


def is_postgres(target_engine: Engine | None = None) -> bool:
    """Return True when the active engine dialect is PostgreSQL."""
    eng = target_engine or get_engine()
    return eng.dialect.name == "postgresql"


def get_readonly_role(
    read_database_url: str | None = None,
    write_database_url: str | None = None,
) -> str | None:
    """Extract the dedicated read-only PostgreSQL role name when READ_DATABASE_URL is set."""
    read_url = (
        read_database_url
        if read_database_url is not None
        else (_configured_read_url or settings.read_database_url)
    ).strip()
    write_url = (write_database_url or _configured_url or settings.database_url).strip()
    if not read_url or read_url.startswith("sqlite") or write_url.startswith("sqlite"):
        return None
    try:
        read_user = (make_url(read_url).username or "").strip()
        write_user = (make_url(write_url).username or "").strip()
    except Exception:
        return None
    if not read_user or read_user == write_user or not _PG_IDENT_RE.match(read_user):
        return None
    return read_user


def tenant_role_name(user_id: int) -> str:
    """Return the deterministic PostgreSQL NOLOGIN role name for a tenant user ID."""
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 0:
        raise ValueError("user_id must be a non-negative integer.")
    return f"qp_tenant_{user_id}"


def _provision_tenant_role_sql(
    executor: Any,
    user_id: int,
    reader_role: str | None = None,
    existing_tables: set[str] | None = None,
) -> str:
    """Create qp_tenant_{id} NOLOGIN and grant schema USAGE, core SELECT, and reader membership."""
    role_name = tenant_role_name(user_id)
    executor.execute(
        text(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role_name}') THEN
                    CREATE ROLE {role_name} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE;
                END IF;
            END
            $$;
            """
        )
    )
    executor.execute(text(f"GRANT USAGE ON SCHEMA public TO {role_name};"))
    core_targets = (
        sorted(CORE_TABLES & existing_tables)
        if existing_tables is not None
        else sorted(CORE_TABLES)
    )
    if core_targets:
        quoted_core = ", ".join(core_targets)
        executor.execute(text(f"GRANT SELECT ON {quoted_core} TO {role_name};"))

    grant_to_role = reader_role or get_readonly_role()
    if grant_to_role and _PG_IDENT_RE.match(grant_to_role):
        reader_exists = executor.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :role"),
            {"role": grant_to_role},
        ).scalar()
        if reader_exists:
            executor.execute(text(f'GRANT {role_name} TO "{grant_to_role}";'))
    return role_name


def ensure_tenant_role(
    user_id: int,
    target_engine: Engine | None = None,
    executor: Any | None = None,
) -> bool:
    """Create qp_tenant_{id} NOLOGIN on PostgreSQL and grant shared core-table SELECT."""
    eng = target_engine or get_engine()
    if not is_postgres(eng):
        return False
    reader_role = get_readonly_role()
    if executor is not None:
        _provision_tenant_role_sql(executor, user_id, reader_role=reader_role)
        return True
    try:
        with eng.begin() as conn:
            _provision_tenant_role_sql(conn, user_id, reader_role=reader_role)
        return True
    except SQLAlchemyError as exc:
        logger.warning("Unable to provision tenant role for user %s: %s", user_id, exc)
        raise


def grant_readonly_select(
    table_name: str,
    target_engine: Engine | None = None,
    user_id: int | None = None,
) -> bool:
    """Grant SELECT on a tenant table ONLY to qp_tenant_{id} (or core table to shared reader)."""
    eng = target_engine or get_engine()
    if not is_postgres(eng) or table_name in INTERNAL_TABLES or not _PG_IDENT_RE.match(table_name):
        return False

    tenant_match = USER_TABLE_PREFIX_RE.match(table_name)
    if tenant_match:
        owner_id = int(tenant_match.group(1))
        if user_id is not None and int(user_id) != owner_id:
            raise ValueError(
                f"Tenant table '{table_name}' does not match user_id={user_id}."
            )
        target_role = tenant_role_name(owner_id)
        reader_role = get_readonly_role()
        try:
            with eng.begin() as conn:
                ensure_tenant_role(owner_id, target_engine=eng, executor=conn)
                conn.execute(text(f'GRANT SELECT ON TABLE "{table_name}" TO {target_role};'))
                if reader_role and _PG_IDENT_RE.match(reader_role):
                    reader_exists = conn.execute(
                        text("SELECT 1 FROM pg_roles WHERE rolname = :role"),
                        {"role": reader_role},
                    ).scalar()
                    if reader_exists:
                        conn.execute(text(f'REVOKE ALL ON TABLE "{table_name}" FROM "{reader_role}";'))
            return True
        except SQLAlchemyError as exc:
            logger.warning(
                "Unable to grant SELECT on '%s' to tenant role '%s': %s",
                table_name,
                target_role,
                exc,
            )
            return False

    if table_name not in CORE_TABLES:
        return False

    role = get_readonly_role()
    if not role:
        return False
    try:
        with eng.begin() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname = :role"),
                {"role": role},
            ).scalar()
            if not exists:
                return False
            conn.execute(text(f'GRANT USAGE ON SCHEMA public TO "{role}";'))
            conn.execute(text(f'GRANT SELECT ON TABLE "{table_name}" TO "{role}";'))
        return True
    except SQLAlchemyError as exc:
        logger.warning("Unable to grant SELECT on '%s' to role '%s': %s", table_name, role, exc)
        return False


def sync_readonly_role_privileges(target_engine: Engine | None = None) -> list[int]:
    """Enforce per-tenant qp_tenant_{id} roles, core grants, and internal/tenant isolation."""
    eng = target_engine or get_engine()
    if not is_postgres(eng):
        return []
    role = get_readonly_role()
    inspector = inspect(eng)
    existing = set(inspector.get_table_names())
    tenant_ids: set[int] = set()
    tenant_tables_by_owner: dict[int, list[str]] = {}
    for tbl in sorted(existing):
        if not _PG_IDENT_RE.match(tbl):
            continue
        match = USER_TABLE_PREFIX_RE.match(tbl)
        if match:
            owner_id = int(match.group(1))
            tenant_ids.add(owner_id)
            tenant_tables_by_owner.setdefault(owner_id, []).append(tbl)

    try:
        with eng.begin() as conn:
            if "users" in existing:
                user_rows = conn.execute(text("SELECT id FROM users ORDER BY id")).scalars().all()
                tenant_ids.update(int(uid) for uid in user_rows)

            reader_exists = bool(
                role
                and conn.execute(
                    text("SELECT 1 FROM pg_roles WHERE rolname = :role"),
                    {"role": role},
                ).scalar()
            )
            if reader_exists and role is not None:
                conn.execute(text(f'GRANT USAGE ON SCHEMA public TO "{role}";'))
                for tbl in sorted(existing):
                    if not _PG_IDENT_RE.match(tbl):
                        continue
                    if tbl in INTERNAL_TABLES or USER_TABLE_PREFIX_RE.match(tbl):
                        conn.execute(text(f'REVOKE ALL ON TABLE "{tbl}" FROM "{role}";'))
                    elif tbl in CORE_TABLES:
                        conn.execute(text(f'GRANT SELECT ON TABLE "{tbl}" TO "{role}";'))

            for uid in sorted(tenant_ids):
                tenant_role = _provision_tenant_role_sql(
                    conn,
                    uid,
                    reader_role=role,
                    existing_tables=existing,
                )
                for internal_tbl in sorted(INTERNAL_TABLES & existing):
                    conn.execute(text(f'REVOKE ALL ON TABLE "{internal_tbl}" FROM {tenant_role};'))
                for owned_tbl in tenant_tables_by_owner.get(uid, []):
                    conn.execute(text(f'GRANT SELECT ON TABLE "{owned_tbl}" TO {tenant_role};'))
        return sorted(tenant_ids)
    except SQLAlchemyError as exc:
        logger.warning("Unable to sync read-only / tenant role privileges: %s", exc)
        return []


def ensure_vector_extension(target_engine: Engine | None = None) -> None:
    """Ensure the pgvector extension is enabled when connected to PostgreSQL."""
    eng = target_engine or get_engine()
    if not is_postgres(eng):
        return
    with eng.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))


def is_table_visible_to_user(table_name: str, user_id: int | None = None) -> bool:
    """Return True if table_name is shared demo data or owned by user_id."""
    if table_name in INTERNAL_TABLES:
        return False
    if table_name in CORE_TABLES:
        return True
    if user_id is not None and table_name.startswith(f"u_{user_id}_"):
        return True
    return False


def init_db(target_engine: Engine | None = None, if_exists: str = "replace") -> None:
    """Seed the 3-table relational mock business schema (620 records total)."""
    eng = target_engine or get_engine()
    regions = ["North America", "Europe", "Asia", "South America"]
    customers = pd.DataFrame(
        {
            "customer_id": range(1, 101),
            "company_name": [f"Client_{i} LLC" for i in range(1, 101)],
            "region": [random.choice(regions) for _ in range(100)],
            "signup_date": [
                datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
                for _ in range(100)
            ],
        }
    )

    categories = ["Software", "Hardware", "Consulting", "Support"]
    products = pd.DataFrame(
        {
            "product_id": range(1, 21),
            "product_name": [f"Prod_{i}" for i in range(1, 21)],
            "category": [random.choice(categories) for _ in range(20)],
            "unit_price": [round(random.uniform(50.0, 5000.0), 2) for _ in range(20)],
        }
    )

    order_dates = [
        datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))
        for _ in range(500)
    ]
    orders = pd.DataFrame(
        {
            "order_id": range(1, 501),
            "customer_id": [random.randint(1, 100) for _ in range(500)],
            "product_id": [random.randint(1, 20) for _ in range(500)],
            "order_date": order_dates,
            "quantity": [random.randint(1, 10) for _ in range(500)],
        }
    )

    orders = orders.merge(products[["product_id", "unit_price"]], on="product_id")
    orders["revenue"] = (orders["quantity"] * orders["unit_price"]).round(2)
    orders.drop(columns=["unit_price"], inplace=True)

    with eng.begin() as conn:
        customers.to_sql("customers", conn, if_exists=if_exists, index=False)
        products.to_sql("products", conn, if_exists=if_exists, index=False)
        orders.to_sql("orders", conn, if_exists=if_exists, index=False)
    sync_readonly_role_privileges(eng)


def check_and_seed_db(target_engine: Engine | None = None) -> None:
    """Enable pgvector and idempotently seed core tables if missing or empty."""
    eng = target_engine or get_engine()
    ensure_vector_extension(eng)

    inspector = inspect(eng)
    existing_tables = set(inspector.get_table_names())
    if not CORE_TABLES.issubset(existing_tables):
        init_db(eng, if_exists="replace")
        return

    with eng.connect() as conn:
        for table_name in CORE_TABLES:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
            if count == 0:
                init_db(eng, if_exists="replace")
                return


def get_structured_schema(
    target_engine: Engine | None = None,
    include_internal: bool = False,
    user_id: int | None = None,
    include_all_tenants: bool = False,
) -> list[dict]:
    """Return structured table and column metadata scoped to user_id when provided."""
    eng = target_engine or get_engine()
    inspector = inspect(eng)
    tables: list[dict] = []
    for table_name in sorted(inspector.get_table_names()):
        if not include_internal and table_name in INTERNAL_TABLES:
            continue
        if not include_all_tenants and not is_table_visible_to_user(table_name, user_id=user_id):
            continue
        columns = inspector.get_columns(table_name)
        tables.append(
            {
                "name": table_name,
                "columns": [
                    {"name": col["name"], "type": str(col["type"])} for col in columns
                ],
            }
        )
    return tables


def get_allowed_tables_for_user(
    user_id: int | None = None,
    target_engine: Engine | None = None,
) -> set[str]:
    """Return the exact set of physical tables user_id is authorized to query."""
    return {
        tbl["name"]
        for tbl in get_structured_schema(target_engine=target_engine, user_id=user_id)
    }


def get_schema_for_llm(
    target_engine: Engine | None = None,
    user_id: int | None = None,
) -> str:
    """Format live table and column definitions visible to user_id."""
    tables = get_structured_schema(target_engine=target_engine, user_id=user_id)
    schema_info = []
    for table in tables:
        col_details = [f"{col['name']} ({col['type']})" for col in table["columns"]]
        schema_info.append(f"Table: {table['name']}\nColumns: {', '.join(col_details)}")
    return "\n\n".join(schema_info)
