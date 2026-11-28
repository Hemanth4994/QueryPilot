"""Streaming CSV upload ingestion (up to 2 GB) via PostgreSQL COPY, auto-indexing, and Schema RAG."""

import codecs
import hashlib
import logging
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid

import pandas as pd
from sqlalchemy import Engine

from ..schemas import TableSchema, UploadResponse
from .agent import serialize_dataframe
from .database import (
    PROTECTED_TABLES,
    get_engine,
    get_structured_schema,
    grant_readonly_select,
    is_postgres,
)
from .schema_rag import upsert_table_embedding

logger = logging.getLogger(__name__)

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_SAMPLE_ROWS = 5000
_STREAM_CHUNK_BYTES = 1024 * 1024
_SQLITE_CHUNK_ROWS = 50000
_MAX_AUTO_INDEXES = 4
_DATE_HINT_RE = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
_PG_COPY_COL_ERR_RE = re.compile(r'column\s+"?([^":\n]+?)"?\s*:', re.IGNORECASE)


def _quote_ident(name: str) -> str:
    """Safely double-quote a PostgreSQL identifier after null-byte validation."""
    if "\x00" in name:
        raise ValueError("Invalid identifier containing null byte.")
    return '"' + name.replace('"', '""') + '"'


def _detect_csv_encoding(file_path: Path) -> str:
    """Detect CSV text encoding from a 64 KB sample without loading the whole file."""
    with file_path.open("rb") as fh:
        sample = fh.read(65536)
    if not sample:
        raise ValueError("Uploaded CSV file is empty.")
    if sample.startswith(codecs.BOM_UTF8):
        try:
            sample.decode("utf-8-sig")
            return "utf-8-sig"
        except UnicodeDecodeError:
            pass
    last_exc: Exception | None = None
    for encoding in _ENCODINGS:
        try:
            codecs.getincrementaldecoder(encoding)(errors="strict").decode(sample, final=False)
            return encoding
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise ValueError(f"Unable to decode CSV file encoding: {last_exc}")


def resolve_table_name(filename: str, user_id: int | None = None) -> str:
    """Derive a PostgreSQL-safe table name scoped to user_id."""
    stem = Path(filename).stem.replace(" ", "_").lower()
    sanitized = re.sub(r"[^a-z0-9_]", "_", stem).strip("_")
    if not sanitized:
        raise ValueError("Invalid CSV filename: unable to derive table name.")

    if user_id is not None:
        prefix = f"u_{user_id}_"
        max_stem = max(12, 58 - len(prefix))
        if len(sanitized) > max_stem:
            digest = hashlib.sha1(sanitized.encode("utf-8")).hexdigest()[:6]
            sanitized = f"{sanitized[: max_stem - 7].rstrip('_')}_{digest}"
        return f"{prefix}{sanitized}"

    if sanitized in PROTECTED_TABLES:
        return f"user_{sanitized}"
    if sanitized[0].isdigit():
        return f"t_{sanitized}"
    return sanitized[:58]


def _infer_pg_column_types(df_sample: pd.DataFrame) -> dict[str, str]:
    """Infer safe PostgreSQL column types from a sample DataFrame."""
    types_map: dict[str, str] = {}
    for col in df_sample.columns:
        col_name = str(col)
        series = df_sample[col]
        if pd.api.types.is_bool_dtype(series):
            types_map[col_name] = "BOOLEAN"
        elif pd.api.types.is_integer_dtype(series):
            types_map[col_name] = "BIGINT"
        elif pd.api.types.is_float_dtype(series):
            types_map[col_name] = "DOUBLE PRECISION"
        elif pd.api.types.is_datetime64_any_dtype(series):
            types_map[col_name] = "TIMESTAMP"
        else:
            non_null = series.dropna().astype(str).str.strip()
            non_null = non_null[non_null != ""]
            if not non_null.empty and non_null.str.match(_DATE_HINT_RE).mean() >= 0.95:
                parsed = pd.to_datetime(non_null, format="mixed", errors="coerce")
                if parsed.notna().mean() >= 0.95:
                    types_map[col_name] = "TIMESTAMP"
                    continue
            types_map[col_name] = "TEXT"
    return types_map


def _select_index_columns(df_sample: pd.DataFrame, col_types: dict[str, str]) -> list[str]:
    """Pick up to 4 high-leverage filter/join/group columns for fast analytical queries."""
    id_cols: list[str] = []
    date_cols: list[str] = []
    cat_cols: list[str] = []

    for col in df_sample.columns:
        col_name = str(col)
        lowered = col_name.lower()
        pg_type = col_types.get(col_name, "TEXT")
        if lowered == "id" or lowered.endswith("_id"):
            id_cols.append(col_name)
        elif pg_type == "TIMESTAMP" or "date" in lowered or "time" in lowered:
            date_cols.append(col_name)
        elif pg_type == "TEXT":
            non_null = df_sample[col].dropna().astype(str)
            total = len(non_null)
            nunique = int(non_null.nunique())
            max_len = int(non_null.str.len().max()) if total else 0
            if 2 <= nunique <= 50 and max_len <= 64 and total > nunique:
                cat_cols.append(col_name)

    selected: list[str] = []
    for candidate in (*id_cols, *date_cols, *cat_cols):
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= _MAX_AUTO_INDEXES:
            break
    return selected


def _build_index_name(table_name: str, idx: int, col_name: str) -> str:
    """Build a deterministic PostgreSQL index name strictly within 63 bytes."""
    digest = hashlib.sha1(f"{table_name}:{col_name}".encode()).hexdigest()[:6]
    return f"ix_{table_name[:42]}_{idx}_{digest}"


def _stream_utf8_to_copy(file_path: Path, encoding: str, copy: Any) -> None:
    """Stream any supported CSV encoding into psycopg COPY as clean UTF-8 chunks."""
    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    first_chunk = True
    with file_path.open("rb") as fh:
        while raw := fh.read(_STREAM_CHUNK_BYTES):
            text_chunk = decoder.decode(raw, final=False)
            if first_chunk:
                text_chunk = text_chunk.lstrip("\ufeff")
                first_chunk = False
            if text_chunk:
                copy.write(text_chunk.encode("utf-8"))
        tail = decoder.decode(b"", final=True)
        if first_chunk:
            tail = tail.lstrip("\ufeff")
        if tail:
            copy.write(tail.encode("utf-8"))


def _ingest_postgres_copy(
    file_path: Path,
    encoding: str,
    table_name: str,
    columns: list[str],
    df_sample: pd.DataFrame,
    eng: Engine,
) -> tuple[int, list[str], str]:
    """Stream CSV into a staging table via PostgreSQL COPY, build indexes, ANALYZE, and swap."""
    col_types = _infer_pg_column_types(df_sample)
    cols_sql = ", ".join(_quote_ident(col) for col in columns)
    staging_name = f"_stg_{table_name[:42]}_{uuid.uuid4().hex[:8]}"

    for attempt in range(3):
        col_defs = ", ".join(
            f"{_quote_ident(col)} {col_types.get(col, 'TEXT')}" for col in columns
        )
        index_cols = _select_index_columns(df_sample, col_types)
        raw_conn = eng.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {_quote_ident(staging_name)};")
                cur.execute(f"CREATE TABLE {_quote_ident(staging_name)} ({col_defs});")
                copy_sql = (
                    f"COPY {_quote_ident(staging_name)} ({cols_sql}) "
                    "FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
                )
                with cur.copy(copy_sql) as copy:
                    _stream_utf8_to_copy(file_path, encoding, copy)

                row_count = int(cur.rowcount) if cur.rowcount is not None and cur.rowcount >= 0 else -1
                if row_count < 0:
                    cur.execute(f"SELECT COUNT(*) FROM {_quote_ident(staging_name)};")
                    row_count = int(cur.fetchone()[0])

                cur.execute(f"DROP TABLE IF EXISTS {_quote_ident(table_name)};")
                cur.execute(
                    f"ALTER TABLE {_quote_ident(staging_name)} RENAME TO {_quote_ident(table_name)};"
                )
                for idx_i, col in enumerate(index_cols):
                    ix_name = _build_index_name(table_name, idx_i, col)
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {_quote_ident(ix_name)} "
                        f"ON {_quote_ident(table_name)} ({_quote_ident(col)});"
                    )
                cur.execute(f"ANALYZE {_quote_ident(table_name)};")
            raw_conn.commit()
            return row_count, index_cols, "postgres_copy"
        except Exception as exc:
            raw_conn.rollback()
            err_msg = str(exc)
            match = _PG_COPY_COL_ERR_RE.search(err_msg)
            widened = False
            if match and match.group(1) in col_types and col_types[match.group(1)] != "TEXT":
                bad_col = match.group(1)
                logger.warning(
                    "Widening column '%s' of '%s' to TEXT after COPY type mismatch: %s",
                    bad_col,
                    table_name,
                    err_msg,
                )
                col_types[bad_col] = "TEXT"
                widened = True
            elif attempt == 0 and any(t != "TEXT" for t in col_types.values()) and (
                "invalid input syntax" in err_msg.lower() or "out of range" in err_msg.lower()
            ):
                logger.warning(
                    "Widening non-TEXT columns of '%s' to TEXT after COPY data error: %s",
                    table_name,
                    err_msg,
                )
                for k in list(col_types.keys()):
                    col_types[k] = "TEXT"
                widened = True

            if widened and attempt < 2:
                continue
            raise ValueError(f"Unable to ingest CSV via PostgreSQL COPY: {exc}") from exc
        finally:
            raw_conn.close()

    raise ValueError("Unable to ingest CSV into PostgreSQL.")


def _ingest_sqlite_chunked(
    file_path: Path,
    encoding: str,
    table_name: str,
    eng: Engine,
) -> tuple[int, list[str], str]:
    """Stream CSV chunks into SQLite for offline tests and lightweight local runs."""
    total_rows = 0
    first = True
    for chunk in pd.read_csv(file_path, encoding=encoding, chunksize=_SQLITE_CHUNK_ROWS, low_memory=False):
        mode = "replace" if first else "append"
        chunk.to_sql(name=table_name, con=eng, if_exists=mode, index=False)
        total_rows += len(chunk)
        first = False
    return total_rows, [], "sqlite_chunked"


def ingest_csv_file(
    file_path: Path,
    filename: str,
    user_id: int | None = None,
) -> UploadResponse:
    """Stream CSV from disk into PostgreSQL/SQLite, index columns, and upsert Schema RAG."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise ValueError("Uploaded CSV file is empty.")

    encoding = _detect_csv_encoding(file_path)
    try:
        df_sample = pd.read_csv(file_path, encoding=encoding, nrows=_SAMPLE_ROWS, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Unable to parse CSV file: {exc}") from exc

    if df_sample.columns.empty:
        raise ValueError("Uploaded CSV contains no columns.")
    if df_sample.empty:
        raise ValueError("Uploaded CSV contains no data rows.")
    columns = [str(col) for col in df_sample.columns]
    if any(not col.strip() for col in columns):
        raise ValueError("Uploaded CSV contains blank column headers.")

    table_name = resolve_table_name(filename, user_id=user_id)
    eng = get_engine()
    if is_postgres(eng):
        row_count, indexed_cols, ingestion_mode = _ingest_postgres_copy(
            file_path=file_path,
            encoding=encoding,
            table_name=table_name,
            columns=columns,
            df_sample=df_sample,
            eng=eng,
        )
    else:
        row_count, indexed_cols, ingestion_mode = _ingest_sqlite_chunked(
            file_path=file_path,
            encoding=encoding,
            table_name=table_name,
            eng=eng,
        )

    grant_readonly_select(table_name, user_id=user_id)

    try:
        embedding_indexed = upsert_table_embedding(table_name, owner_id=user_id, force=True)
    except RuntimeError as exc:
        logger.warning(
            "CSV table '%s' saved, but Schema RAG indexing failed: %s",
            table_name,
            exc,
        )
        embedding_indexed = False

    preview = serialize_dataframe(df_sample.head(3))
    schema_tables = [
        TableSchema(**tbl)
        for tbl in get_structured_schema(user_id=user_id)
    ]
    return UploadResponse(
        table_name=table_name,
        row_count=row_count,
        columns=columns,
        preview=preview,
        tables=schema_tables,
        embedding_indexed=embedding_indexed,
        indexed_columns=indexed_cols,
        ingestion_mode=ingestion_mode,
    )


def ingest_csv_bytes(
    content: bytes,
    filename: str,
    user_id: int | None = None,
) -> UploadResponse:
    """Persist in-memory CSV bytes via streaming ingestion (preserving legacy API)."""
    if not content:
        raise ValueError("Uploaded CSV file is empty.")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        return ingest_csv_file(tmp_path, filename=filename, user_id=user_id)
    finally:
        tmp_path.unlink(missing_ok=True)
