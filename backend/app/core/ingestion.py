"""CSV upload ingestion with tenant-scoped table naming and Schema RAG upsert."""

from io import BytesIO
import logging
from pathlib import Path
import re

import pandas as pd

from ..schemas import TableSchema, UploadResponse
from .agent import serialize_dataframe
from .database import PROTECTED_TABLES, get_engine, get_structured_schema, grant_readonly_select
from .schema_rag import upsert_table_embedding

logger = logging.getLogger(__name__)


def _read_csv_with_fallback(content: bytes) -> pd.DataFrame:
    """Parse CSV bytes trying common encodings before raising ValueError."""
    last_exc: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_exc = exc
        except Exception as exc:
            raise ValueError(f"Unable to parse CSV file: {exc}") from exc
    raise ValueError(f"Unable to decode CSV file encoding: {last_exc}")


def resolve_table_name(filename: str, user_id: int | None = None) -> str:
    """Derive a safe table name scoped to user_id (or user_-prefixed for legacy calls)."""
    stem = Path(filename).stem.replace(" ", "_").lower()
    sanitized = re.sub(r"[^a-z0-9_]", "_", stem).strip("_")
    if not sanitized:
        raise ValueError("Invalid CSV filename: unable to derive table name.")

    if user_id is not None:
        return f"u_{user_id}_{sanitized}"

    if sanitized in PROTECTED_TABLES:
        return f"user_{sanitized}"
    if sanitized[0].isdigit():
        return f"t_{sanitized}"
    return sanitized


def ingest_csv_bytes(
    content: bytes,
    filename: str,
    user_id: int | None = None,
) -> UploadResponse:
    """Parse CSV bytes, persist to tenant-scoped table, and upsert Schema RAG embedding."""
    if not content:
        raise ValueError("Uploaded CSV file is empty.")

    df = _read_csv_with_fallback(content)

    if df.columns.empty:
        raise ValueError("Uploaded CSV contains no columns.")
    if df.empty:
        raise ValueError("Uploaded CSV contains no data rows.")
    if any(not str(col).strip() for col in df.columns):
        raise ValueError("Uploaded CSV contains blank column headers.")

    table_name = resolve_table_name(filename, user_id=user_id)
    df.to_sql(name=table_name, con=get_engine(), if_exists="replace", index=False)
    grant_readonly_select(table_name)

    try:
        embedding_indexed = upsert_table_embedding(table_name, owner_id=user_id, force=True)
    except RuntimeError as exc:
        logger.warning(
            "CSV table '%s' saved, but Schema RAG indexing failed: %s",
            table_name,
            exc,
        )
        embedding_indexed = False

    preview = serialize_dataframe(df.head(3))
    schema_tables = [
        TableSchema(**tbl)
        for tbl in get_structured_schema(user_id=user_id)
    ]
    return UploadResponse(
        table_name=table_name,
        row_count=len(df),
        columns=[str(col) for col in df.columns],
        preview=preview,
        tables=schema_tables,
        embedding_indexed=embedding_indexed,
    )
