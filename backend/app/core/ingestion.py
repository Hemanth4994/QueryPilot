"""CSV upload ingestion with tenant-scoped table naming and Schema RAG upsert."""

from io import BytesIO
from pathlib import Path
import re

import pandas as pd

from ..schemas import TableSchema, UploadResponse
from .agent import serialize_dataframe
from .database import PROTECTED_TABLES, get_engine, get_structured_schema
from .schema_rag import upsert_table_embedding


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

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Unable to parse CSV file: {exc}") from exc

    if df.columns.empty:
        raise ValueError("Uploaded CSV contains no columns.")

    table_name = resolve_table_name(filename, user_id=user_id)
    df.to_sql(name=table_name, con=get_engine(), if_exists="replace", index=False)

    embedding_indexed = upsert_table_embedding(table_name, owner_id=user_id, force=True)

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
