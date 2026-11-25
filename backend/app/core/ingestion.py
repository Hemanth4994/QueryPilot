"""CSV upload ingestion with protected table-name prefixing."""

from io import BytesIO
from pathlib import Path
import re

import pandas as pd

from ..schemas import TableSchema, UploadResponse
from .agent import serialize_dataframe
from .database import PROTECTED_TABLES, get_engine, get_structured_schema


def resolve_table_name(filename: str) -> str:
    """Derive a safe SQLite table name, prefixing protected core tables with user_."""
    stem = Path(filename).stem.replace(" ", "_").lower()
    sanitized = re.sub(r"[^a-z0-9_]", "_", stem).strip("_")
    if not sanitized:
        raise ValueError("Invalid CSV filename: unable to derive table name.")
    if sanitized in PROTECTED_TABLES:
        return f"user_{sanitized}"
    if sanitized[0].isdigit():
        return f"t_{sanitized}"
    return sanitized


def ingest_csv_bytes(content: bytes, filename: str) -> UploadResponse:
    """Parse CSV bytes and persist them to SQLite without overwriting core tables."""
    if not content:
        raise ValueError("Uploaded CSV file is empty.")

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Unable to parse CSV file: {exc}") from exc

    if df.columns.empty:
        raise ValueError("Uploaded CSV contains no columns.")

    table_name = resolve_table_name(filename)
    df.to_sql(name=table_name, con=get_engine(), if_exists="replace", index=False)

    preview = serialize_dataframe(df.head(3))
    schema_tables = [TableSchema(**tbl) for tbl in get_structured_schema()]
    return UploadResponse(
        table_name=table_name,
        row_count=len(df),
        columns=[str(col) for col in df.columns],
        preview=preview,
        tables=schema_tables,
    )
