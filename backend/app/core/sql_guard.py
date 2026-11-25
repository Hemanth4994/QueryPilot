"""Deterministic read-only SQL guard and safe query execution."""

import pandas as pd
from sqlalchemy import text

from .database import get_read_engine


def validate_read_only_sql(query: str) -> str | None:
    """Return an error string if query is not a single read-only SELECT/WITH statement."""
    stripped = query.strip()
    if not stripped:
        return "Security Error: Only read-only SELECT/WITH queries are permitted."

    clean_query = stripped.upper()
    if not (clean_query.startswith("SELECT") or clean_query.startswith("WITH")):
        return "Security Error: Only read-only SELECT/WITH queries are permitted."

    # Block multi-statement injection via semicolons
    if ";" in stripped[:-1]:
        return "Security Error: Multiple SQL statements are not allowed."

    return None


def execute_sql(query: str) -> tuple[pd.DataFrame | None, str | None]:
    """Validate and execute read-only SELECT/WITH SQL against the SQLite database."""
    security_error = validate_read_only_sql(query)
    if security_error is not None:
        return None, security_error

    try:
        with get_read_engine().connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df, None
    except Exception as exc:
        return None, str(exc)
