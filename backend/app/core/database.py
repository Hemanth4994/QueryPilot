"""SQLite engine management, seeding, and dynamic schema introspection."""

from datetime import datetime, timedelta
from pathlib import Path
import random

import pandas as pd
from sqlalchemy import Engine, create_engine, event, inspect

from ..config import settings

PROTECTED_TABLES = frozenset({"customers", "products", "orders"})

_engine: Engine | None = None
_read_engine: Engine | None = None
_configured_url: str | None = None


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    """Create parent directory for file-backed SQLite URLs when missing."""
    if not database_url.startswith("sqlite:///"):
        return
    raw_path = database_url.removeprefix("sqlite:///")
    if not raw_path or raw_path == ":memory:":
        return
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)


def _create_pair(database_url: str) -> tuple[Engine, Engine]:
    """Create write engine and PRAGMA query_only read engine for a database URL."""
    _ensure_sqlite_parent_dir(database_url)
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}

    write_eng = create_engine(database_url, connect_args=connect_args)
    read_eng = create_engine(database_url, connect_args=connect_args)

    if database_url.startswith("sqlite"):
        # Separate engine for running generated SQL: SQLite itself refuses every write on it,
        # so a statement the text check misses (e.g. WITH ... DELETE) still cannot change data.
        @event.listens_for(read_eng, "connect")
        def _make_read_only(dbapi_connection, connection_record):
            dbapi_connection.execute("PRAGMA query_only = ON")

    return write_eng, read_eng


def configure_database(database_url: str | None = None) -> tuple[Engine, Engine]:
    """Reconfigure active database engines (used at startup and in isolated tests)."""
    global _engine, _read_engine, _configured_url
    target_url = database_url or settings.database_url
    if _engine is not None:
        _engine.dispose()
    if _read_engine is not None:
        _read_engine.dispose()
    _engine, _read_engine = _create_pair(target_url)
    _configured_url = target_url
    return _engine, _read_engine


def get_engine() -> Engine:
    """Return the active SQLAlchemy write engine."""
    if _engine is None:
        configure_database(settings.database_url)
    assert _engine is not None
    return _engine


def get_read_engine() -> Engine:
    """Return the active SQLAlchemy read-only engine."""
    if _read_engine is None:
        configure_database(settings.database_url)
    assert _read_engine is not None
    return _read_engine


def init_db(target_engine: Engine | None = None) -> None:
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
    orders["revenue"] = orders["quantity"] * orders["unit_price"]
    orders.drop(columns=["unit_price"], inplace=True)

    with eng.begin() as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        products.to_sql("products", conn, if_exists="replace", index=False)
        orders.to_sql("orders", conn, if_exists="replace", index=False)


def check_and_seed_db(target_engine: Engine | None = None) -> None:
    """Validate database existence and seed core tables if missing."""
    eng = target_engine or get_engine()
    inspector = inspect(eng)
    existing_tables = set(inspector.get_table_names())
    if not PROTECTED_TABLES.issubset(existing_tables):
        init_db(eng)


def get_structured_schema(target_engine: Engine | None = None) -> list[dict]:
    """Return structured table and column metadata for API responses."""
    eng = target_engine or get_engine()
    inspector = inspect(eng)
    tables: list[dict] = []
    for table_name in inspector.get_table_names():
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


def get_schema_for_llm(target_engine: Engine | None = None) -> str:
    """Format live table and column definitions for the Text-to-SQL prompt."""
    tables = get_structured_schema(target_engine=target_engine)
    schema_info = []
    for table in tables:
        col_details = [f"{col['name']} ({col['type']})" for col in table["columns"]]
        schema_info.append(f"Table: {table['name']}\nColumns: {', '.join(col_details)}")
    return "\n\n".join(schema_info)
