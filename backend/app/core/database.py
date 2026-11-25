"""PostgreSQL (and test SQLite) engine management, pooling, seeding, and tenant schema scope."""

from datetime import datetime, timedelta
from pathlib import Path
import random
import re

import pandas as pd
from sqlalchemy import Engine, create_engine, event, inspect, text

from ..config import settings

CORE_TABLES = frozenset({"customers", "products", "orders"})
INTERNAL_TABLES = frozenset({"schema_embeddings", "users"})
PROTECTED_TABLES = CORE_TABLES | INTERNAL_TABLES
USER_TABLE_PREFIX_RE = re.compile(r"^u_(\d+)_[a-z0-9_]+$")

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
    """Create pooled PostgreSQL engine or SQLite write/read-only engine pair."""
    if database_url.startswith("sqlite"):
        _ensure_sqlite_parent_dir(database_url)
        connect_args = {"check_same_thread": False}
        write_eng = create_engine(database_url, connect_args=connect_args)
        read_eng = create_engine(database_url, connect_args=connect_args)

        @event.listens_for(read_eng, "connect")
        def _make_read_only(dbapi_connection, connection_record):
            dbapi_connection.execute("PRAGMA query_only = ON")

        return write_eng, read_eng

    pg_eng = create_engine(
        database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
    )
    return pg_eng, pg_eng


def configure_database(database_url: str | None = None) -> tuple[Engine, Engine]:
    """Reconfigure active database engines (used at startup and in isolated tests)."""
    global _engine, _read_engine, _configured_url
    target_url = database_url or settings.database_url
    if _engine is not None:
        _engine.dispose()
    if _read_engine is not None and _read_engine is not _engine:
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
    """Return the active SQLAlchemy read execution engine."""
    if _read_engine is None:
        configure_database(settings.database_url)
    assert _read_engine is not None
    return _read_engine


def is_postgres(target_engine: Engine | None = None) -> bool:
    """Return True when the active engine dialect is PostgreSQL."""
    eng = target_engine or get_engine()
    return eng.dialect.name == "postgresql"


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
