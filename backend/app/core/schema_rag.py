"""pgvector-backed Schema RAG indexing and tenant-scoped semantic table retrieval."""

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import re
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Engine, Integer, String, Text, inspect, or_, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.types import TypeDecorator

from ..config import settings
from .database import (
    CORE_TABLES,
    ensure_vector_extension,
    get_engine,
    get_structured_schema,
    is_postgres,
)
from .llm_client import generate_embedding

TABLE_DESCRIPTIONS: dict[str, str] = {
    "customers": "B2B customer accounts including customer_id, company_name, region, and signup_date.",
    "products": "Product catalog including product_id, product_name, category (Software, Hardware, Consulting, Support), and unit_price.",
    "orders": "Sales order transactions linking customer_id and product_id with order_date, quantity, and revenue.",
}

_TENANT_TABLE_RE = re.compile(r"^u_(\d+)_(.+)$")


class EmbeddingColumnType(TypeDecorator):
    """Store pgvector Vector on PostgreSQL and JSON float list on SQLite test fallback."""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: list[float] | None, dialect) -> Any:
        if value is None:
            return None
        floats = [float(x) for x in value]
        if dialect.name == "postgresql":
            return floats
        return json.dumps(floats)

    def process_result_value(self, value: Any, dialect) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [float(x) for x in json.loads(value)]
        return [float(x) for x in value]


class SchemaRagBase(DeclarativeBase):
    """Declarative base for Schema RAG metadata tables."""


class SchemaEmbedding(SchemaRagBase):
    """Stores table DDL descriptions, tenant ownership, and vectors for Schema RAG."""

    __tablename__ = "schema_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    owner_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    ddl_text: Mapped[str] = mapped_column(Text, nullable=False)
    ddl_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(EmbeddingColumnType(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )


@dataclass(frozen=True)
class RetrievedTable:
    """A table matched by Schema RAG along with its DDL context and cosine similarity."""

    table_name: str
    ddl_text: str
    similarity: float


def infer_table_owner(table_name: str) -> int | None:
    """Return owning user_id for tenant tables (u_{id}_*) or None for shared tables."""
    if table_name in CORE_TABLES:
        return None
    match = _TENANT_TABLE_RE.match(table_name)
    if match:
        return int(match.group(1))
    return None


def build_table_ddl(table_name: str, columns: list[dict[str, str]]) -> str:
    """Build a compact DDL + semantic description block for embedding and prompting."""
    col_lines = ",\n".join(f"  {col['name']} {col['type']}" for col in columns)
    col_names = ", ".join(col["name"] for col in columns)
    description = TABLE_DESCRIPTIONS.get(
        table_name,
        f"Custom dataset table '{table_name}' containing columns: {col_names}.",
    )
    return f"CREATE TABLE {table_name} (\n{col_lines}\n);\n-- Description: {description}"


def compute_ddl_hash(ddl_text: str, embedding_model: str | None = None) -> str:
    """Compute a deterministic hash covering both DDL content and embedding model."""
    model_name = embedding_model or settings.embedding_model
    payload = f"{model_name}\n{ddl_text}".encode()
    return hashlib.sha256(payload).hexdigest()


def ensure_schema_embeddings_table(target_engine: Engine | None = None) -> None:
    """Create pgvector extension (on PostgreSQL) and schema_embeddings table if absent."""
    eng = target_engine or get_engine()
    ensure_vector_extension(eng)
    SchemaRagBase.metadata.create_all(bind=eng)

    inspector = inspect(eng)
    if "schema_embeddings" in inspector.get_table_names():
        col_names = {col["name"] for col in inspector.get_columns("schema_embeddings")}
        if "owner_id" not in col_names:
            with eng.begin() as conn:
                conn.execute(text("ALTER TABLE schema_embeddings ADD COLUMN owner_id INTEGER;"))
                if is_postgres(eng):
                    conn.execute(
                        text(
                            "CREATE INDEX IF NOT EXISTS ix_schema_embeddings_owner_id "
                            "ON schema_embeddings (owner_id);"
                        )
                    )


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity for offline non-PostgreSQL fallback testing."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def upsert_table_embedding(
    table_name: str,
    owner_id: int | None = None,
    force: bool = False,
    target_engine: Engine | None = None,
) -> bool:
    """Compute and upsert a single table's DDL embedding; returns True if updated."""
    eng = target_engine or get_engine()
    ensure_schema_embeddings_table(eng)

    resolved_owner = owner_id if owner_id is not None else infer_table_owner(table_name)
    tables = {
        tbl["name"]: tbl["columns"]
        for tbl in get_structured_schema(eng, include_all_tenants=True)
    }
    if table_name not in tables:
        raise ValueError(f"Table '{table_name}' does not exist in active schema.")

    ddl_text = build_table_ddl(table_name, tables[table_name])
    ddl_hash = compute_ddl_hash(ddl_text)

    with Session(eng) as session:
        existing = session.scalar(
            select(SchemaEmbedding).where(SchemaEmbedding.table_name == table_name)
        )
        if (
            existing is not None
            and existing.ddl_hash == ddl_hash
            and existing.owner_id == resolved_owner
            and not force
        ):
            return False

        vector = generate_embedding(ddl_text, task_type="RETRIEVAL_DOCUMENT")
        now = datetime.now(UTC)
        if existing is None:
            session.add(
                SchemaEmbedding(
                    table_name=table_name,
                    owner_id=resolved_owner,
                    ddl_text=ddl_text,
                    ddl_hash=ddl_hash,
                    embedding=vector,
                    updated_at=now,
                )
            )
        else:
            existing.owner_id = resolved_owner
            existing.ddl_text = ddl_text
            existing.ddl_hash = ddl_hash
            existing.embedding = vector
            existing.updated_at = now
        session.commit()
        return True


def index_all_tables(
    force: bool = False,
    user_id: int | None = None,
    target_engine: Engine | None = None,
) -> dict[str, Any]:
    """Index business tables into schema_embeddings, scoped to user_id when provided."""
    eng = target_engine or get_engine()
    ensure_schema_embeddings_table(eng)

    if user_id is None:
        active_tables = get_structured_schema(eng, include_all_tenants=True)
    else:
        active_tables = get_structured_schema(eng, user_id=user_id)

    active_names = {tbl["name"] for tbl in active_tables}

    with Session(eng) as session:
        if user_id is None:
            existing_rows = session.scalars(select(SchemaEmbedding)).all()
        else:
            existing_rows = session.scalars(
                select(SchemaEmbedding).where(
                    or_(SchemaEmbedding.owner_id.is_(None), SchemaEmbedding.owner_id == user_id)
                )
            ).all()

        for row in existing_rows:
            if row.table_name not in active_names:
                session.delete(row)
        session.commit()

    indexed_tables: list[str] = []
    skipped_tables: list[str] = []

    for tbl in active_tables:
        name = tbl["name"]
        owner = infer_table_owner(name)
        updated = upsert_table_embedding(name, owner_id=owner, force=force, target_engine=eng)
        if updated:
            indexed_tables.append(name)
        else:
            skipped_tables.append(name)

    return {
        "indexed_tables": indexed_tables,
        "skipped_tables": skipped_tables,
        "total_tables": len(active_tables),
    }


def retrieve_relevant_tables(
    question: str,
    user_id: int | None = None,
    top_k: int | None = None,
    min_similarity: float | None = None,
    target_engine: Engine | None = None,
) -> list[RetrievedTable]:
    """Embed question and return top-k tenant-visible tables above min_similarity."""
    eng = target_engine or get_engine()
    ensure_schema_embeddings_table(eng)

    k = top_k if top_k is not None else settings.schema_rag_top_k
    threshold = min_similarity if min_similarity is not None else settings.schema_rag_min_similarity

    with Session(eng) as session:
        if user_id is None:
            visibility_filter = SchemaEmbedding.owner_id.is_(None)
        else:
            visibility_filter = or_(
                SchemaEmbedding.owner_id.is_(None),
                SchemaEmbedding.owner_id == user_id,
            )
        has_rows = session.scalar(select(SchemaEmbedding.id).where(visibility_filter).limit(1))
    if has_rows is None:
        return []

    query_vec = generate_embedding(question, task_type="RETRIEVAL_QUERY")

    if is_postgres(eng):
        vec_literal = "[" + ",".join(str(float(x)) for x in query_vec) + "]"
        if user_id is None:
            sql = text(
                """
                SELECT table_name,
                       ddl_text,
                       1.0 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
                FROM schema_embeddings
                WHERE owner_id IS NULL
                ORDER BY embedding <=> CAST(:query_vec AS vector)
                LIMIT :top_k
                """
            )
            params: dict[str, Any] = {"query_vec": vec_literal, "top_k": k}
        else:
            sql = text(
                """
                SELECT table_name,
                       ddl_text,
                       1.0 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
                FROM schema_embeddings
                WHERE owner_id IS NULL OR owner_id = :user_id
                ORDER BY embedding <=> CAST(:query_vec AS vector)
                LIMIT :top_k
                """
            )
            params = {"query_vec": vec_literal, "user_id": user_id, "top_k": k}

        with eng.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()

        return [
            RetrievedTable(
                table_name=row["table_name"],
                ddl_text=row["ddl_text"],
                similarity=float(row["similarity"]),
            )
            for row in rows
            if float(row["similarity"]) >= threshold
        ]

    # Non-PostgreSQL fallback for offline local/test runs
    with Session(eng) as session:
        if user_id is None:
            records = session.scalars(
                select(SchemaEmbedding).where(SchemaEmbedding.owner_id.is_(None))
            ).all()
        else:
            records = session.scalars(
                select(SchemaEmbedding).where(
                    or_(
                        SchemaEmbedding.owner_id.is_(None),
                        SchemaEmbedding.owner_id == user_id,
                    )
                )
            ).all()

    scored = [
        RetrievedTable(
            table_name=rec.table_name,
            ddl_text=rec.ddl_text,
            similarity=_cosine_similarity(query_vec, rec.embedding),
        )
        for rec in records
    ]
    scored.sort(key=lambda item: item.similarity, reverse=True)
    return [item for item in scored[:k] if item.similarity >= threshold]
