"""Pydantic v2 request and response models for QueryPilot API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Service, database, and cache readiness payload."""

    status: Literal["ok", "degraded"]
    database_ready: bool
    redis_ready: bool = False
    tables_count: int
    version: str = "4.0.0"


class _EmailAuthBase(BaseModel):
    """Shared normalized email validation for authentication payloads."""

    email: str = Field(..., min_length=3, max_length=255)

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if "@" not in cleaned or "." not in cleaned.split("@", 1)[-1]:
            raise ValueError("Invalid email address.")
        return cleaned


class UserCredentialsRequest(_EmailAuthBase):
    """Email and password payload for new account registration."""

    password: str = Field(..., min_length=8, max_length=128)


class UserLoginRequest(_EmailAuthBase):
    """Email and non-empty password payload for existing account login."""

    password: str = Field(..., min_length=1, max_length=128)


class UserResponse(BaseModel):
    """Public user profile returned by auth endpoints."""

    id: int
    email: str
    created_at: datetime


class AuthTokenResponse(BaseModel):
    """Bearer JWT response returned on register and login."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: UserResponse


class ColumnSchema(BaseModel):
    """Column name and SQL type metadata."""

    name: str
    type: str


class TableSchema(BaseModel):
    """Table schema metadata returned by schema and upload endpoints."""

    name: str
    columns: list[ColumnSchema]


class SchemaResponse(BaseModel):
    """Structured and text representations of active user-visible database tables."""

    tables: list[TableSchema]
    schema_text: str


class ReindexResponse(BaseModel):
    """Summary of an on-demand Schema RAG reindexing run."""

    indexed_tables: list[str]
    skipped_tables: list[str]
    total_tables: int


class QueryRequest(BaseModel):
    """Natural-language analytics question payload."""

    question: str = Field(..., min_length=1, description="Plain-English question to convert to SQL.")

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be empty or whitespace.")
        return cleaned


class ChartSpec(BaseModel):
    """Metadata describing an automatically generated Plotly visualization."""

    chart_type: Literal["bar"] = "bar"
    x_col: str
    y_col: str
    title: str
    template: str = "plotly_dark"


class QueryResponse(BaseModel):
    """Result payload for natural-language SQL generation, execution, and telemetry."""

    request_id: str | None = None
    question: str
    sql_query: str
    status: Literal["SUCCESS", "NO_DATA", "INSUFFICIENT_SCHEMA", "ERROR"]
    rows: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    retrieved_tables: list[str] = Field(default_factory=list)
    truncated: bool = False
    cache_hit: bool = False
    chart_spec: ChartSpec | None = None
    plotly_json: dict[str, Any] | None = None
    error: str | None = None
    message: str | None = None
    model: str | None = None
    latency_ms: float
    corrected: bool = False
    self_correction_attempts: int = 0
    fallback_triggered: bool = False
    sql_guard_status: str = "passed"
    input_tokens: int = 0
    output_tokens: int = 0


class UploadResponse(BaseModel):
    """Response returned after synchronously ingesting a custom CSV table."""

    table_name: str
    row_count: int
    columns: list[str]
    preview: list[dict[str, Any]]
    tables: list[TableSchema]
    embedding_indexed: bool = True


class JobSubmittedResponse(BaseModel):
    """202 Accepted payload returned when a query or CSV upload is queued."""

    job_id: str
    job_type: Literal["query", "upload"]
    status: Literal["queued"] = "queued"


class JobStatusResponse(BaseModel):
    """Tenant-scoped background job state and final payload."""

    job_id: str
    job_type: Literal["query", "upload"]
    status: Literal["queued", "running", "completed", "failed"]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class MetricsResponse(BaseModel):
    """Live telemetry counters exposed by GET /api/v1/metrics."""

    total_queries: int
    cache_hits: int
    cache_hit_rate: float
    self_corrections: int
    self_correction_rate: float
    fallback_count: int
    guard_rejections: int
    error_count: int
    total_input_tokens: int
    total_output_tokens: int


class ErrorResponse(BaseModel):
    """Structured error response body."""

    detail: str
    error_code: str
