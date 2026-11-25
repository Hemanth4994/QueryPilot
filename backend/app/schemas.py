"""Pydantic v2 request and response models for QueryPilot API."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Service and database readiness payload."""

    status: Literal["ok", "degraded"]
    database_ready: bool
    tables_count: int
    version: str = "1.0.0"


class ColumnSchema(BaseModel):
    """Column name and SQLite type metadata."""

    name: str
    type: str


class TableSchema(BaseModel):
    """Table schema metadata returned by schema and upload endpoints."""

    name: str
    columns: list[ColumnSchema]


class SchemaResponse(BaseModel):
    """Structured and text representations of active database tables."""

    tables: list[TableSchema]
    schema_text: str


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
    """Result payload for natural-language SQL generation and execution."""

    question: str
    sql_query: str
    status: Literal["SUCCESS", "NO_DATA", "INSUFFICIENT_SCHEMA", "ERROR"]
    rows: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    chart_spec: ChartSpec | None = None
    plotly_json: dict[str, Any] | None = None
    error: str | None = None
    message: str | None = None
    model: str | None = None
    latency_ms: float
    corrected: bool = False


class UploadResponse(BaseModel):
    """Response returned after ingesting a custom CSV table."""

    table_name: str
    row_count: int
    columns: list[str]
    preview: list[dict[str, Any]]
    tables: list[TableSchema]


class ErrorResponse(BaseModel):
    """Structured error response body."""

    detail: str
    error_code: str
