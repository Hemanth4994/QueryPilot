"""Versioned v1 FastAPI routes for QueryPilot."""

from fastapi import APIRouter, File, UploadFile

from ..core.agent import run_query_pipeline
from ..core.database import get_schema_for_llm, get_structured_schema
from ..core.ingestion import ingest_csv_bytes
from ..schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SchemaResponse,
    TableSchema,
    UploadResponse,
)

router = APIRouter(prefix="/api/v1", tags=["querypilot-v1"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Verify service health and inspect active database table readiness."""
    tables = get_structured_schema()
    return HealthResponse(
        status="ok",
        database_ready=len(tables) >= 3,
        tables_count=len(tables),
    )


@router.get("/schema", response_model=SchemaResponse)
def get_schema() -> SchemaResponse:
    """Return active SQLite tables and columns."""
    tables = [TableSchema(**tbl) for tbl in get_structured_schema()]
    return SchemaResponse(tables=tables, schema_text=get_schema_for_llm())


@router.post("/query", response_model=QueryResponse)
def query_data(payload: QueryRequest) -> QueryResponse:
    """Translate a natural-language question to read-only SQL, execute, and chart."""
    return run_query_pipeline(payload.question)


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a CSV file into SQLite with protected core-table prefixing."""
    filename = file.filename or "uploaded.csv"
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only .csv files are supported.")
    content = await file.read()
    return ingest_csv_bytes(content=content, filename=filename)
