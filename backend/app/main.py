"""FastAPI application entrypoint with lifespan initialization, auth, Redis, and CORS."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router as v1_router
from .config import settings
from .core.auth import ensure_users_table
from .core.cache import close_redis_client
from .core.database import check_and_seed_db, configure_database
from .core.schema_rag import ensure_schema_embeddings_table, index_all_tables
from .schemas import ErrorResponse
from .worker import close_arq_pool

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Configure DB pool, seed demo tables, ensure auth/vector tables, and index schema."""
    configure_database(settings.database_url)
    check_and_seed_db()
    ensure_users_table()
    ensure_schema_embeddings_table()
    try:
        index_all_tables(force=False)
    except RuntimeError as exc:
        if "GEMINI_API_KEY" in str(exc):
            logger.warning("Skipping startup Schema RAG indexing: %s", exc)
        else:
            raise
    yield
    await close_arq_pool()
    await close_redis_client()


app = FastAPI(
    title="QueryPilot API",
    description="Text-to-SQL BI Backend (PostgreSQL + pgvector Schema RAG + Auth + Redis + ARQ)",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    """Return HTTP 400 with structured payload for invalid inputs."""
    payload = ErrorResponse(detail=str(exc), error_code="BAD_REQUEST")
    return JSONResponse(status_code=400, content=payload.model_dump())


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_request: Request, exc: RuntimeError) -> JSONResponse:
    """Return HTTP 503 when upstream LLM/queue/config dependencies are unavailable."""
    payload = ErrorResponse(detail=str(exc), error_code="SERVICE_UNAVAILABLE")
    return JSONResponse(status_code=503, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Return structured HTTP 500 response for unexpected server failures."""
    payload = ErrorResponse(detail="Internal server error.", error_code="INTERNAL_ERROR")
    return JSONResponse(status_code=500, content=payload.model_dump())


app.include_router(v1_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.port)
