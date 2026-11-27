"""Versioned v1 FastAPI routes with JWT auth, Redis cache, rate limit, ARQ, and metrics."""

import logging
from pathlib import Path
import time
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..core.agent import run_query_pipeline
from ..core.auth import (
    User,
    authenticate_user,
    create_access_token,
    get_current_user,
    register_user,
)
from ..core.cache import (
    check_redis_health,
    compute_schema_hash,
    enforce_rate_limit,
    get_cached_query,
    get_cached_schema,
    get_job_state,
    invalidate_user_cache,
    save_job_state,
    set_cached_query,
    set_cached_schema,
)
from ..core.database import get_engine, get_schema_for_llm, get_structured_schema
from ..core.ingestion import ingest_csv_file
from ..core.schema_rag import index_all_tables
from ..core.telemetry import (
    QueryTelemetryEvent,
    emit_query_telemetry,
    metrics_registry,
)
from ..schemas import (
    AuthTokenResponse,
    HealthResponse,
    JobStatusResponse,
    JobSubmittedResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    ReindexResponse,
    SchemaResponse,
    TableSchema,
    UploadResponse,
    UserCredentialsRequest,
    UserLoginRequest,
    UserResponse,
)
from ..worker import close_arq_pool, get_arq_pool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["querypilot-v1"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Verify PostgreSQL and Redis readiness without crashing when degraded."""
    def _db_probe() -> tuple[bool, int]:
        try:
            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            tables = get_structured_schema(user_id=None)
            return len(tables) >= 3, len(tables)
        except Exception as exc:
            logger.warning("Database health probe failed (degraded): %s", exc)
            return False, 0

    database_ready, tables_count = await run_in_threadpool(_db_probe)
    redis_ready = await check_redis_health()
    overall = "ok" if (database_ready and redis_ready) else "degraded"
    return HealthResponse(
        status=overall,
        database_ready=database_ready,
        redis_ready=redis_ready,
        tables_count=tables_count,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(current_user: User = Depends(get_current_user)) -> MetricsResponse:
    """Return live telemetry counters for authenticated monitoring."""
    _ = current_user
    return MetricsResponse(**metrics_registry.snapshot())


@router.post("/auth/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCredentialsRequest) -> AuthTokenResponse:
    """Register a new user account and issue a Bearer JWT access token."""
    user = await run_in_threadpool(register_user, payload.email, payload.password)
    token, expires_in = create_access_token(user.id, user.email)
    return AuthTokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse(id=user.id, email=user.email, created_at=user.created_at),
    )


@router.post("/auth/login", response_model=AuthTokenResponse)
async def login(payload: UserLoginRequest) -> AuthTokenResponse:
    """Authenticate credentials and issue a Bearer JWT access token."""
    user = await run_in_threadpool(authenticate_user, payload.email, payload.password)
    token, expires_in = create_access_token(user.id, user.email)
    return AuthTokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse(id=user.id, email=user.email, created_at=user.created_at),
    )


@router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Return the currently authenticated user profile."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at,
    )


@router.get("/schema", response_model=SchemaResponse)
async def get_schema(current_user: User = Depends(get_current_user)) -> SchemaResponse:
    """Return tenant-visible tables and columns, backed by per-user Redis caching."""
    cached = await get_cached_schema(current_user.id)
    if cached is not None:
        return SchemaResponse(**cached)

    def _load_schema() -> dict:
        tables = [TableSchema(**tbl) for tbl in get_structured_schema(user_id=current_user.id)]
        schema_text = get_schema_for_llm(user_id=current_user.id)
        return SchemaResponse(tables=tables, schema_text=schema_text).model_dump(mode="json")

    payload = await run_in_threadpool(_load_schema)
    await set_cached_schema(current_user.id, payload)
    return SchemaResponse(**payload)


@router.post("/schema/reindex", response_model=ReindexResponse)
async def reindex_schema(current_user: User = Depends(get_current_user)) -> ReindexResponse:
    """Rebuild Schema RAG embeddings for shared and current user's tables and clear cache."""
    summary = await run_in_threadpool(index_all_tables, True, current_user.id)
    await invalidate_user_cache(current_user.id)
    return ReindexResponse(**summary)


@router.post("/query", response_model=QueryResponse)
async def query_data(
    payload: QueryRequest,
    current_user: User = Depends(get_current_user),
) -> QueryResponse:
    """Rate-limit, check per-user Redis query cache, or execute tenant-scoped Schema RAG."""
    request_id = uuid.uuid4().hex
    req_start = time.perf_counter()
    await enforce_rate_limit(current_user.id, scope="api")

    schema_text = await run_in_threadpool(get_schema_for_llm, None, current_user.id)
    schema_hash = compute_schema_hash(schema_text)

    cached = await get_cached_query(current_user.id, schema_hash, payload.question)
    if cached is not None:
        cached["request_id"] = request_id
        cached["latency_ms"] = round((time.perf_counter() - req_start) * 1000, 2)
        cached_resp = QueryResponse(**cached)
        emit_query_telemetry(
            QueryTelemetryEvent(
                request_id=request_id,
                user_id=current_user.id,
                model_used=cached_resp.model,
                fallback_triggered=cached_resp.fallback_triggered,
                self_correction_attempts=cached_resp.self_correction_attempts,
                cache_hit=True,
                retrieved_tables=cached_resp.retrieved_tables,
                sql_guard_status=cached_resp.sql_guard_status,
                latency_ms=cached_resp.latency_ms,
                input_tokens=0,
                output_tokens=0,
            )
        )
        return cached_resp

    result = await run_in_threadpool(
        run_query_pipeline,
        payload.question,
        current_user.id,
        request_id,
    )
    await set_cached_query(
        current_user.id,
        schema_hash,
        payload.question,
        result.model_dump(mode="json"),
    )
    return result


async def _enqueue_worker_job(
    user_id: int,
    job_type: str,
    func_name: str,
    *job_args: object,
) -> str:
    """Persist initial job state and enqueue ARQ work, failing cleanly with 503 if Redis dies."""
    pool = await get_arq_pool()
    job_id = uuid.uuid4().hex
    await save_job_state(
        job_id=job_id,
        user_id=user_id,
        job_type=job_type,
        job_status="queued",
    )
    try:
        await pool.enqueue_job(func_name, job_id, user_id, *job_args)
    except (RedisError, OSError) as exc:
        await close_arq_pool()
        try:
            await save_job_state(
                job_id=job_id,
                user_id=user_id,
                job_type=job_type,
                job_status="failed",
                error=f"Queue connection lost while submitting job: {exc}",
            )
        except RuntimeError:
            pass
        raise RuntimeError(f"Background job queue unavailable: {exc}") from exc
    return job_id


@router.post(
    "/query/async",
    response_model=JobSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def query_data_async(
    payload: QueryRequest,
    current_user: User = Depends(get_current_user),
) -> JobSubmittedResponse:
    """Enqueue an analytical query on the ARQ Redis worker queue."""
    await enforce_rate_limit(current_user.id, scope="api")
    job_id = await _enqueue_worker_job(
        current_user.id,
        "query",
        "run_async_query_job",
        payload.question,
    )
    return JobSubmittedResponse(job_id=job_id, job_type="query", status="queued")


_UPLOAD_STREAM_CHUNK_BYTES = 1024 * 1024


async def _stream_upload_to_disk(file: UploadFile) -> tuple[Path, int]:
    """Stream multipart CSV upload directly to shared disk while enforcing MAX_UPLOAD_SIZE_BYTES."""
    upload_root = Path(settings.upload_dir)
    upload_root.mkdir(parents=True, exist_ok=True)
    staged_path = upload_root / f"{uuid.uuid4().hex}.csv"
    total_bytes = 0
    max_bytes = int(settings.max_upload_size_bytes)

    try:
        with staged_path.open("wb") as out:
            while chunk := await file.read(_UPLOAD_STREAM_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Uploaded CSV exceeds maximum limit ({max_bytes // (1024 * 1024)} MB).",
                    )
                out.write(chunk)
    except Exception:
        staged_path.unlink(missing_ok=True)
        raise

    if total_bytes == 0:
        staged_path.unlink(missing_ok=True)
        raise ValueError("Uploaded CSV file is empty.")
    return staged_path, total_bytes


@router.post(
    "/upload",
    response_model=UploadResponse | JobSubmittedResponse,
)
async def upload_csv(
    file: UploadFile = File(...),
    async_mode: bool = Query(default=False, description="Enqueue CSV ingestion in worker."),
    current_user: User = Depends(get_current_user),
):
    """Stream tenant-scoped CSV (up to 2 GB) to disk and ingest via Postgres COPY or worker."""
    await enforce_rate_limit(current_user.id, scope="api")

    filename = file.filename or "uploaded.csv"
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only .csv files are supported.")
    staged_path, total_bytes = await _stream_upload_to_disk(file)

    if async_mode or total_bytes > settings.async_upload_threshold_bytes:
        try:
            job_id = await _enqueue_worker_job(
                current_user.id,
                "upload",
                "run_async_upload_job",
                filename,
                str(staged_path),
                True,
            )
        except Exception:
            staged_path.unlink(missing_ok=True)
            raise
        accepted = JobSubmittedResponse(job_id=job_id, job_type="upload", status="queued")
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=accepted.model_dump())

    try:
        result = await run_in_threadpool(ingest_csv_file, staged_path, filename, current_user.id)
    finally:
        staged_path.unlink(missing_ok=True)
    await invalidate_user_cache(current_user.id)
    return result


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> JobStatusResponse:
    """Return status and result for a background job owned by the authenticated user."""
    job = await get_job_state(job_id)
    if job is None:
        if not await check_redis_health():
            raise RuntimeError("Background job store is currently unavailable.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    if int(job["user_id"]) != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to access this job.",
        )
    return JobStatusResponse(
        job_id=job["job_id"],
        job_type=job["job_type"],
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )
