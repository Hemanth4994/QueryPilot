"""ARQ background worker tasks and pool management for async queries and CSV uploads."""

import asyncio
import base64
import logging
from pathlib import Path
import time
from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from redis.exceptions import RedisError
from starlette.concurrency import run_in_threadpool

from .config import settings
from .core.agent import run_query_pipeline
from .core.auth import ensure_users_table
from .core.cache import (
    close_redis_client,
    compute_schema_hash,
    get_cached_query,
    get_job_state,
    invalidate_user_cache,
    save_job_state,
    set_cached_query,
)
from .core.database import check_and_seed_db, configure_database, get_schema_for_llm
from .core.ingestion import ingest_csv_bytes, ingest_csv_file
from .core.schema_rag import ensure_schema_embeddings_table
from .core.telemetry import QueryTelemetryEvent, emit_query_telemetry, trace_query_span

logger = logging.getLogger(__name__)

_arq_pool: ArqRedis | None = None
_arq_loop_id: int | None = None
_arq_override: Any = None


def set_arq_pool_for_tests(pool: Any) -> None:
    """Inject a mock/stub ARQ pool during offline tests."""
    global _arq_override
    _arq_override = pool


async def close_arq_pool() -> None:
    """Close the active ARQ Redis pool cleanly."""
    global _arq_pool, _arq_loop_id
    if _arq_pool is not None:
        try:
            await _arq_pool.aclose()
        except Exception:
            pass
    _arq_pool = None
    _arq_loop_id = None


async def get_arq_pool() -> ArqRedis:
    """Return an active ARQ Redis pool bound to the current event loop."""
    global _arq_pool, _arq_loop_id
    if _arq_override is not None:
        return _arq_override

    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = None

    if _arq_pool is None or _arq_loop_id != loop_id:
        try:
            _arq_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
            _arq_loop_id = loop_id
        except (RedisError, OSError) as exc:
            raise RuntimeError(f"Background job queue unavailable: {exc}") from exc
    return _arq_pool


async def run_async_query_job(
    _ctx: dict[str, Any],
    job_id: str,
    user_id: int,
    question: str,
) -> dict[str, Any]:
    """Execute a natural-language Text-to-SQL query in the ARQ worker with telemetry."""
    start = time.perf_counter()
    existing = await get_job_state(job_id)
    created_at = existing["created_at"] if existing else None
    await save_job_state(
        job_id=job_id,
        user_id=user_id,
        job_type="query",
        job_status="running",
        created_at=created_at,
    )

    try:
        with trace_query_span("worker.run_async_query_job", {"job_id": job_id, "user_id": user_id}):
            schema_text = await run_in_threadpool(get_schema_for_llm, None, user_id)
            schema_hash = compute_schema_hash(schema_text)
            cached = await get_cached_query(user_id, schema_hash, question)
            if cached is not None:
                cached["request_id"] = job_id
                cached["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
                result_dict = cached
                emit_query_telemetry(
                    QueryTelemetryEvent(
                        request_id=job_id,
                        user_id=user_id,
                        model_used=cached.get("model"),
                        fallback_triggered=bool(cached.get("fallback_triggered", False)),
                        self_correction_attempts=int(cached.get("self_correction_attempts", 0)),
                        cache_hit=True,
                        retrieved_tables=list(cached.get("retrieved_tables", [])),
                        sql_guard_status=str(cached.get("sql_guard_status", "passed")),
                        latency_ms=float(cached["latency_ms"]),
                        input_tokens=0,
                        output_tokens=0,
                    )
                )
            else:
                response = await run_in_threadpool(
                    run_query_pipeline,
                    question,
                    user_id,
                    job_id,
                )
                result_dict = response.model_dump(mode="json")
                await set_cached_query(user_id, schema_hash, question, result_dict)

            return await save_job_state(
                job_id=job_id,
                user_id=user_id,
                job_type="query",
                job_status="completed",
                result=result_dict,
                created_at=created_at,
            )
    except Exception as exc:
        logger.exception("Async query job %s failed", job_id)
        emit_query_telemetry(
            QueryTelemetryEvent(
                request_id=job_id,
                user_id=user_id,
                model_used=None,
                fallback_triggered=False,
                self_correction_attempts=0,
                cache_hit=False,
                retrieved_tables=[],
                sql_guard_status="error",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                input_tokens=0,
                output_tokens=0,
            )
        )
        return await save_job_state(
            job_id=job_id,
            user_id=user_id,
            job_type="query",
            job_status="failed",
            error=str(exc),
            created_at=created_at,
        )


async def run_async_upload_job(
    _ctx: dict[str, Any],
    job_id: str,
    user_id: int,
    filename: str,
    payload_or_path: str,
    is_file_path: bool = False,
) -> dict[str, Any]:
    """Ingest a large CSV upload from staged disk path (or legacy base64) in ARQ."""
    existing = await get_job_state(job_id)
    created_at = existing["created_at"] if existing else None
    await save_job_state(
        job_id=job_id,
        user_id=user_id,
        job_type="upload",
        job_status="running",
        created_at=created_at,
    )

    staged_path = Path(payload_or_path) if is_file_path else None
    try:
        if staged_path is not None:
            upload_resp = await run_in_threadpool(ingest_csv_file, staged_path, filename, user_id)
        else:
            raw_bytes = base64.b64decode(payload_or_path.encode("ascii"))
            upload_resp = await run_in_threadpool(ingest_csv_bytes, raw_bytes, filename, user_id)
        await invalidate_user_cache(user_id)
        return await save_job_state(
            job_id=job_id,
            user_id=user_id,
            job_type="upload",
            job_status="completed",
            result=upload_resp.model_dump(mode="json"),
            created_at=created_at,
        )
    except Exception as exc:
        logger.exception("Async upload job %s failed", job_id)
        return await save_job_state(
            job_id=job_id,
            user_id=user_id,
            job_type="upload",
            job_status="failed",
            error=str(exc),
            created_at=created_at,
        )
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)


async def startup(_ctx: dict[str, Any]) -> None:
    """Initialize SQLAlchemy connection pool and metadata tables inside ARQ worker."""
    configure_database(settings.database_url)
    check_and_seed_db()
    ensure_users_table()
    ensure_schema_embeddings_table()


async def shutdown(_ctx: dict[str, Any]) -> None:
    """Close Redis connections on ARQ worker shutdown."""
    await close_redis_client()


class WorkerSettings:
    """ARQ worker configuration (`arq app.worker.WorkerSettings`)."""

    functions = [run_async_query_job, run_async_upload_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 10
    job_timeout = settings.worker_job_timeout_seconds
