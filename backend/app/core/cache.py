"""Redis caching, sliding-window rate limiting, and tenant-scoped job metadata store."""

import asyncio
from datetime import UTC, datetime
import hashlib
import json
import logging
import time
from typing import Any
import uuid

from fastapi import HTTPException, status
import redis.asyncio as aioredis
from redis.exceptions import RedisError

from ..config import settings

logger = logging.getLogger(__name__)

_redis_client: aioredis.Redis | None = None
_redis_loop_id: int | None = None
_redis_override: aioredis.Redis | None = None


def set_redis_client_for_tests(client: aioredis.Redis | None) -> None:
    """Inject a FakeRedis or mock client during offline tests."""
    global _redis_override
    _redis_override = client


async def close_redis_client() -> None:
    """Close any active Redis connection pool cleanly."""
    global _redis_client, _redis_loop_id
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
    _redis_client = None
    _redis_loop_id = None


def get_redis_client() -> aioredis.Redis | None:
    """Return an async Redis client bound to the running event loop, or None."""
    global _redis_client, _redis_loop_id
    if _redis_override is not None:
        return _redis_override

    url = settings.redis_url.strip()
    if not url:
        return None

    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = None

    if _redis_client is None or _redis_loop_id != loop_id:
        _redis_client = aioredis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=1.5,
            socket_timeout=2.0,
        )
        _redis_loop_id = loop_id
    return _redis_client


async def check_redis_health() -> bool:
    """Return True when Redis responds to PING, False otherwise."""
    client = get_redis_client()
    if client is None:
        return False
    try:
        return bool(await client.ping())
    except (RedisError, OSError) as exc:
        logger.warning("Redis health check failed (graceful bypass): %s", exc)
        return False


def schema_cache_key(user_id: int) -> str:
    """Build Redis cache key for a user's schema response."""
    return f"schema:{user_id}"


def compute_schema_hash(schema_text: str) -> str:
    """Compute a deterministic hash for a user's active schema DDL/text."""
    return hashlib.sha256(schema_text.encode("utf-8")).hexdigest()[:16]


def query_cache_key(user_id: int, schema_hash: str, question: str) -> str:
    """Build Redis cache key scoped by user, schema version hash, and normalized question."""
    normalized_q = " ".join(question.strip().lower().split())
    q_hash = hashlib.sha256(normalized_q.encode("utf-8")).hexdigest()[:24]
    return f"query:{user_id}:{schema_hash}:{q_hash}"


async def get_cached_schema(user_id: int) -> dict[str, Any] | None:
    """Fetch cached schema JSON for user_id, returning None on miss or Redis outage."""
    client = get_redis_client()
    if client is None:
        return None
    try:
        raw = await client.get(schema_cache_key(user_id))
        return json.loads(raw) if raw else None
    except (RedisError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Redis schema cache read failed (bypassed): %s", exc)
        return None


async def set_cached_schema(user_id: int, payload: dict[str, Any]) -> None:
    """Store user schema JSON in Redis with CACHE_TTL_SECONDS expiration."""
    client = get_redis_client()
    if client is None:
        return
    try:
        await client.set(
            schema_cache_key(user_id),
            json.dumps(payload),
            ex=settings.cache_ttl_seconds,
        )
    except (RedisError, OSError) as exc:
        logger.warning("Redis schema cache write failed (bypassed): %s", exc)


async def get_cached_query(user_id: int, schema_hash: str, question: str) -> dict[str, Any] | None:
    """Fetch cached query response for (user_id, schema_hash, question)."""
    client = get_redis_client()
    if client is None:
        return None
    try:
        raw = await client.get(query_cache_key(user_id, schema_hash, question))
        if not raw:
            return None
        data = json.loads(raw)
        data["cache_hit"] = True
        return data
    except (RedisError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Redis query cache read failed (bypassed): %s", exc)
        return None


async def set_cached_query(
    user_id: int,
    schema_hash: str,
    question: str,
    payload: dict[str, Any],
) -> None:
    """Cache non-error query responses in Redis with CACHE_TTL_SECONDS expiration."""
    if payload.get("status") == "ERROR":
        return
    client = get_redis_client()
    if client is None:
        return
    try:
        to_store = dict(payload)
        to_store["cache_hit"] = False
        await client.set(
            query_cache_key(user_id, schema_hash, question),
            json.dumps(to_store),
            ex=settings.cache_ttl_seconds,
        )
    except (RedisError, OSError) as exc:
        logger.warning("Redis query cache write failed (bypassed): %s", exc)


async def invalidate_user_cache(user_id: int) -> None:
    """Invalidate schema and query caches for user_id after CSV upload or reindex."""
    client = get_redis_client()
    if client is None:
        return
    try:
        keys_to_delete = [schema_cache_key(user_id)]
        async for key in client.scan_iter(match=f"query:{user_id}:*"):
            keys_to_delete.append(key)
        if keys_to_delete:
            await client.delete(*keys_to_delete)
    except (RedisError, OSError) as exc:
        logger.warning("Redis cache invalidation failed (bypassed): %s", exc)


async def enforce_rate_limit(user_id: int, scope: str = "api") -> None:
    """Apply a 60-second sliding-window rate limit per user; bypass gracefully if Redis is down."""
    client = get_redis_client()
    if client is None:
        return

    limit = int(settings.rate_limit_per_minute)
    now = time.time()
    window_start = now - 60.0
    key = f"ratelimit:{user_id}:{scope}"
    member = f"{now}:{uuid.uuid4().hex[:8]}"

    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            results = await pipe.execute()

        current_count = int(results[1])
        if current_count >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({limit} requests per minute). Please retry shortly.",
                headers={"Retry-After": "60"},
            )

        async with client.pipeline(transaction=True) as pipe:
            pipe.zadd(key, {member: now})
            pipe.expire(key, 65)
            await pipe.execute()
    except HTTPException:
        raise
    except (RedisError, OSError) as exc:
        logger.warning("Redis rate limiter unavailable (graceful bypass): %s", exc)


def job_metadata_key(job_id: str) -> str:
    """Build Redis key storing tenant-owned async job state."""
    return f"job:{job_id}"


async def save_job_state(
    job_id: str,
    user_id: int,
    job_type: str,
    job_status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Persist async job state in Redis scoped by user_id."""
    client = get_redis_client()
    if client is None:
        raise RuntimeError("Redis is required to track async jobs.")

    now_iso = datetime.now(UTC).isoformat()
    payload = {
        "job_id": job_id,
        "user_id": user_id,
        "job_type": job_type,
        "status": job_status,
        "result": result,
        "error": error,
        "created_at": created_at or now_iso,
        "updated_at": now_iso,
    }
    try:
        await client.set(job_metadata_key(job_id), json.dumps(payload), ex=86400)
    except (RedisError, OSError) as exc:
        logger.warning("Failed to persist job state %s: %s", job_id, exc)
        raise RuntimeError(f"Background job store is unavailable: {exc}") from exc
    return payload


async def get_job_state(job_id: str) -> dict[str, Any] | None:
    """Retrieve async job state from Redis."""
    client = get_redis_client()
    if client is None:
        raise RuntimeError("Redis is unavailable.")
    try:
        raw = await client.get(job_metadata_key(job_id))
    except (RedisError, OSError) as exc:
        logger.warning("Failed to fetch job state %s: %s", job_id, exc)
        raise RuntimeError(f"Background job store is unavailable: {exc}") from exc
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Corrupt job state JSON for %s: %s", job_id, exc)
        return None
