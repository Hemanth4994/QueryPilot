"""Phase 3 tests: JWT auth, cross-tenant isolation, Redis cache/rate limit, and ARQ jobs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
from fastapi.testclient import TestClient
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from backend.app.config import Settings, settings
from backend.app.core import auth, cache
from backend.app.core.database import configure_database
from backend.app.core.llm_client import reset_genai_client
from backend.app.main import app
from backend.app.worker import (
    run_async_query_job,
    run_async_upload_job,
    set_arq_pool_for_tests,
)


def _deterministic_embedding(text: str) -> list[float]:
    lowered = text.lower()
    if "create table customers" in lowered:
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    if "create table products" in lowered:
        return [0.0, 1.0, 0.0, 0.0, 0.0]
    if "create table orders" in lowered:
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    if "create table u_" in lowered and ("secrets" in lowered or "targets" in lowered):
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    if "customer" in lowered or "region" in lowered:
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    if "order" in lowered or "revenue" in lowered:
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    if "secret" in lowered or "target" in lowered:
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 0.0, 1.0]


def _install_mock_genai(monkeypatch, responses: list[str] | None = None):
    mock_client = MagicMock()
    llm_calls: list[tuple[str, str]] = []
    embed_calls: list[tuple[str, str]] = []
    iterator = iter(responses or [])

    def fake_generate_content(*, model: str, contents: str):
        llm_calls.append((model, contents))
        return SimpleNamespace(text=next(iterator))

    def fake_embed_content(*, model: str, contents: str, config=None):
        embed_calls.append((model, str(contents)))
        vec = _deterministic_embedding(str(contents))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vec)])

    mock_client.models.generate_content.side_effect = fake_generate_content
    mock_client.models.embed_content.side_effect = fake_embed_content
    monkeypatch.setattr("backend.app.core.llm_client.get_genai_client", lambda: mock_client)
    return llm_calls, embed_calls


@pytest.fixture
def phase3_env(tmp_path, monkeypatch):
    """Provide isolated DB, FakeRedis, stub ARQ pool, and mocked GenAI."""
    prev_url = settings.database_url
    prev_rate = settings.rate_limit_per_minute
    prev_threshold = settings.async_upload_threshold_bytes
    db_file = tmp_path / "phase3_test.sqlite3"
    settings.database_url = f"sqlite:///{db_file.as_posix()}"
    reset_genai_client()

    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cache.set_redis_client_for_tests(fake_redis)

    enqueued_jobs: list[tuple] = []
    stub_arq = MagicMock()

    async def fake_enqueue(func_name: str, *args):
        enqueued_jobs.append((func_name, args))
        return SimpleNamespace(job_id=args[0] if args else "job")

    stub_arq.enqueue_job = AsyncMock(side_effect=fake_enqueue)
    set_arq_pool_for_tests(stub_arq)
    llm_calls, embed_calls = _install_mock_genai(monkeypatch)

    try:
        with TestClient(app) as test_client:
            yield SimpleNamespace(
                client=test_client,
                redis=fake_redis,
                enqueued=enqueued_jobs,
                llm_calls=llm_calls,
                embed_calls=embed_calls,
                monkeypatch=monkeypatch,
            )
    finally:
        cache.set_redis_client_for_tests(None)
        set_arq_pool_for_tests(None)
        reset_genai_client()
        settings.rate_limit_per_minute = prev_rate
        settings.async_upload_threshold_bytes = prev_threshold
        settings.database_url = prev_url
        configure_database(prev_url)


def _register_user(client: TestClient, email: str, password: str = "SecurePass123!") -> tuple[int, dict[str, str]]:
    resp = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201
    data = resp.json()
    return data["user"]["id"], {"Authorization": f"Bearer {data['access_token']}"}


def test_jwt_secret_key_has_no_hardcoded_default(monkeypatch):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("jwt_secret_key", raising=False)
    clean_settings = Settings(_env_file=None)
    assert clean_settings.jwt_secret_key == ""

    prev_secret = settings.jwt_secret_key
    settings.jwt_secret_key = ""
    try:
        with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
            auth.create_access_token(1, "user@example.com")
    finally:
        settings.jwt_secret_key = prev_secret


def test_auth_register_login_me_and_401_enforcement(phase3_env):
    client = phase3_env.client

    # Unauthenticated calls must fail with 401
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/schema").status_code == 401
    assert client.post("/api/v1/query", json={"question": "Count customers"}).status_code == 401
    assert client.post("/api/v1/query/async", json={"question": "Count customers"}).status_code == 401
    assert (
        client.post("/api/v1/upload", files={"file": ("a.csv", b"a\n1\n", "text/csv")}).status_code
        == 401
    )
    assert client.get("/api/v1/jobs/nonexistent").status_code == 401

    user_id, headers = _register_user(client, "Alice@Example.com", "StrongPass99!")

    # Duplicate email returns 409
    dup = client.post(
        "/api/v1/auth/register",
        json={"email": "alice@example.com", "password": "StrongPass99!"},
    )
    assert dup.status_code == 409

    # Wrong password returns 401
    bad_login = client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": "WrongPassword!"},
    )
    assert bad_login.status_code == 401

    # Valid login returns Bearer token and /auth/me resolves profile without password
    good_login = client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": "StrongPass99!"},
    )
    assert good_login.status_code == 200
    token = good_login.json()["access_token"]
    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    me_payload = me_resp.json()
    assert me_payload["id"] == user_id
    assert me_payload["email"] == "alice@example.com"
    assert "hashed_password" not in me_payload


def test_cross_tenant_isolation_for_schema_rag_sql_and_jobs(phase3_env):
    client = phase3_env.client
    alice_id, alice_headers = _register_user(client, "alice_iso@example.com")
    _, bob_headers = _register_user(client, "bob_iso@example.com")

    # Alice uploads a private table
    upload_resp = client.post(
        "/api/v1/upload",
        headers=alice_headers,
        files={"file": ("secrets.csv", b"secret_id,amount\n1,9999\n", "text/csv")},
    )
    assert upload_resp.status_code == 200
    alice_table = upload_resp.json()["table_name"]
    assert alice_table == f"u_{alice_id}_secrets"

    # Bob's schema must not reveal Alice's table
    bob_schema = client.get("/api/v1/schema", headers=bob_headers)
    assert bob_schema.status_code == 200
    bob_tables = {t["name"] for t in bob_schema.json()["tables"]}
    assert alice_table not in bob_tables
    assert {"customers", "products", "orders"}.issubset(bob_tables)

    # Bob's Schema RAG must not retrieve Alice's table -> short-circuits INSUFFICIENT_SCHEMA
    _install_mock_genai(phase3_env.monkeypatch, ["SELECT * FROM u_1_secrets;"])
    bob_rag_resp = client.post(
        "/api/v1/query",
        headers=bob_headers,
        json={"question": "Show all secrets"},
    )
    assert bob_rag_resp.status_code == 200
    assert bob_rag_resp.json()["status"] == "INSUFFICIENT_SCHEMA"
    assert bob_rag_resp.json()["retrieved_tables"] == []

    # Even if LLM emits SQL explicitly targeting Alice's table, SQL guard blocks Bob
    _install_mock_genai(
        phase3_env.monkeypatch,
        [
            f"SELECT * FROM {alice_table};",
            f"SELECT * FROM {alice_table};",
        ],
    )
    bob_attack = client.post(
        "/api/v1/query",
        headers=bob_headers,
        json={"question": "Count orders"},
    )
    assert bob_attack.status_code == 200
    assert bob_attack.json()["status"] == "ERROR"
    assert "Security Error" in bob_attack.json()["error"]

    # Bob cannot read Alice's async job
    alice_job_resp = client.post(
        "/api/v1/query/async",
        headers=alice_headers,
        json={"question": "Count customers"},
    )
    assert alice_job_resp.status_code == 202
    job_id = alice_job_resp.json()["job_id"]

    bob_poll = client.get(f"/api/v1/jobs/{job_id}", headers=bob_headers)
    assert bob_poll.status_code == 403


def test_redis_cache_hit_and_invalidation_after_upload_and_reindex(phase3_env):
    client = phase3_env.client
    redis_client = phase3_env.redis
    user_id, headers = _register_user(client, "cache@example.com")

    llm_calls, embed_calls = _install_mock_genai(
        phase3_env.monkeypatch,
        [
            "SELECT COUNT(*) AS cnt FROM customers;",
            "SELECT COUNT(*) AS cnt FROM customers;",
            "SELECT COUNT(*) AS cnt FROM customers;",
        ],
    )

    # Schema endpoint populates schema:{user_id} cache
    s1 = client.get("/api/v1/schema", headers=headers)
    assert s1.status_code == 200
    assert client.portal.call(redis_client.get, cache.schema_cache_key(user_id)) is not None

    # First query misses cache; second identical query hits cache without calling GenAI
    q1 = client.post("/api/v1/query", headers=headers, json={"question": "How many customers?"})
    assert q1.status_code == 200
    assert q1.json()["cache_hit"] is False
    calls_after_q1 = (len(llm_calls), len(embed_calls))

    q2 = client.post("/api/v1/query", headers=headers, json={"question": "  how many   customers? "})
    assert q2.status_code == 200
    assert q2.json()["cache_hit"] is True
    assert q2.json()["rows"] == [{"cnt": 100}]
    assert (len(llm_calls), len(embed_calls)) == calls_after_q1

    # Uploading a CSV invalidates user's schema and query caches
    up = client.post(
        "/api/v1/upload",
        headers=headers,
        files={"file": ("targets.csv", b"month,val\nJan,10\n", "text/csv")},
    )
    assert up.status_code == 200
    assert client.portal.call(redis_client.get, cache.schema_cache_key(user_id)) is None

    q3 = client.post("/api/v1/query", headers=headers, json={"question": "How many customers?"})
    assert q3.status_code == 200
    assert q3.json()["cache_hit"] is False
    assert len(llm_calls) == 2

    # Reindex also invalidates caches
    reindex = client.post("/api/v1/schema/reindex", headers=headers)
    assert reindex.status_code == 200
    q4 = client.post("/api/v1/query", headers=headers, json={"question": "How many customers?"})
    assert q4.status_code == 200
    assert q4.json()["cache_hit"] is False
    assert len(llm_calls) == 3


def test_graceful_degradation_when_redis_unavailable(phase3_env):
    client = phase3_env.client
    _, headers = _register_user(client, "degraded@example.com")
    _install_mock_genai(phase3_env.monkeypatch, ["SELECT COUNT(*) AS cnt FROM customers;"])

    broken_redis = MagicMock()
    broken_redis.ping = AsyncMock(side_effect=RedisConnectionError("Redis offline"))
    broken_redis.get = AsyncMock(side_effect=RedisConnectionError("Redis offline"))
    broken_redis.set = AsyncMock(side_effect=RedisConnectionError("Redis offline"))
    broken_redis.pipeline.side_effect = RedisConnectionError("Redis offline")
    cache.set_redis_client_for_tests(broken_redis)

    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["redis_ready"] is False

    schema_resp = client.get("/api/v1/schema", headers=headers)
    assert schema_resp.status_code == 200

    query_resp = client.post(
        "/api/v1/query",
        headers=headers,
        json={"question": "How many customers?"},
    )
    assert query_resp.status_code == 200
    assert query_resp.json()["status"] == "SUCCESS"
    assert query_resp.json()["cache_hit"] is False


def test_rate_limiter_returns_429_when_limit_exceeded(phase3_env):
    client = phase3_env.client
    _, headers = _register_user(client, "ratelimit@example.com")
    settings.rate_limit_per_minute = 3
    _install_mock_genai(
        phase3_env.monkeypatch,
        [
            "SELECT COUNT(*) AS cnt FROM customers;",
            "SELECT COUNT(*) AS cnt FROM orders;",
            "SELECT COUNT(*) AS cnt FROM products;",
        ],
    )

    for q in ("Count customers", "Count orders", "Count products"):
        ok = client.post("/api/v1/query", headers=headers, json={"question": q})
        assert ok.status_code == 200

    limited_query = client.post(
        "/api/v1/query",
        headers=headers,
        json={"question": "Another customer query"},
    )
    assert limited_query.status_code == 429
    assert "Retry-After" in limited_query.headers

    limited_upload = client.post(
        "/api/v1/upload",
        headers=headers,
        files={"file": ("extra.csv", b"a\n1\n", "text/csv")},
    )
    assert limited_upload.status_code == 429


def test_async_query_and_upload_worker_lifecycle(phase3_env):
    client = phase3_env.client
    user_id, headers = _register_user(client, "worker@example.com")

    # 1. Enqueue async query
    _install_mock_genai(
        phase3_env.monkeypatch,
        ["SELECT COUNT(*) AS total_orders FROM orders;"],
    )
    q_submit = client.post(
        "/api/v1/query/async",
        headers=headers,
        json={"question": "How many orders?"},
    )
    assert q_submit.status_code == 202
    q_job_id = q_submit.json()["job_id"]
    assert q_submit.json()["status"] == "queued"

    queued_poll = client.get(f"/api/v1/jobs/{q_job_id}", headers=headers)
    assert queued_poll.status_code == 200
    assert queued_poll.json()["status"] == "queued"

    # Simulate ARQ worker executing the queued query job
    func_name, job_args = phase3_env.enqueued[-1]
    assert func_name == "run_async_query_job"
    client.portal.call(run_async_query_job, {}, *job_args)

    done_poll = client.get(f"/api/v1/jobs/{q_job_id}", headers=headers)
    assert done_poll.status_code == 200
    done_payload = done_poll.json()
    assert done_payload["status"] == "completed"
    assert done_payload["result"]["status"] == "SUCCESS"
    assert done_payload["result"]["rows"] == [{"total_orders": 500}]

    # 2. Enqueue large CSV upload automatically (above threshold)
    settings.async_upload_threshold_bytes = 10
    csv_bytes = b"region,quota\nEurope,5000\nAsia,7000\n"
    u_submit = client.post(
        "/api/v1/upload",
        headers=headers,
        files={"file": ("targets.csv", csv_bytes, "text/csv")},
    )
    assert u_submit.status_code == 202
    u_job_id = u_submit.json()["job_id"]

    up_func_name, up_args = phase3_env.enqueued[-1]
    assert up_func_name == "run_async_upload_job"
    client.portal.call(run_async_upload_job, {}, *up_args)

    u_done = client.get(f"/api/v1/jobs/{u_job_id}", headers=headers)
    assert u_done.status_code == 200
    assert u_done.json()["status"] == "completed"
    assert u_done.json()["result"]["table_name"] == f"u_{user_id}_targets"
    assert u_done.json()["result"]["embedding_indexed"] is True
