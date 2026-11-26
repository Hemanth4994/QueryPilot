"""Fault-tolerance and negative-path tests for degraded dependencies and malformed inputs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
from fastapi.testclient import TestClient
from google.genai import errors
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from backend.app.config import settings
from backend.app.core import cache
from backend.app.core.database import configure_database
from backend.app.core.llm_client import reset_genai_client
from backend.app.core.telemetry import metrics_registry
from backend.app.main import app
from backend.app.worker import set_arq_pool_for_tests


def _embed_vector(text_val: str) -> list[float]:
    lowered = text_val.lower()
    if "customers" in lowered or "customer" in lowered:
        return [1.0, 0.0, 0.0]
    if "orders" in lowered or "order" in lowered:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


@pytest.fixture
def fault_env(tmp_path, monkeypatch):
    """Set up an isolated SQLite DB, FakeRedis, stub ARQ pool, and clean metrics."""
    prev_url = settings.database_url
    db_file = tmp_path / "fault_test.sqlite3"
    settings.database_url = f"sqlite:///{db_file.as_posix()}"
    reset_genai_client()
    metrics_registry.reset()

    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cache.set_redis_client_for_tests(fake_redis)

    stub_arq = MagicMock()
    stub_arq.enqueue_job = AsyncMock(return_value=SimpleNamespace(job_id="stub"))
    set_arq_pool_for_tests(stub_arq)

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = lambda *, model, contents, config=None: (
        SimpleNamespace(embeddings=[SimpleNamespace(values=_embed_vector(str(contents)))])
    )
    monkeypatch.setattr("backend.app.core.llm_client.get_genai_client", lambda: mock_client)

    try:
        with TestClient(app) as client:
            reg = client.post(
                "/api/v1/auth/register",
                json={"email": "resilience@example.com", "password": "StrongPassword1!"},
            )
            assert reg.status_code == 201
            headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
            yield SimpleNamespace(
                client=client,
                headers=headers,
                mock_genai=mock_client,
                stub_arq=stub_arq,
                monkeypatch=monkeypatch,
            )
    finally:
        cache.set_redis_client_for_tests(None)
        set_arq_pool_for_tests(None)
        reset_genai_client()
        metrics_registry.reset()
        settings.database_url = prev_url
        configure_database(prev_url)


def test_health_reports_degraded_instead_of_500_when_db_and_redis_fail(fault_env):
    fault_env.monkeypatch.setattr(
        "backend.app.api.routes.get_engine",
        MagicMock(side_effect=RuntimeError("Postgres unreachable")),
    )
    broken_redis = MagicMock()
    broken_redis.ping = AsyncMock(side_effect=RedisConnectionError("Redis offline"))
    cache.set_redis_client_for_tests(broken_redis)

    resp = fault_env.client.get("/api/v1/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "degraded"
    assert payload["database_ready"] is False
    assert payload["redis_ready"] is False


def test_empty_llm_response_falls_back_then_returns_503_and_tracks_error(fault_env):
    # Empty response from primary falls back cleanly when fallback succeeds
    calls: list[str] = []

    def empty_then_valid(*, model: str, contents: str):
        _ = contents
        calls.append(model)
        if len(calls) == 1:
            return SimpleNamespace(text=None, usage_metadata=None)
        return SimpleNamespace(
            text="SELECT COUNT(*) AS cnt FROM customers;",
            usage_metadata=SimpleNamespace(prompt_token_count=20, candidates_token_count=5),
        )

    fault_env.mock_genai.models.generate_content.side_effect = empty_then_valid
    ok = fault_env.client.post(
        "/api/v1/query",
        headers=fault_env.headers,
        json={"question": "How many customers are there?"},
    )
    assert ok.status_code == 200
    assert ok.json()["status"] == "SUCCESS"
    assert ok.json()["fallback_triggered"] is True

    # Both primary and fallback returning None yields structured 503 and increments error_count
    fault_env.mock_genai.models.generate_content.side_effect = lambda *, model, contents: (
        SimpleNamespace(text="   ", usage_metadata=None)
    )
    fail = fault_env.client.post(
        "/api/v1/query",
        headers=fault_env.headers,
        json={"question": "Total customers count please"},
    )
    assert fail.status_code == 503
    assert fail.json()["error_code"] == "SERVICE_UNAVAILABLE"

    metrics = fault_env.client.get("/api/v1/metrics", headers=fault_env.headers).json()
    assert metrics["error_count"] == 1


def test_embedding_429_error_returns_structured_503(fault_env):
    err_429 = errors.ClientError(429, {"error": {"message": "429 RESOURCE_EXHAUSTED"}})
    fault_env.mock_genai.models.embed_content.side_effect = err_429

    resp = fault_env.client.post(
        "/api/v1/query",
        headers=fault_env.headers,
        json={"question": "Count orders"},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "SERVICE_UNAVAILABLE"
    assert "Embedding generation failed" in resp.json()["detail"]


def test_csv_upload_handles_cp1252_rejects_empty_rows_and_survives_embedding_outage(fault_env):
    # Windows-1252 encoding succeeds
    cp1252_csv = "city,note\nCafé,Top tier\n".encode("cp1252")
    ok_resp = fault_env.client.post(
        "/api/v1/upload",
        headers=fault_env.headers,
        files={"file": ("regions.csv", cp1252_csv, "text/csv")},
    )
    assert ok_resp.status_code == 200
    assert ok_resp.json()["row_count"] == 1
    assert ok_resp.json()["embedding_indexed"] is True

    # Header-only CSV rejected cleanly with 400 BAD_REQUEST
    empty_rows = fault_env.client.post(
        "/api/v1/upload",
        headers=fault_env.headers,
        files={"file": ("empty_rows.csv", b"col_a,col_b\n", "text/csv")},
    )
    assert empty_rows.status_code == 400
    assert empty_rows.json()["error_code"] == "BAD_REQUEST"

    # Embedding outage saves table cleanly with embedding_indexed=False
    err_429 = errors.ClientError(429, {"error": {"message": "429 RESOURCE_EXHAUSTED"}})
    fault_env.mock_genai.models.embed_content.side_effect = err_429
    degraded_upload = fault_env.client.post(
        "/api/v1/upload",
        headers=fault_env.headers,
        files={"file": ("backup.csv", b"item,qty\nWidget,5\n", "text/csv")},
    )
    assert degraded_upload.status_code == 200
    assert degraded_upload.json()["embedding_indexed"] is False
    assert degraded_upload.json()["row_count"] == 1


def test_async_queue_failure_and_job_status_outage_return_503(fault_env):
    fault_env.stub_arq.enqueue_job = AsyncMock(side_effect=RedisConnectionError("Redis dropped"))
    enqueue_resp = fault_env.client.post(
        "/api/v1/query/async",
        headers=fault_env.headers,
        json={"question": "Count orders"},
    )
    assert enqueue_resp.status_code == 503
    assert enqueue_resp.json()["error_code"] == "SERVICE_UNAVAILABLE"

    broken_redis = MagicMock()
    broken_redis.ping = AsyncMock(side_effect=RedisConnectionError("Redis offline"))
    broken_redis.get = AsyncMock(side_effect=RedisConnectionError("Redis offline"))
    cache.set_redis_client_for_tests(broken_redis)

    job_resp = fault_env.client.get("/api/v1/jobs/missing123", headers=fault_env.headers)
    assert job_resp.status_code == 503
    assert job_resp.json()["error_code"] == "SERVICE_UNAVAILABLE"
