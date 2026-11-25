"""Phase 4 tests: golden eval normalization, offline eval runner, telemetry, and /metrics."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
from fastapi.testclient import TestClient
from google.genai import errors
import pytest

from backend.app.config import settings
from backend.app.core import cache, telemetry
from backend.app.core.database import check_and_seed_db, configure_database
from backend.app.core.llm_client import reset_genai_client
from backend.app.main import app
from backend.app.worker import run_async_query_job, set_arq_pool_for_tests
from evals import run_eval


def _deterministic_embedding(text: str) -> list[float]:
    lowered = text.lower()
    if "create table customers" in lowered:
        return [1.0, 0.0, 0.0, 0.0]
    if "create table products" in lowered:
        return [0.0, 1.0, 0.0, 0.0]
    if "create table orders" in lowered:
        return [0.0, 0.0, 1.0, 0.0]
    if "customer" in lowered or "region" in lowered:
        return [1.0, 0.0, 0.0, 0.0]
    if "product" in lowered:
        return [0.0, 1.0, 0.0, 0.0]
    if "order" in lowered or "revenue" in lowered:
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


def _install_mock_genai(monkeypatch, responses: list[tuple[str | Exception, int, int]]):
    mock_client = MagicMock()
    iterator = iter(responses)

    def fake_generate_content(*, model: str, contents: str):
        _ = (model, contents)
        item, in_tok, out_tok = next(iterator)
        if isinstance(item, Exception):
            raise item
        usage = SimpleNamespace(prompt_token_count=in_tok, candidates_token_count=out_tok)
        return SimpleNamespace(text=item, usage_metadata=usage)

    def fake_embed_content(*, model: str, contents: str, config=None):
        _ = (model, config)
        vec = _deterministic_embedding(str(contents))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vec)])

    mock_client.models.generate_content.side_effect = fake_generate_content
    mock_client.models.embed_content.side_effect = fake_embed_content
    monkeypatch.setattr("backend.app.core.llm_client.get_genai_client", lambda: mock_client)


@pytest.fixture
def telemetry_env(tmp_path, monkeypatch):
    prev_url = settings.database_url
    db_file = tmp_path / "phase4_test.sqlite3"
    settings.database_url = f"sqlite:///{db_file.as_posix()}"
    reset_genai_client()
    telemetry.metrics_registry.reset()

    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cache.set_redis_client_for_tests(fake_redis)

    stub_arq = MagicMock()
    stub_arq.enqueue_job = AsyncMock(return_value=SimpleNamespace(job_id="stub"))
    set_arq_pool_for_tests(stub_arq)

    _install_mock_genai(monkeypatch, [])
    try:
        with TestClient(app) as test_client:
            yield test_client, monkeypatch
    finally:
        cache.set_redis_client_for_tests(None)
        set_arq_pool_for_tests(None)
        reset_genai_client()
        telemetry.metrics_registry.reset()
        settings.database_url = prev_url
        configure_database(prev_url)


def test_golden_dataset_has_expected_30_case_breakdown():
    cases = json.loads(Path("evals/golden_dataset.json").read_text(encoding="utf-8"))
    assert len(cases) == 30

    counts: dict[str, int] = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1

    assert counts == {
        "single_table": 8,
        "multi_table_join": 10,
        "cte_window_timeseries": 5,
        "insufficient_schema": 4,
        "adversarial": 3,
    }


def test_compare_result_sets_normalizes_order_aliases_and_floats():
    expected = [
        {"region": "Europe", "total_revenue": 1234.5678},
        {"region": "Asia", "total_revenue": 500.0},
    ]
    # Different column alias + swapped row order + float precision differences
    actual_unordered = [
        {"customer_region": "Asia", "sum_sales": 500},
        {"customer_region": "Europe", "sum_sales": 1234.57},
    ]

    ok, detail = run_eval.compare_result_sets(actual_unordered, expected, order_matters=False)
    assert ok is True, detail

    # When order_matters=True, reversed order must fail
    ok_ordered, _ = run_eval.compare_result_sets(actual_unordered, expected, order_matters=True)
    assert ok_ordered is False


def test_offline_eval_passes_all_30_cases_and_writes_reports(tmp_path):
    prev_url = settings.database_url
    db_file = tmp_path / "offline_eval.sqlite3"
    settings.database_url = f"sqlite:///{db_file.as_posix()}"
    configure_database(settings.database_url)
    check_and_seed_db()

    try:
        cases = json.loads(Path("evals/golden_dataset.json").read_text(encoding="utf-8"))
        report = run_eval.run_offline_eval(cases)
        assert report["total_cases"] == 30
        assert report["passed"] == 30
        assert report["failed"] == 0
        assert report["execution_accuracy"] == 1.0

        json_path, md_path = run_eval.write_reports(report, output_dir=tmp_path)
        assert json_path.exists()
        assert md_path.exists()
        assert "30 / 30" in md_path.read_text(encoding="utf-8")
    finally:
        settings.database_url = prev_url
        configure_database(prev_url)


def test_telemetry_json_logs_and_metrics_endpoint(telemetry_env, monkeypatch, caplog):
    client, _ = telemetry_env

    # /metrics must enforce authentication
    assert client.get("/api/v1/metrics").status_code == 401

    reg = client.post(
        "/api/v1/auth/register",
        json={"email": "metrics@example.com", "password": "StrongPassword1!"},
    )
    assert reg.status_code == 201
    user_id = reg.json()["user"]["id"]
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    monkeypatch.setattr("backend.app.core.llm_client.time.sleep", lambda _s: None)
    err_503 = errors.ServerError(503, {"error": {"message": "503 UNAVAILABLE"}})

    _install_mock_genai(
        monkeypatch,
        [
            # Query 1: self-correction (bad SQL -> good SQL)
            ("SELECT missing_col FROM orders;", 50, 10),
            ("SELECT COUNT(*) AS cnt FROM orders;", 60, 12),
            # Query 2: repeated Query 1 -> Redis cache hit (no LLM call)
            # Query 3: 503 x3 -> fallback model
            (err_503, 0, 0),
            (err_503, 0, 0),
            (err_503, 0, 0),
            ("SELECT COUNT(*) AS cnt FROM customers;", 40, 8),
            # Query 4: guard rejection
            ("DROP TABLE orders;", 30, 5),
            ("DELETE FROM orders;", 30, 5),
            # Worker async query job
            ("SELECT COUNT(*) AS cnt FROM products;", 25, 6),
        ],
    )

    caplog.set_level(logging.INFO, logger="querypilot.telemetry")

    r1 = client.post("/api/v1/query", headers=headers, json={"question": "Count orders"})
    assert r1.status_code == 200 and r1.json()["corrected"] is True

    r2 = client.post("/api/v1/query", headers=headers, json={"question": "Count orders"})
    assert r2.status_code == 200 and r2.json()["cache_hit"] is True

    r3 = client.post("/api/v1/query", headers=headers, json={"question": "Count customers"})
    assert r3.status_code == 200 and r3.json()["fallback_triggered"] is True

    r4 = client.post("/api/v1/query", headers=headers, json={"question": "Drop orders table"})
    assert r4.status_code == 200 and r4.json()["sql_guard_status"] == "blocked"

    # Worker execution emits structured telemetry too
    job_submit = client.post(
        "/api/v1/query/async",
        headers=headers,
        json={"question": "Count products"},
    )
    assert job_submit.status_code == 202
    job_id = job_submit.json()["job_id"]
    client.portal.call(run_async_query_job, {}, job_id, user_id, "Count products")

    # Verify structured JSON log records contain all required fields
    required_keys = {
        "request_id",
        "user_id",
        "model_used",
        "fallback_triggered",
        "self_correction_attempts",
        "cache_hit",
        "retrieved_tables",
        "sql_guard_status",
        "latency_ms",
        "input_tokens",
        "output_tokens",
    }
    telemetry_events = [
        json.loads(rec.message)
        for rec in caplog.records
        if rec.name == "querypilot.telemetry"
    ]
    assert len(telemetry_events) == 5
    for evt in telemetry_events:
        assert required_keys.issubset(evt.keys())

    metrics_resp = client.get("/api/v1/metrics", headers=headers)
    assert metrics_resp.status_code == 200
    m = metrics_resp.json()
    assert m["total_queries"] == 5
    assert m["cache_hits"] == 1
    assert m["cache_hit_rate"] == pytest.approx(0.2)
    assert m["self_corrections"] == 2
    assert m["fallback_count"] == 1
    assert m["guard_rejections"] == 1
    assert m["error_count"] == 1
    assert m["total_input_tokens"] == 235
    assert m["total_output_tokens"] == 46


def test_tracing_disabled_by_default():
    assert settings.enable_tracing is False
    with telemetry.trace_query_span("test.span", {"request_id": "abc"}) as span:
        assert span is None
