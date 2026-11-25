"""FastAPI integration tests for QueryPilot endpoints (offline with mocked Gemini & FakeRedis)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
from fastapi.testclient import TestClient
from google.genai import errors
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.core.cache import set_redis_client_for_tests
from backend.app.core.database import configure_database, get_engine
from backend.app.core.llm_client import reset_genai_client
from backend.app.core.schema_rag import SchemaEmbedding
from backend.app.core.sql_guard import execute_sql
from backend.app.main import app
from backend.app.worker import set_arq_pool_for_tests


def _deterministic_embedding(text: str) -> list[float]:
    """Map DDL texts and questions to deterministic 5D basis vectors for offline RAG tests."""
    lowered = text.lower()
    if "create table customers" in lowered:
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    if "create table products" in lowered:
        return [0.0, 1.0, 0.0, 0.0, 0.0]
    if "create table orders" in lowered:
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    if "create table u_" in lowered and (
        "monthly_targets" in lowered or "customers" in lowered or "secrets" in lowered
    ):
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    if "customer" in lowered or "region" in lowered:
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    if "product" in lowered or "category" in lowered:
        return [0.0, 1.0, 0.0, 0.0, 0.0]
    if "order" in lowered or "revenue" in lowered:
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    if "target" in lowered or "month" in lowered or "secret" in lowered:
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 0.0, 1.0]


def _install_mock_genai(monkeypatch, responses: list[str | Exception] | None = None):
    """Patch genai client factory with scripted generate_content and deterministic embeddings."""
    mock_client = MagicMock()
    llm_calls: list[tuple[str, str]] = []
    embed_calls: list[tuple[str, str]] = []
    iterator = iter(responses or [])

    def fake_generate_content(*, model: str, contents: str):
        llm_calls.append((model, contents))
        next_item = next(iterator)
        if isinstance(next_item, Exception):
            raise next_item
        return SimpleNamespace(text=next_item)

    def fake_embed_content(*, model: str, contents: str, config=None):
        embed_calls.append((model, str(contents)))
        vec = _deterministic_embedding(str(contents))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vec)])

    mock_client.models.generate_content.side_effect = fake_generate_content
    mock_client.models.embed_content.side_effect = fake_embed_content
    monkeypatch.setattr("backend.app.core.llm_client.get_genai_client", lambda: mock_client)
    return llm_calls, embed_calls


def _register_and_headers(client: TestClient, email: str, password: str = "StrongPass123!") -> tuple[int, dict[str, str]]:
    """Register a user and return (user_id, Authorization header dict)."""
    resp = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201
    data = resp.json()
    return data["user"]["id"], {"Authorization": f"Bearer {data['access_token']}"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Provide a TestClient wired to an isolated DB, FakeRedis, stub ARQ, and mocked Gemini."""
    prev_url = settings.database_url
    db_file = tmp_path / "api_test.sqlite3"
    settings.database_url = f"sqlite:///{db_file.as_posix()}"
    reset_genai_client()

    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    set_redis_client_for_tests(fake_redis)

    stub_arq = MagicMock()
    stub_arq.enqueue_job = AsyncMock(return_value=SimpleNamespace(job_id="stub"))
    set_arq_pool_for_tests(stub_arq)

    _install_mock_genai(monkeypatch)
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        set_redis_client_for_tests(None)
        set_arq_pool_for_tests(None)
        reset_genai_client()
        settings.database_url = prev_url
        configure_database(prev_url)


def test_health_endpoint_returns_ready(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database_ready"] is True
    assert payload["redis_ready"] is True
    assert payload["tables_count"] >= 3


def test_schema_endpoint_excludes_internal_tables(client):
    _, headers = _register_and_headers(client, "schema@example.com")
    response = client.get("/api/v1/schema", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    table_names = {tbl["name"] for tbl in payload["tables"]}
    assert {"customers", "products", "orders"}.issubset(table_names)
    assert "schema_embeddings" not in table_names
    assert "users" not in table_names
    assert "Table: orders" in payload["schema_text"]


def test_reindex_endpoint_rebuilds_embeddings(client, monkeypatch):
    _, headers = _register_and_headers(client, "reindex@example.com")
    _, embed_calls = _install_mock_genai(monkeypatch)
    response = client.post("/api/v1/schema/reindex", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_tables"] >= 3
    assert set(payload["indexed_tables"]) >= {"customers", "products", "orders"}
    assert len(embed_calls) >= 3


def test_query_endpoint_executes_sql_builds_chart_and_returns_retrieved_tables(client, monkeypatch):
    _, headers = _register_and_headers(client, "chart@example.com")
    llm_calls, _ = _install_mock_genai(
        monkeypatch,
        [
            "```sql\nSELECT region, COUNT(*) AS customer_count FROM customers GROUP BY region ORDER BY customer_count DESC;\n```"
        ],
    )

    response = client.post(
        "/api/v1/query",
        headers=headers,
        json={"question": "How many customers per region?"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "SUCCESS"
    assert payload["cache_hit"] is False
    assert payload["model"] == settings.primary_model
    assert payload["corrected"] is False
    assert payload["retrieved_tables"] == ["customers"]
    assert payload["columns"] == ["region", "customer_count"]
    assert payload["row_count"] >= 1
    assert payload["chart_spec"]["chart_type"] == "bar"
    assert payload["plotly_json"] is not None
    assert "data" in payload["plotly_json"]
    assert len(llm_calls) == 1
    assert "CREATE TABLE customers" in llm_calls[0][1]
    assert "CREATE TABLE products" not in llm_calls[0][1]


def test_query_endpoint_rejects_non_readonly_sql(client, monkeypatch):
    _, headers = _register_and_headers(client, "guard@example.com")
    _install_mock_genai(
        monkeypatch,
        [
            "DROP TABLE orders;",
            "DELETE FROM orders;",
        ],
    )

    response = client.post(
        "/api/v1/query",
        headers=headers,
        json={"question": "Delete all orders"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "ERROR"
    assert "Security Error" in payload["error"]

    df_ord, err = execute_sql("SELECT COUNT(*) AS cnt FROM orders;")
    assert err is None and df_ord.iloc[0]["cnt"] == 500


def test_query_endpoint_two_stage_self_correction(client, monkeypatch):
    _, headers = _register_and_headers(client, "repair@example.com")
    llm_calls, _ = _install_mock_genai(
        monkeypatch,
        [
            "SELECT non_existent_column FROM orders;",
            "SELECT COUNT(*) AS total_orders FROM orders;",
        ],
    )

    response = client.post(
        "/api/v1/query",
        headers=headers,
        json={"question": "How many orders are there?"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "SUCCESS"
    assert payload["corrected"] is True
    assert payload["retrieved_tables"] == ["orders"]
    assert payload["rows"] == [{"total_orders": 500}]
    assert len(llm_calls) == 2
    assert "non_existent_column" in llm_calls[1][1]


def test_query_endpoint_503_exponential_backoff_and_fallback(client, monkeypatch):
    _, headers = _register_and_headers(client, "backoff@example.com")
    sleep_calls: list[int] = []
    monkeypatch.setattr("backend.app.core.llm_client.time.sleep", lambda sec: sleep_calls.append(sec))

    err_503 = errors.ServerError(503, {"error": {"message": "503 UNAVAILABLE"}})
    llm_calls, _ = _install_mock_genai(
        monkeypatch,
        [
            err_503,
            err_503,
            err_503,
            "SELECT COUNT(*) AS total_customers FROM customers;",
        ],
    )

    response = client.post(
        "/api/v1/query",
        headers=headers,
        json={"question": "Count customers"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "SUCCESS"
    assert payload["model"] == settings.fallback_model
    assert payload["rows"] == [{"total_customers": 100}]
    assert sleep_calls == [1, 2]
    assert [model for model, _ in llm_calls] == [
        settings.primary_model,
        settings.primary_model,
        settings.primary_model,
        settings.fallback_model,
    ]


def test_query_endpoint_short_circuits_insufficient_schema_before_sql_llm(client, monkeypatch):
    _, headers = _register_and_headers(client, "rag@example.com")
    llm_calls, embed_calls = _install_mock_genai(
        monkeypatch,
        ["SELECT 'SHOULD_NOT_BE_CALLED';"],
    )

    response = client.post(
        "/api/v1/query",
        headers=headers,
        json={"question": "What is our employee payroll?"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "INSUFFICIENT_SCHEMA"
    assert payload["retrieved_tables"] == []
    assert payload["sql_query"] == ""
    assert payload["model"] is None
    assert payload["rows"] == []
    assert payload["plotly_json"] is None
    assert len(embed_calls) == 1
    assert len(llm_calls) == 0


def test_upload_endpoint_prefixes_protected_table_with_tenant_id_and_upserts_embedding(client):
    user_id, headers = _register_and_headers(client, "upload@example.com")
    csv_bytes = b"id,name\n1,OverrideAttempt\n2,SecondRow\n"
    response = client.post(
        "/api/v1/upload",
        headers=headers,
        files={"file": ("customers.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()

    expected_table = f"u_{user_id}_customers"
    assert payload["table_name"] == expected_table
    assert payload["embedding_indexed"] is True
    assert payload["row_count"] == 2
    assert len(payload["preview"]) == 2
    table_names = {tbl["name"] for tbl in payload["tables"]}
    assert "customers" in table_names
    assert expected_table in table_names

    with Session(get_engine()) as session:
        stored = session.scalar(
            select(SchemaEmbedding).where(SchemaEmbedding.table_name == expected_table)
        )
        assert stored is not None
        assert stored.owner_id == user_id
        assert f"CREATE TABLE {expected_table}" in stored.ddl_text

    df_core, err = execute_sql("SELECT COUNT(*) AS cnt FROM customers;")
    assert err is None and df_core.iloc[0]["cnt"] == 100


def test_upload_endpoint_custom_table_becomes_discoverable_by_schema_rag(client, monkeypatch):
    user_id, headers = _register_and_headers(client, "custom@example.com")
    expected_table = f"u_{user_id}_monthly_targets"
    _install_mock_genai(
        monkeypatch,
        [f"SELECT month, target FROM {expected_table} ORDER BY month;"],
    )

    csv_bytes = b"month,target\nJan,1000\nFeb,1200\n"
    upload_resp = client.post(
        "/api/v1/upload",
        headers=headers,
        files={"file": ("Monthly Targets.csv", csv_bytes, "text/csv")},
    )
    assert upload_resp.status_code == 200
    assert upload_resp.json()["table_name"] == expected_table
    assert upload_resp.json()["embedding_indexed"] is True

    query_resp = client.post(
        "/api/v1/query",
        headers=headers,
        json={"question": "Show monthly targets"},
    )
    assert query_resp.status_code == 200
    query_payload = query_resp.json()
    assert query_payload["status"] == "SUCCESS"
    assert query_payload["retrieved_tables"] == [expected_table]
    assert query_payload["row_count"] == 2


def test_frontend_has_no_direct_sqlalchemy_or_gemini_imports():
    frontend_source = Path("frontend/app.py").read_text(encoding="utf-8")
    assert "sqlalchemy" not in frontend_source.lower()
    assert "google.genai" not in frontend_source.lower()
    assert "google.generativeai" not in frontend_source.lower()
