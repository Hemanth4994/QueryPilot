"""FastAPI integration tests for QueryPilot endpoints (offline with mocked Gemini)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from google.genai import errors
import pytest

from backend.app.config import settings
from backend.app.core.database import configure_database
from backend.app.core.llm_client import (
    FALLBACK_MODEL,
    PRIMARY_MODEL,
    reset_genai_client,
)
from backend.app.core.sql_guard import execute_sql
from backend.app.main import app


@pytest.fixture
def client(tmp_path):
    """Provide a TestClient wired to a fresh isolated SQLite file per test."""
    prev_url = settings.database_url
    db_file = tmp_path / "api_test.sqlite3"
    settings.database_url = f"sqlite:///{db_file.as_posix()}"
    reset_genai_client()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        reset_genai_client()
        settings.database_url = prev_url
        configure_database(prev_url)


def _mock_gemini_sequence(monkeypatch, responses: list[str | Exception]):
    """Patch genai client factory with deterministic scripted responses or exceptions."""
    mock_client = MagicMock()
    calls: list[tuple[str, str]] = []
    iterator = iter(responses)

    def fake_generate_content(*, model: str, contents: str):
        calls.append((model, contents))
        next_item = next(iterator)
        if isinstance(next_item, Exception):
            raise next_item
        return SimpleNamespace(text=next_item)

    mock_client.models.generate_content.side_effect = fake_generate_content
    monkeypatch.setattr("backend.app.core.llm_client.get_genai_client", lambda: mock_client)
    return calls


def test_health_endpoint_returns_ready(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database_ready"] is True
    assert payload["tables_count"] >= 3


def test_schema_endpoint_returns_seeded_tables(client):
    response = client.get("/api/v1/schema")
    assert response.status_code == 200
    payload = response.json()
    table_names = {tbl["name"] for tbl in payload["tables"]}
    assert {"customers", "products", "orders"}.issubset(table_names)
    assert "Table: orders" in payload["schema_text"]


def test_query_endpoint_executes_sql_and_builds_plotly_chart(client, monkeypatch):
    calls = _mock_gemini_sequence(
        monkeypatch,
        [
            "```sql\nSELECT region, COUNT(*) AS customer_count FROM customers GROUP BY region ORDER BY customer_count DESC;\n```"
        ],
    )

    response = client.post("/api/v1/query", json={"question": "How many customers per region?"})
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "SUCCESS"
    assert payload["model"] == PRIMARY_MODEL
    assert payload["corrected"] is False
    assert payload["columns"] == ["region", "customer_count"]
    assert payload["row_count"] >= 1
    assert payload["chart_spec"]["chart_type"] == "bar"
    assert payload["plotly_json"] is not None
    assert "data" in payload["plotly_json"]
    assert len(calls) == 1


def test_query_endpoint_rejects_non_readonly_sql(client, monkeypatch):
    _mock_gemini_sequence(
        monkeypatch,
        [
            "DROP TABLE orders;",
            "DELETE FROM orders;",
        ],
    )

    response = client.post("/api/v1/query", json={"question": "Delete all orders"})
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "ERROR"
    assert "Security Error" in payload["error"]

    df_ord, err = execute_sql("SELECT COUNT(*) AS cnt FROM orders;")
    assert err is None and df_ord.iloc[0]["cnt"] == 500


def test_query_endpoint_two_stage_self_correction(client, monkeypatch):
    calls = _mock_gemini_sequence(
        monkeypatch,
        [
            "SELECT non_existent_column FROM orders;",
            "SELECT COUNT(*) AS total_orders FROM orders;",
        ],
    )

    response = client.post("/api/v1/query", json={"question": "How many orders are there?"})
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "SUCCESS"
    assert payload["corrected"] is True
    assert payload["rows"] == [{"total_orders": 500}]
    assert len(calls) == 2
    assert "non_existent_column" in calls[1][1]


def test_query_endpoint_503_exponential_backoff_and_fallback(client, monkeypatch):
    sleep_calls: list[int] = []
    monkeypatch.setattr("backend.app.core.llm_client.time.sleep", lambda sec: sleep_calls.append(sec))

    err_503 = errors.ServerError(503, {"error": {"message": "503 UNAVAILABLE"}})
    calls = _mock_gemini_sequence(
        monkeypatch,
        [
            err_503,
            err_503,
            err_503,
            "SELECT COUNT(*) AS total_customers FROM customers;",
        ],
    )

    response = client.post("/api/v1/query", json={"question": "Count customers"})
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "SUCCESS"
    assert payload["model"] == FALLBACK_MODEL
    assert payload["rows"] == [{"total_customers": 100}]
    assert sleep_calls == [1, 2]
    assert [model for model, _ in calls] == [
        PRIMARY_MODEL,
        PRIMARY_MODEL,
        PRIMARY_MODEL,
        FALLBACK_MODEL,
    ]


def test_query_endpoint_insufficient_schema(client, monkeypatch):
    _mock_gemini_sequence(
        monkeypatch,
        ["SELECT 'INSUFFICIENT_SCHEMA' AS error;"],
    )

    response = client.post("/api/v1/query", json={"question": "What is our employee payroll?"})
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "INSUFFICIENT_SCHEMA"
    assert payload["rows"] == []
    assert payload["plotly_json"] is None


def test_upload_endpoint_prefixes_protected_table_name(client):
    csv_bytes = b"id,name\n1,OverrideAttempt\n2,SecondRow\n"
    response = client.post(
        "/api/v1/upload",
        files={"file": ("customers.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["table_name"] == "user_customers"
    assert payload["row_count"] == 2
    assert len(payload["preview"]) == 2
    table_names = {tbl["name"] for tbl in payload["tables"]}
    assert "customers" in table_names
    assert "user_customers" in table_names

    df_core, err = execute_sql("SELECT COUNT(*) AS cnt FROM customers;")
    assert err is None and df_core.iloc[0]["cnt"] == 100


def test_upload_endpoint_allows_custom_table_name(client):
    csv_bytes = b"month,target\nJan,1000\nFeb,1200\n"
    response = client.post(
        "/api/v1/upload",
        files={"file": ("Monthly Targets.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["table_name"] == "monthly_targets"
    assert payload["row_count"] == 2


def test_frontend_has_no_direct_sqlalchemy_or_gemini_imports():
    frontend_source = Path("frontend/app.py").read_text(encoding="utf-8")
    assert "sqlalchemy" not in frontend_source.lower()
    assert "google.genai" not in frontend_source.lower()
    assert "google.generativeai" not in frontend_source.lower()
