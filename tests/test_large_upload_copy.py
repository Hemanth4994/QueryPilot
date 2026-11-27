"""Tests for 2 GB upload limits, disk streaming, PostgreSQL COPY, auto-indexing, and ANALYZE."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
from fastapi.testclient import TestClient
import pandas as pd
import pytest

from backend.app.config import settings
from backend.app.core import cache, ingestion
from backend.app.core.database import configure_database
from backend.app.core.llm_client import reset_genai_client
from backend.app.main import app
from backend.app.worker import WorkerSettings, run_async_upload_job, set_arq_pool_for_tests


@pytest.fixture
def upload_env(tmp_path, monkeypatch):
    """Isolated SQLite DB, FakeRedis, stub ARQ pool, and temp upload staging dir."""
    prev_url = settings.database_url
    prev_threshold = settings.async_upload_threshold_bytes
    prev_max = settings.max_upload_size_bytes
    prev_upload_dir = settings.upload_dir

    db_file = tmp_path / "upload_test.sqlite3"
    upload_dir = tmp_path / "staged_uploads"
    settings.database_url = f"sqlite:///{db_file.as_posix()}"
    settings.upload_dir = str(upload_dir)
    reset_genai_client()

    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cache.set_redis_client_for_tests(fake_redis)

    enqueued: list[tuple[str, tuple]] = []
    stub_arq = MagicMock()

    async def fake_enqueue(func_name: str, *args):
        enqueued.append((func_name, args))
        return SimpleNamespace(job_id=args[0])

    stub_arq.enqueue_job = AsyncMock(side_effect=fake_enqueue)
    set_arq_pool_for_tests(stub_arq)

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = lambda *, model, contents, config=None: (
        SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0, 0.0])])
    )
    monkeypatch.setattr("backend.app.core.llm_client.get_genai_client", lambda: mock_client)

    try:
        with TestClient(app) as client:
            reg = client.post(
                "/api/v1/auth/register",
                json={"email": "stream@example.com", "password": "StrongPassword1!"},
            )
            assert reg.status_code == 201
            headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
            yield SimpleNamespace(
                client=client,
                user_id=reg.json()["user"]["id"],
                headers=headers,
                upload_dir=upload_dir,
                enqueued=enqueued,
            )
    finally:
        cache.set_redis_client_for_tests(None)
        set_arq_pool_for_tests(None)
        reset_genai_client()
        settings.database_url = prev_url
        settings.async_upload_threshold_bytes = prev_threshold
        settings.max_upload_size_bytes = prev_max
        settings.upload_dir = prev_upload_dir
        configure_database(prev_url)


def test_2gb_limits_and_worker_timeout_configured():
    assert settings.max_upload_size_bytes >= 2 * 1024 * 1024 * 1024
    assert settings.worker_job_timeout_seconds >= 1800
    assert WorkerSettings.job_timeout == settings.worker_job_timeout_seconds


def test_streaming_upload_stores_on_disk_cleans_up_and_enforces_413(upload_env):
    # 1. Sync upload parses via chunked path and cleans up staged disk file
    small_csv = b"order_id,region,revenue\n1,Europe,120.5\n2,Asia,340.0\n"
    sync_resp = upload_env.client.post(
        "/api/v1/upload",
        headers=upload_env.headers,
        files={"file": ("small_sales.csv", small_csv, "text/csv")},
    )
    assert sync_resp.status_code == 200
    assert sync_resp.json()["row_count"] == 2
    assert sync_resp.json()["ingestion_mode"] == "sqlite_chunked"
    assert list(upload_env.upload_dir.glob("*")) == []

    # 2. Large async upload passes disk path to worker (not base64) and unlinks after completion
    settings.async_upload_threshold_bytes = 10
    big_csv = b"customer_id,region,amount\n1,Europe,99.5\n2,Asia,150.25\n3,Europe,45.0\n"
    async_resp = upload_env.client.post(
        "/api/v1/upload",
        headers=upload_env.headers,
        files={"file": ("big_sales.csv", big_csv, "text/csv")},
    )
    assert async_resp.status_code == 202
    job_id = async_resp.json()["job_id"]

    func_name, job_args = upload_env.enqueued[-1]
    assert func_name == "run_async_upload_job"
    staged_file = Path(job_args[3])
    assert job_args[4] is True
    assert staged_file.exists()

    upload_env.client.portal.call(run_async_upload_job, {}, *job_args)
    assert not staged_file.exists()

    done = upload_env.client.get(f"/api/v1/jobs/{job_id}", headers=upload_env.headers)
    assert done.status_code == 200
    assert done.json()["status"] == "completed"
    assert done.json()["result"]["row_count"] == 3

    # 3. Upload exceeding MAX_UPLOAD_SIZE_BYTES fails cleanly with 413 and deletes partial temp file
    settings.max_upload_size_bytes = 20
    too_big = upload_env.client.post(
        "/api/v1/upload",
        headers=upload_env.headers,
        files={"file": ("too_big.csv", b"col1,col2\n" + b"a,b\n" * 20, "text/csv")},
    )
    assert too_big.status_code == 413
    assert list(upload_env.upload_dir.glob("*")) == []


def test_postgres_copy_auto_indexing_analyze_and_column_widening(tmp_path):
    csv_file = tmp_path / "big_demo.csv"
    csv_file.write_text(
        "order_id,order_date,region,qty,revenue,notes\n"
        "1,2026-01-01,Europe,5,100.5,First\n"
        "2,2026-01-02,Asia,8,220.0,Second\n"
        "3,2026-01-03,Europe,2,50.0,Third\n",
        encoding="utf-8",
    )
    df_sample = pd.read_csv(csv_file)
    col_types = ingestion._infer_pg_column_types(df_sample)
    assert col_types["order_id"] == "BIGINT"
    assert col_types["order_date"] == "TIMESTAMP"
    assert col_types["revenue"] == "DOUBLE PRECISION"

    selected_indexes = ingestion._select_index_columns(df_sample, col_types)
    assert selected_indexes == ["order_id", "order_date", "region"]
    assert "notes" not in selected_indexes
    assert "revenue" not in selected_indexes

    executed_sql: list[str] = []
    copied_chunks: list[bytes] = []
    attempt_counter = {"copy_calls": 0}

    class FakeCopy:
        def write(self, data: bytes) -> None:
            copied_chunks.append(data)

    class FakeCursor:
        def __init__(self):
            self.rowcount = 3

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql: str) -> None:
            executed_sql.append(sql)

        @contextmanager
        def copy(self, copy_sql: str):
            executed_sql.append(copy_sql)
            attempt_counter["copy_calls"] += 1
            if attempt_counter["copy_calls"] == 1:
                raise ValueError(
                    'invalid input syntax for type bigint: "N/A"\n'
                    'CONTEXT:  COPY _stg_u_1_big_demo, line 2, column "qty": "N/A"'
                )
            yield FakeCopy()

    fake_raw = MagicMock()
    fake_raw.cursor.side_effect = lambda: FakeCursor()
    fake_eng = MagicMock()
    fake_eng.raw_connection.side_effect = lambda: fake_raw

    row_count, idx_cols, mode = ingestion._ingest_postgres_copy(
        file_path=csv_file,
        encoding="utf-8",
        table_name="u_1_big_demo",
        columns=list(df_sample.columns),
        df_sample=df_sample,
        eng=fake_eng,
    )
    assert row_count == 3
    assert mode == "postgres_copy"
    assert {"order_id", "order_date", "region"}.issubset(idx_cols)
    assert attempt_counter["copy_calls"] == 2
    assert any('"qty" TEXT' in sql for sql in executed_sql)
    assert any("COPY " in sql and "FORMAT csv" in sql for sql in executed_sql)
    assert any("CREATE INDEX IF NOT EXISTS" in sql for sql in executed_sql)
    assert any('ANALYZE "u_1_big_demo";' in sql for sql in executed_sql)
    assert b"order_id,order_date,region" in b"".join(copied_chunks)


def test_detect_csv_encoding_handles_multibyte_split_at_64k_boundary(tmp_path):
    csv_file = tmp_path / "split_utf8.csv"
    prefix = b"a," + b"x" * 65533  # total 65,535 bytes
    split_char = "é".encode()  # 2 bytes: straddles byte 65535 and 65536
    csv_file.write_bytes(prefix + split_char + b"\n")
    assert ingestion._detect_csv_encoding(csv_file) in ("utf-8-sig", "utf-8")
