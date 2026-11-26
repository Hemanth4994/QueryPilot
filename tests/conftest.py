"""Shared pytest fixtures ensuring offline tests run against isolated temporary DBs."""

import os

import pytest

from backend.app.config import settings
from backend.app.core.auth import ensure_users_table
from backend.app.core.cache import set_redis_client_for_tests
from backend.app.core.database import check_and_seed_db, configure_database
from backend.app.core.llm_client import reset_genai_client
from backend.app.worker import set_arq_pool_for_tests


@pytest.fixture(scope="session", autouse=True)
def isolated_session_db(tmp_path_factory):
    """Point settings to temp DB + test-only JWT secret and prevent live API calls."""
    original_url = settings.database_url
    original_read_url = settings.read_database_url
    original_key = settings.gemini_api_key
    original_jwt_secret = settings.jwt_secret_key
    original_env_key = os.environ.pop("GEMINI_API_KEY", None)

    db_file = tmp_path_factory.mktemp("querypilot_db") / "session_test.sqlite3"
    settings.database_url = f"sqlite:///{db_file.as_posix()}"
    settings.read_database_url = ""
    settings.gemini_api_key = ""
    settings.jwt_secret_key = "test-only-ephemeral-jwt-signing-key-32-bytes"
    configure_database(settings.database_url)
    check_and_seed_db()
    ensure_users_table()

    yield settings.database_url

    set_redis_client_for_tests(None)
    set_arq_pool_for_tests(None)
    reset_genai_client()
    settings.database_url = original_url
    settings.read_database_url = original_read_url
    settings.gemini_api_key = original_key
    settings.jwt_secret_key = original_jwt_secret
    if original_env_key is not None:
        os.environ["GEMINI_API_KEY"] = original_env_key
    configure_database(original_url)
