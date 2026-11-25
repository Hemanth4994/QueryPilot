"""Shared pytest fixtures ensuring tests run against isolated temporary SQLite databases."""

import pytest

from backend.app.config import settings
from backend.app.core.database import check_and_seed_db, configure_database
from backend.app.core.llm_client import reset_genai_client


@pytest.fixture(scope="session", autouse=True)
def isolated_session_db(tmp_path_factory):
    """Point settings.database_url to a temporary SQLite file for the test session."""
    original_url = settings.database_url
    db_file = tmp_path_factory.mktemp("querypilot_db") / "session_test.sqlite3"
    settings.database_url = f"sqlite:///{db_file.as_posix()}"
    configure_database(settings.database_url)
    check_and_seed_db()
    yield settings.database_url
    reset_genai_client()
    settings.database_url = original_url
    configure_database(original_url)
