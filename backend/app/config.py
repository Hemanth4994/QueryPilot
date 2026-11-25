"""Application configuration loaded via Pydantic Settings."""

from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def _parse_cors_origins(value: str | list[str]) -> list[str]:
    """Accept either a JSON list or a comma-separated string for CORS origins."""
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or ["*"]
    return value


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=(".env", str(ENV_FILE)),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "gemini_api_key"),
    )
    database_url: str = Field(
        default="sqlite:///mock_business.sqlite3",
        validation_alias=AliasChoices("DATABASE_URL", "DB_URL", "database_url"),
    )
    api_host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("API_HOST", "api_host"),
    )
    port: int = Field(
        default=8000,
        validation_alias=AliasChoices("PORT", "port"),
    )
    cors_origins: Annotated[list[str], BeforeValidator(_parse_cors_origins)] = Field(
        default_factory=lambda: ["*"],
        validation_alias=AliasChoices("CORS_ORIGINS", "cors_origins"),
    )
    primary_model: str = "gemini-3.6-flash"
    fallback_model: str = "gemini-3.5-flash"
    max_retries: int = 3


settings = Settings()
