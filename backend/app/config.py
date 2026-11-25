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
        default="postgresql+psycopg://querypilot:querypilot@localhost:5432/querypilot",
        validation_alias=AliasChoices("DATABASE_URL", "DB_URL", "database_url"),
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices("REDIS_URL", "redis_url"),
    )
    jwt_secret_key: str = Field(
        default="",
        validation_alias=AliasChoices("JWT_SECRET_KEY", "jwt_secret_key"),
    )
    jwt_algorithm: str = Field(
        default="HS256",
        validation_alias=AliasChoices("JWT_ALGORITHM", "jwt_algorithm"),
    )
    jwt_expire_minutes: int = Field(
        default=60,
        ge=1,
        validation_alias=AliasChoices("JWT_EXPIRE_MINUTES", "jwt_expire_minutes"),
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=1,
        validation_alias=AliasChoices("CACHE_TTL_SECONDS", "cache_ttl_seconds"),
    )
    rate_limit_per_minute: int = Field(
        default=30,
        ge=1,
        validation_alias=AliasChoices("RATE_LIMIT_PER_MINUTE", "rate_limit_per_minute"),
    )
    async_upload_threshold_bytes: int = Field(
        default=524288,
        ge=1,
        validation_alias=AliasChoices(
            "ASYNC_UPLOAD_THRESHOLD_BYTES",
            "async_upload_threshold_bytes",
        ),
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
    primary_model: str = Field(
        default="gemini-3.6-flash",
        validation_alias=AliasChoices("PRIMARY_MODEL", "primary_model"),
    )
    fallback_model: str = Field(
        default="gemini-3.5-flash",
        validation_alias=AliasChoices("FALLBACK_MODEL", "fallback_model"),
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        validation_alias=AliasChoices("MAX_RETRIES", "max_retries"),
    )
    embedding_model: str = Field(
        default="gemini-embedding-001",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model"),
    )
    schema_rag_top_k: int = Field(
        default=5,
        ge=1,
        validation_alias=AliasChoices("SCHEMA_RAG_TOP_K", "schema_rag_top_k"),
    )
    schema_rag_min_similarity: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("SCHEMA_RAG_MIN_SIMILARITY", "schema_rag_min_similarity"),
    )
    query_timeout_ms: int = Field(
        default=10000,
        ge=100,
        validation_alias=AliasChoices("QUERY_TIMEOUT_MS", "query_timeout_ms"),
    )
    max_query_rows: int = Field(
        default=1000,
        ge=1,
        validation_alias=AliasChoices("MAX_QUERY_ROWS", "max_query_rows"),
    )
    db_pool_size: int = Field(
        default=5,
        ge=1,
        validation_alias=AliasChoices("DB_POOL_SIZE", "db_pool_size"),
    )
    db_max_overflow: int = Field(
        default=10,
        ge=0,
        validation_alias=AliasChoices("DB_MAX_OVERFLOW", "db_max_overflow"),
    )
    enable_tracing: bool = Field(
        default=False,
        validation_alias=AliasChoices("ENABLE_TRACING", "enable_tracing"),
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="",
        validation_alias=AliasChoices(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "otel_exporter_otlp_endpoint",
        ),
    )
    otel_service_name: str = Field(
        default="querypilot-backend",
        validation_alias=AliasChoices("OTEL_SERVICE_NAME", "otel_service_name"),
    )


settings = Settings()
