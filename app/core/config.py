"""Core configuration module using pydantic-settings."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    app_env: str = Field(default="development", alias="APP_ENV")

    # Source API
    source_url: str = Field(
        default="https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
        alias="SOURCE_URL",
    )
    source_api_url: str = Field(
        default="https://api.ar-lottery01.com/api/Lottery/GetHistoryIssuePage",
        alias="SOURCE_API_URL",
    )

    # Collector
    poll_interval_seconds: int = Field(default=3, alias="POLL_INTERVAL_SECONDS")
    request_timeout_seconds: int = Field(default=10, alias="REQUEST_TIMEOUT_SECONDS")
    max_retries: int = Field(default=5, alias="MAX_RETRIES")
    backoff_base_seconds: float = Field(default=1.0, alias="BACKOFF_BASE_SECONDS")
    backoff_max_seconds: float = Field(default=30.0, alias="BACKOFF_MAX_SECONDS")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/wingo_db",
        alias="DATABASE_URL",
    )
    database_sync_url: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/wingo_db",
        alias="DATABASE_SYNC_URL",
    )
    db_pool_size: int = Field(default=10, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, alias="DB_MAX_OVERFLOW")
    db_pool_recycle: int = Field(default=3600, alias="DB_POOL_RECYCLE")

    # API Security
    api_key: str = Field(default="dev-api-key-change-me", alias="API_KEY")

    # AI API Keys for Pattern Rotation
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_api_key_2: str = Field(default="", alias="GROQ_API_KEY_2")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_api_key_2: str = Field(default="", alias="OPENROUTER_API_KEY_2")
    openrouter_api_key_3: str = Field(default="", alias="OPENROUTER_API_KEY_3")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # CORS
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # Cache TTLs (seconds)
    cache_latest_ttl: int = Field(default=2, alias="CACHE_LATEST_TTL")
    cache_summary_ttl: int = Field(default=5, alias="CACHE_SUMMARY_TTL")
    cache_analytics_ttl: int = Field(default=15, alias="CACHE_ANALYTICS_TTL")

    # Retention
    raw_response_retention_days: int = Field(default=30, alias="RAW_RESPONSE_RETENTION_DAYS")

    # Monitoring
    health_degraded_threshold_seconds: int = Field(
        default=120, alias="HEALTH_DEGRADED_THRESHOLD_SECONDS"
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
