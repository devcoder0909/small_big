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

    # AI API Settings & Failover Architecture
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_api_key_2: str = Field(default="", alias="GROQ_API_KEY_2")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_api_key_2: str = Field(default="", alias="OPENROUTER_API_KEY_2")
    openrouter_api_key_3: str = Field(default="", alias="OPENROUTER_API_KEY_3")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(
        default="meta-llama/llama-3.1-70b-instruct", alias="OPENROUTER_MODEL"
    )
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_api_key_2: str = Field(default="", alias="GEMINI_API_KEY_2")
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_api_key_2: str = Field(default="", alias="NVIDIA_API_KEY_2")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL"
    )
    nvidia_model: str = Field(
        default="nvidia/nemotron-3-ultra-550b-a55b", alias="NVIDIA_MODEL"
    )

    # AI Failover, Timeout & Rate-Limit Safety Controls
    ai_providers: str = Field(default="nvidia,openrouter,groq,gemini", alias="AI_PROVIDERS")
    ai_max_requests_per_cycle: int = Field(default=1, alias="AI_MAX_REQUESTS_PER_CYCLE")
    ai_timeout_seconds: float = Field(default=3.0, alias="AI_TIMEOUT_SECONDS")
    ai_max_retries: int = Field(default=1, alias="AI_MAX_RETRIES")
    ai_provider_cooldown_seconds: float = Field(default=60.0, alias="AI_PROVIDER_COOLDOWN_SECONDS")

    # CORS
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # Cache TTLs (seconds)
    cache_latest_ttl: int = Field(default=2, alias="CACHE_LATEST_TTL")
    cache_summary_ttl: int = Field(default=5, alias="CACHE_SUMMARY_TTL")
    cache_analytics_ttl: int = Field(default=15, alias="CACHE_ANALYTICS_TTL")

    # Retention & Analysis Window Depth
    raw_response_retention_days: int = Field(default=30, alias="RAW_RESPONSE_RETENTION_DAYS")
    max_game_results_retention: int = Field(default=10000, alias="MAX_GAME_RESULTS_RETENTION")
    analysis_history_window: int = Field(default=10000, alias="ANALYSIS_HISTORY_WINDOW")
    game_history_fetch_limit: int = Field(default=10000, alias="GAME_HISTORY_FETCH_LIMIT")
    prediction_analysis_window: int = Field(default=10000, alias="PREDICTION_ANALYSIS_WINDOW")

    # Monitoring
    health_degraded_threshold_seconds: int = Field(
        default=120, alias="HEALTH_DEGRADED_THRESHOLD_SECONDS"
    )

    # Selective High-Confluence Gating Settings
    confluence_min_agreement_pct: float = Field(default=65.0, alias="CONFLUENCE_MIN_AGREEMENT_PCT")
    confluence_max_entropy: float = Field(default=0.985, alias="CONFLUENCE_MAX_ENTROPY")
    confluence_min_agreeing_indicators: int = Field(default=4, alias="CONFLUENCE_MIN_AGREEING_INDICATORS")
    confluence_min_sample_size: int = Field(default=20, alias="CONFLUENCE_MIN_SAMPLE_SIZE")
    prediction_health_drift_threshold_pct: float = Field(default=55.0, alias="PREDICTION_HEALTH_DRIFT_THRESHOLD_PCT")
    prediction_health_recovery_threshold_pct: float = Field(default=58.0, alias="PREDICTION_HEALTH_RECOVERY_THRESHOLD_PCT")

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


def get_build_commit() -> str:
    """Return the git commit SHA (from commit env vars, git rev-parse, or fallback)."""
    import os
    import subprocess

    for env_name in ("BUILD_COMMIT", "NF_COMMIT_SHA", "NORTHFLANK_COMMIT_SHA", "COMMIT_SHA", "GIT_COMMIT"):
        env_commit = os.getenv(env_name, "").strip()
        if (
            env_commit
            and not env_commit.startswith("${")
            and env_commit.upper() not in ("BUILD_COMMIT", "COMMIT_SHA", "GIT_COMMIT", "NF_COMMIT_SHA")
            and len(env_commit) >= 7
        ):
            return env_commit[:7]

    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass

    return "7d27e26"

