"""Application configuration.

Settings are loaded from environment variables (and a local `.env` file in
development) via `pydantic-settings`. This is the single source of truth
for configuration — no other module should call `os.environ` directly.

Settings are read lazily through `get_settings()`, never at import time:
importing this module (or any module that imports it) must not read the
environment, contact a database, or initialize any external service client
— see CLAUDE.md "Do not initialize external services during module
import".
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnv = Literal["local", "dev", "staging", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Environments in which the app is expected to talk to real external
# services and must therefore be configured with real provider credentials.
# "local" is exempt so the app (and its test suite) can boot without live
# secrets.
_ENVS_REQUIRING_PROVIDER_CREDENTIALS: frozenset[AppEnv] = frozenset({"dev", "staging", "prod"})

_REQUIRED_IN_NON_LOCAL_ENVS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "LANGSMITH_API_KEY",
)


class Settings(BaseSettings):
    """Typed application settings, populated from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "enterprise-ai-assistant"
    app_env: AppEnv = "local"
    log_level: LogLevel = "INFO"

    # --- Anthropic Claude ---
    anthropic_api_key: SecretStr = SecretStr("")

    # --- OpenAI (embeddings) ---
    openai_api_key: SecretStr = SecretStr("")

    # --- Pinecone ---
    pinecone_api_key: SecretStr = SecretStr("")
    pinecone_index_name: str = "enterprise-knowledge"
    pinecone_namespace: str = ""

    # --- LangSmith observability ---
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "enterprise-ai-assistant"
    langsmith_tracing: bool = True

    # --- Data stores (no default — must be supplied explicitly) ---
    postgres_url: str
    redis_url: str

    # --- Rate limiting ---
    rate_limit_requests: int = Field(default=100, gt=0)
    rate_limit_window_seconds: int = Field(default=60, gt=0)

    # --- Models ---
    model_name: str = "claude-opus-5"
    embedding_model: str = "text-embedding-3-large"

    # --- CORS ---
    # Least-privilege default: only the documented local Streamlit UI, not
    # a wildcard. Override per environment via a comma-separated list.
    cors_allowed_origins: tuple[str, ...] = ("http://localhost:8501",)

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        """Accept any case (`info`, `Info`, `INFO`) for LOG_LEVEL."""
        return value.upper() if isinstance(value, str) else value

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _split_comma_separated_origins(cls, value: object) -> object:
        """Accept `CORS_ALLOWED_ORIGINS` as a comma-separated env var string."""
        if isinstance(value, str):
            return tuple(origin.strip() for origin in value.split(",") if origin.strip())
        return value

    @model_validator(mode="after")
    def _require_provider_credentials_outside_local(self) -> Settings:
        """Fail fast if a non-local environment is missing real credentials.

        Local development (and the test suite) may boot with empty provider
        keys; any environment that talks to real infrastructure may not.
        This is the "safe startup validation" step — it runs when
        `Settings` is constructed, not when a provider client is later used.
        """
        if self.app_env not in _ENVS_REQUIRING_PROVIDER_CREDENTIALS:
            return self

        secrets_by_env_var = {
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            "OPENAI_API_KEY": self.openai_api_key,
            "PINECONE_API_KEY": self.pinecone_api_key,
            "LANGSMITH_API_KEY": self.langsmith_api_key,
        }
        missing = [
            env_var
            for env_var in _REQUIRED_IN_NON_LOCAL_ENVS
            if not secrets_by_env_var[env_var].get_secret_value()
        ]
        if missing:
            raise ValueError(
                f"Missing required configuration for APP_ENV={self.app_env!r}: {', '.join(missing)}"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached, validated `Settings` instance.

    Cached so configuration is parsed and validated once per process.
    Construct `Settings(...)` directly in tests to bypass the cache and
    exercise specific configurations.
    """
    return Settings()  # type: ignore[call-arg]
