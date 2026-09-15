"""Application configuration.

Settings are loaded from environment variables (and a local `.env` file in
development) via `pydantic-settings`. This is the single source of truth for
configuration — no other module should call `os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_env: Literal["local", "dev", "staging", "prod"] = "local"
    app_name: str = "enterprise-ai-assistant"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Hardcoded POC auth (see src/security/auth.py) ---
    poc_auth_enabled: bool = True
    poc_admin_username: str = "admin"
    poc_admin_password: str = "change-me"

    # --- Anthropic Claude ---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5"

    # --- OpenAI embeddings ---
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-large"

    # --- Pinecone ---
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "enterprise-knowledge"

    # --- PostgreSQL ---
    database_url: str = "postgresql+asyncpg://ai_assistant:change-me@localhost:5432/ai_assistant"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- MCP ---
    mcp_server_host: str = "localhost"
    mcp_server_port: int = 8765

    # --- LangSmith ---
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "enterprise-ai-assistant"


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance.

    Cached so configuration is parsed once per process; tests may bypass this
    by constructing `Settings(...)` directly.
    """
    return Settings()
