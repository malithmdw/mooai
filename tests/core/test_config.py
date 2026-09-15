"""Tests for `src.core.config.Settings`.

`Settings(_env_file=None)` is used throughout so each test is driven only
by the environment variables it (or the `_default_settings_env` autouse
fixture in `tests/conftest.py`) sets, ignoring any local `.env` file that
might exist on a developer's machine.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import Settings

_PROVIDER_CREDENTIAL_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "LANGSMITH_API_KEY",
)


def _clear_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in _PROVIDER_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


class TestValidConfiguration:
    """A fully specified environment produces the expected typed settings."""

    def test_loads_full_configuration_from_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_NAME", "acme-assistant")
        monkeypatch.setenv("APP_ENV", "prod")
        monkeypatch.setenv("LOG_LEVEL", "debug")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.setenv("PINECONE_API_KEY", "pc-test")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "acme-knowledge")
        monkeypatch.setenv("PINECONE_NAMESPACE", "team-a")
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
        monkeypatch.setenv("LANGSMITH_PROJECT", "acme-assistant")
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        monkeypatch.setenv("POSTGRES_URL", "postgresql+asyncpg://u:p@db:5432/acme")
        monkeypatch.setenv("REDIS_URL", "redis://cache:6379/1")
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", "250")
        monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "30")
        monkeypatch.setenv("MODEL_NAME", "claude-opus-5")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

        settings = Settings(_env_file=None)

        assert settings.app_name == "acme-assistant"
        assert settings.app_env == "prod"
        assert settings.log_level == "DEBUG"
        assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test"
        assert settings.openai_api_key.get_secret_value() == "sk-openai-test"
        assert settings.pinecone_api_key.get_secret_value() == "pc-test"
        assert settings.pinecone_index_name == "acme-knowledge"
        assert settings.pinecone_namespace == "team-a"
        assert settings.langsmith_api_key.get_secret_value() == "ls-test"
        assert settings.langsmith_project == "acme-assistant"
        assert settings.langsmith_tracing is False
        assert settings.postgres_url == "postgresql+asyncpg://u:p@db:5432/acme"
        assert settings.redis_url == "redis://cache:6379/1"
        assert settings.rate_limit_requests == 250
        assert settings.rate_limit_window_seconds == 30
        assert settings.model_name == "claude-opus-5"
        assert settings.embedding_model == "text-embedding-3-small"

    def test_secrets_are_not_exposed_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret")

        settings = Settings(_env_file=None)

        assert "sk-ant-super-secret" not in repr(settings)
        assert "sk-ant-super-secret" not in str(settings)


class TestMissingRequiredConfiguration:
    """Required configuration that is absent fails fast and explicitly."""

    def test_missing_postgres_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("POSTGRES_URL", raising=False)

        with pytest.raises(ValidationError, match="postgres_url"):
            Settings(_env_file=None)

    def test_missing_redis_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REDIS_URL", raising=False)

        with pytest.raises(ValidationError, match="redis_url"):
            Settings(_env_file=None)

    def test_non_local_env_without_provider_credentials_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_ENV", "prod")
        _clear_provider_credentials(monkeypatch)

        with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
            Settings(_env_file=None)

    def test_error_names_every_missing_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "staging")
        _clear_provider_credentials(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)

        message = str(exc_info.value)
        assert "OPENAI_API_KEY" in message
        assert "PINECONE_API_KEY" in message
        assert "LANGSMITH_API_KEY" in message


class TestDevelopmentConfiguration:
    """`APP_ENV=local` is exempt from the provider-credential requirement."""

    def test_local_env_boots_without_provider_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        _clear_provider_credentials(monkeypatch)

        settings = Settings(_env_file=None)

        assert settings.app_env == "local"
        assert settings.anthropic_api_key.get_secret_value() == ""
        assert settings.openai_api_key.get_secret_value() == ""
        assert settings.pinecone_api_key.get_secret_value() == ""
        assert settings.langsmith_api_key.get_secret_value() == ""

    def test_local_env_is_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("APP_ENV", raising=False)
        _clear_provider_credentials(monkeypatch)

        settings = Settings(_env_file=None)

        assert settings.app_env == "local"
        assert settings.rate_limit_requests == 100
        assert settings.rate_limit_window_seconds == 60
        assert settings.model_name == "claude-opus-5"
        assert settings.embedding_model == "text-embedding-3-large"
