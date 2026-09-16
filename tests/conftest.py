"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from src.core.config import get_settings


def pytest_configure(config: pytest.Config) -> None:
    """Ensure required settings exist before any test module is imported.

    Several modules (e.g. `src.api.main`, `src.observability.tracing`)
    call `get_logger(__name__)` at import time, which calls
    `configure_logging()` -> `get_settings()` — so `Settings`' required
    fields (`POSTGRES_URL`, `REDIS_URL`) must already be present in the
    environment before pytest imports the *first* test module, earlier
    than even an autouse fixture can run. `pytest_configure` is the one
    hook guaranteed to run before collection. Individual tests still
    override these via `monkeypatch` as needed (see
    `_default_settings_env` below).
    """
    del config  # required by the pytest_configure hook signature, unused here
    os.environ.setdefault("POSTGRES_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture(autouse=True)
def _default_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide the minimum required configuration for every test.

    `POSTGRES_URL` and `REDIS_URL` have no defaults (see
    `src.core.config.Settings`), so every test needs them present unless it
    is specifically exercising the "missing required configuration" case,
    in which case it can `monkeypatch.delenv(...)` them within the same
    test — `monkeypatch` fixtures are shared across a test's dependency
    graph. The `get_settings()` cache is cleared before and after so no
    test observes a value cached by an earlier one.
    """
    monkeypatch.setenv("POSTGRES_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A `TestClient` bound to a freshly created app instance per test.

    `create_app` is imported here rather than at module scope: importing
    `src.api.main` triggers its module-level `app = create_app()`, which
    reads settings — that must happen after `_default_settings_env` has
    set the required environment variables, not at conftest import time.
    """
    from src.api.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client() -> AsyncIterator[httpx.AsyncClient]:
    """An `httpx.AsyncClient` bound to a freshly created app instance per test.

    For integration tests exercising the app's async endpoints end-to-end
    over ASGI, rather than through `TestClient`'s sync wrapper. See
    `client` for why `create_app` is imported inside the fixture body
    rather than at module scope.
    """
    from src.api.main import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
