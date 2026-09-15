"""FastAPI application entrypoint.

Run locally with:

    uvicorn src.api.main:app --reload

No business functionality is wired up yet beyond the health check — this
is the foundation commit only.
"""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import health
from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.observability import configure_tracing

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Application factory.

    Using a factory (rather than a module-level `app`) keeps startup
    explicit and makes it straightforward to build isolated app instances
    in tests.
    """
    configure_logging()
    configure_tracing()
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Enterprise AI Assistant POC — foundation build.",
        version="0.1.0",
    )

    app.include_router(health.router)

    logger.info("Application initialized (env=%s)", settings.app_env)
    return app


app = create_app()
