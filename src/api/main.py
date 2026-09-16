"""FastAPI application entrypoint.

Run locally with:

    uvicorn src.api.main:app --reload

No LLM/agent functionality is implemented yet — `/api/v1/chat` returns a
placeholder response (see `src.api.routes.chat`), and
`/api/v1/conversations/{conversation_id}` reports `501 Not Implemented`
(see `src.api.routes.conversations`). This module wires up the app's
cross-cutting concerns (CORS, request-ID middleware, global exception
handling, logging, tracing) and routes.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.exception_handlers import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from src.api.middleware import RequestContextMiddleware
from src.api.routes import chat, conversations, health, ready
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

    # Middleware order matters: Starlette applies `add_middleware` calls
    # outermost-first, so CORS (added first) wraps everything — including
    # error responses — and RequestContextMiddleware (added second) still
    # sits outside FastAPI's exception-handling layer, so it sees the
    # final resolved status code for every request, success or error.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    # `add_exception_handler`'s stub expects a handler generic over
    # `Exception`; ours are correctly narrowed to the specific exception
    # type each is registered for, which mypy can't reconcile structurally.
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

    app.include_router(health.router)
    app.include_router(ready.router)
    app.include_router(chat.router)
    app.include_router(conversations.router)

    logger.info("Application initialized (env=%s)", settings.app_env)
    return app


app = create_app()
