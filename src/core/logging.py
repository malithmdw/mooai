"""Structured logging setup.

All application logging must go through `get_logger`. Do not use `print()`
or the bare `logging` module elsewhere in the codebase — see CLAUDE.md
"Logging requirements".
"""

from __future__ import annotations

import logging
import sys

from src.core.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure the root logger once, based on application settings.

    Idempotent: safe to call multiple times (e.g. from tests and from the
    app entrypoint) without installing duplicate handlers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s "
            "[request_id=%(request_id)s user=%(user)s] %(message)s",
            defaults={"request_id": "-", "user": "-"},
        )
    )

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers = [handler]
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use.

    Usage: `logger = get_logger(__name__)` at module scope.
    """
    configure_logging()
    return logging.getLogger(name)
