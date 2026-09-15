"""Structured JSON logging.

All application logging must go through `get_logger` plus the `log_*`
helper functions in this module — never `print()`, never the bare
`logging` module configured ad hoc elsewhere — see CLAUDE.md "Logging
requirements".

Request/conversation/user/agent context is carried via `contextvars`
(`bind_log_context`), which — unlike thread-locals — propagates correctly
across `await` points and `asyncio.create_task`, so a request's context is
still attached to log records emitted deep inside async retrieval/agent
code without threading it through every function signature.

Never log secrets (API keys, passwords, authorization tokens) or complete
confidential document contents. `log_event` and friends sanitize known
sensitive field names automatically, but that is a safety net, not a
license to pass secrets in — callers must not do so deliberately.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime

from src.core.config import get_settings
from src.core.redaction import sanitize_fields as _sanitize_extra

_CONFIGURED = False

# --- Request/conversation/user/agent context propagation ------------------

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_conversation_id_var: ContextVar[str | None] = ContextVar("conversation_id", default=None)
_user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
_agent_name_var: ContextVar[str | None] = ContextVar("agent_name", default=None)


@contextmanager
def bind_log_context(
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    agent_name: str | None = None,
) -> Iterator[None]:
    """Bind log context for the duration of this `with` block.

    Only fields explicitly passed are set; omitted fields keep whatever an
    outer `bind_log_context` (if any) already set — so an API request
    handler can bind `request_id`/`conversation_id`/`user_id` once, and
    each agent node can nest a narrower `bind_log_context(agent_name=...)`
    without disturbing the outer values. Safe across `await`: each asyncio
    task gets its own copy of the context, and restores cleanly even if
    the block raises.
    """
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    for var, value in (
        (_request_id_var, request_id),
        (_conversation_id_var, conversation_id),
        (_user_id_var, user_id),
        (_agent_name_var, agent_name),
    ):
        if value is not None:
            tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def get_request_id() -> str | None:
    """Return the request ID bound by the innermost `bind_log_context`, if any."""
    return _request_id_var.get()


def get_conversation_id() -> str | None:
    """Return the conversation ID bound by the innermost `bind_log_context`, if any."""
    return _conversation_id_var.get()


def get_user_id() -> str | None:
    """Return the user ID bound by the innermost `bind_log_context`, if any."""
    return _user_id_var.get()


def get_agent_name() -> str | None:
    """Return the agent/node name bound by the innermost `bind_log_context`, if any."""
    return _agent_name_var.get()


class _ContextFilter(logging.Filter):
    """Attach the active log context to every record passing through it.

    Defense in depth for records emitted via a bare `logging.getLogger(...)`
    call that bypasses the `log_*` helpers (which already attach context
    themselves — see `log_event`): this filter, installed on the
    application's configured handler, fills in the same fields.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(record, "request_id", None) or _request_id_var.get()
        record.conversation_id = (
            getattr(record, "conversation_id", None) or _conversation_id_var.get()
        )
        record.user_id = getattr(record, "user_id", None) or _user_id_var.get()
        record.agent_name = getattr(record, "agent_name", None) or _agent_name_var.get()
        return True


# --- JSON formatting --------------------------------------------------------
#
# Sensitive-field redaction (`_sanitize_extra`, imported above from
# `src.core.redaction.sanitize_fields`) lives in one place, shared with
# `src.observability.tracing`, so the "never log/trace secrets or full
# document content" rule can't drift between the two call sites.


class _JsonFormatter(logging.Formatter):
    """Render each `LogRecord` as one JSON object per line."""

    def __init__(self, *, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "service": self._service_name,
            "logger": record.name,
            "message": record.getMessage(),
            "event_type": getattr(record, "event_type", None),
            "request_id": getattr(record, "request_id", None),
            "conversation_id": getattr(record, "conversation_id", None),
            "user_id": getattr(record, "user_id", None),
            "agent_name": getattr(record, "agent_name", None),
        }

        fields = getattr(record, "fields", None)
        if fields:
            payload["fields"] = _sanitize_extra(fields)

        if record.exc_info:
            exc_type, exc_value, _traceback = record.exc_info
            payload["error"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(payload, default=str)


# --- Setup -------------------------------------------------------------------


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
    handler.setFormatter(_JsonFormatter(service_name=settings.app_name))
    handler.addFilter(_ContextFilter())

    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers = [handler]
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use.

    Usage: `logger = get_logger(__name__)` at module scope.
    """
    configure_logging()
    return logging.getLogger(name)


# --- Safe structured logging helpers ---------------------------------------


def _emit(
    logger: logging.Logger,
    level: int,
    event_type: str,
    message: str,
    error: BaseException | None,
    fields: Mapping[str, object],
) -> None:
    """Build the structured `extra` payload and hand it to stdlib logging.

    Shared by `log_event` and the `log_debug`/`log_info`/`log_warning`/
    `log_error` convenience wrappers below, each of which captures its own
    `**fields` and passes it through as a plain mapping rather than
    re-unpacking it — re-unpacking an already-collected `dict[str, object]`
    into another function's specifically typed keyword parameter (`error`)
    is exactly the kind of possible name collision mypy strict mode (and,
    for the same reason, a careless caller) should be protected against.
    """
    extra: dict[str, object] = {
        "event_type": event_type,
        "fields": _sanitize_extra(fields),
        "request_id": get_request_id(),
        "conversation_id": get_conversation_id(),
        "user_id": get_user_id(),
        "agent_name": get_agent_name(),
    }
    logger.log(level, message, extra=extra, exc_info=error)


def log_event(
    logger: logging.Logger,
    level: int,
    event_type: str,
    message: str,
    *,
    error: BaseException | None = None,
    **fields: object,
) -> None:
    """Emit one structured log record with context and safe field handling.

    `event_type` names what happened (e.g. `"retrieval.completed"`,
    `"tool_call.failed"`) so records can be queried/aggregated by kind.
    `fields` is arbitrary structured context (e.g. `document_id=...`,
    `latency_ms=...`); it is sanitized before being attached to the record
    — see `_sanitize_extra` — but that is a safety net, not a license to
    pass secrets or full document content here. Pass `error` (an exception
    instance) to attach structured error information (type, message,
    traceback).
    """
    _emit(logger, level, event_type, message, error, fields)


def log_debug(logger: logging.Logger, event_type: str, message: str, **fields: object) -> None:
    """Log a `DEBUG`-level structured event — developer detail only."""
    _emit(logger, logging.DEBUG, event_type, message, None, fields)


def log_info(logger: logging.Logger, event_type: str, message: str, **fields: object) -> None:
    """Log an `INFO`-level structured event — a significant lifecycle event."""
    _emit(logger, logging.INFO, event_type, message, None, fields)


def log_warning(logger: logging.Logger, event_type: str, message: str, **fields: object) -> None:
    """Log a `WARNING`-level structured event — a recoverable problem."""
    _emit(logger, logging.WARNING, event_type, message, None, fields)


def log_error(
    logger: logging.Logger,
    event_type: str,
    message: str,
    *,
    error: BaseException | None = None,
    **fields: object,
) -> None:
    """Log an `ERROR`-level structured event — a failure needing attention."""
    _emit(logger, logging.ERROR, event_type, message, error, fields)
