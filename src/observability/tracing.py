"""LangSmith tracing.

Configurable from `src.core.config.Settings` (`langsmith_tracing`,
`langsmith_api_key`, `langsmith_project`) — no other module reads that
configuration or calls the `langsmith` SDK directly; see the package
docstring.

Every `trace_*` helper degrades gracefully: when tracing is disabled (no
`LANGSMITH_API_KEY`, or `LANGSMITH_TRACING=false`) they no-op with zero
LangSmith interaction, and if the LangSmith service itself is unreachable
or errors, that failure is logged and swallowed — never raised into the
business logic being traced. Business code must work identically whether
or not LangSmith is configured; the only difference is whether a trace
gets recorded.

What is (and is not) sent to LangSmith: only identifiers, names, counts,
and scores — see `docs/observability.md`. In particular, none of these
helpers accept raw user messages, retrieved document content, or tool
output as a parameter; only bounded metadata (e.g. `query_length`, not
`query`). Any caller-supplied `**metadata` is additionally run through the
same redaction used for logs (`src.core.redaction.sanitize_fields`) as a
defense-in-depth safety net.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from src.core.config import get_settings
from src.core.logging import get_logger, log_info, log_warning
from src.core.redaction import sanitize_fields

if TYPE_CHECKING:
    from langsmith import Client

logger = get_logger(__name__)

_CONFIGURED = False

RunType = Literal["chain", "tool", "retriever"]


def is_tracing_enabled() -> bool:
    """Whether LangSmith tracing is actually active for this process.

    True only when both `LANGSMITH_TRACING=true` (the default) and a
    `LANGSMITH_API_KEY` are configured. Every `trace_*` helper checks this
    and no-ops when it is False — see module docstring.
    """
    settings = get_settings()
    return settings.langsmith_tracing and bool(settings.langsmith_api_key.get_secret_value())


def configure_tracing() -> None:
    """Log whether LangSmith tracing is active for this process.

    Idempotent — safe to call multiple times (e.g. from tests and from the
    app entrypoint), mirroring `src.core.logging.configure_logging`.
    Nothing here mutates global/environment state: each `trace_*` call
    independently checks `is_tracing_enabled()` and builds its own
    LangSmith client on demand, so `Settings` stays the single source of
    truth for configuration (see CLAUDE.md).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if is_tracing_enabled():
        log_info(
            logger,
            "observability.tracing_configured",
            "LangSmith tracing enabled",
            project=get_settings().langsmith_project,
        )
    else:
        log_info(logger, "observability.tracing_configured", "LangSmith tracing disabled")

    _CONFIGURED = True


@lru_cache
def _get_client() -> Client:
    """Construct (once) the LangSmith client used by every `trace_*` call.

    Only ever called when `is_tracing_enabled()` is True, and imports
    `langsmith` lazily — importing this module must not import the
    LangSmith SDK or construct a client when tracing is disabled/absent.
    """
    from langsmith import Client

    settings = get_settings()
    return Client(api_key=settings.langsmith_api_key.get_secret_value())


def _start_run(name: str, run_type: RunType, metadata: Mapping[str, object]) -> str | None:
    """Start one LangSmith run, or return `None` if that fails.

    A failure here (network error, bad credentials, SDK bug) is logged and
    swallowed, never raised — see module docstring "graceful behavior".
    """
    run_id = str(uuid.uuid4())
    try:
        _get_client().create_run(
            id=run_id,
            name=name,
            run_type=run_type,
            inputs={},
            project_name=get_settings().langsmith_project,
            extra={"metadata": dict(metadata)},
        )
    except Exception as exc:  # tracing must never break business logic
        log_warning(
            logger,
            "observability.trace_start_failed",
            "failed to start LangSmith run",
            run_name=name,
            error_type=type(exc).__name__,
        )
        return None
    return run_id


def _end_run(run_id: str | None, *, error: str | None) -> None:
    """End a run started by `_start_run`; a no-op if starting it failed."""
    if run_id is None:
        return
    try:
        _get_client().update_run(run_id, outputs={}, error=error)
    except Exception as exc:  # tracing must never break business logic
        log_warning(
            logger,
            "observability.trace_end_failed",
            "failed to end LangSmith run",
            run_id=run_id,
            error_type=type(exc).__name__,
        )


@contextmanager
def _traced_run(name: str, run_type: RunType, **metadata: object) -> Iterator[None]:
    """Wrap the `with` block in one LangSmith run, or no-op if disabled.

    The underlying primitive for every public `trace_*` helper below. Only
    exceptions raised by the wrapped block itself propagate; any error
    talking to LangSmith is handled inside `_start_run`/`_end_run`.
    """
    if not is_tracing_enabled():
        yield
        return

    run_id = _start_run(name, run_type, sanitize_fields(metadata))
    try:
        yield
    except Exception as exc:
        _end_run(run_id, error=str(exc))
        raise
    else:
        _end_run(run_id, error=None)


# --- Tracing helpers ---------------------------------------------------------


@contextmanager
def trace_conversation(conversation_id: str, *, user_id: str | None = None) -> Iterator[None]:
    """Trace one end-to-end conversation turn as a top-level LangSmith run."""
    with _traced_run("conversation", "chain", conversation_id=conversation_id, user_id=user_id):
        yield


@contextmanager
def trace_agent_node(node_name: str, *, conversation_id: str | None = None) -> Iterator[None]:
    """Trace one LangGraph agent node's execution (e.g. supervisor, memory)."""
    with _traced_run(
        f"agent_node:{node_name}", "chain", node_name=node_name, conversation_id=conversation_id
    ):
        yield


@contextmanager
def trace_tool_call(tool_name: str, *, conversation_id: str | None = None) -> Iterator[None]:
    """Trace one external tool invocation (via MCP)."""
    with _traced_run(
        f"tool:{tool_name}", "tool", tool_name=tool_name, conversation_id=conversation_id
    ):
        yield


@contextmanager
def trace_retrieval(
    *, query_length: int | None = None, conversation_id: str | None = None
) -> Iterator[None]:
    """Trace one hybrid-search retrieval operation.

    Takes `query_length`, not the query text itself: the query is user
    input and is not sent to LangSmith by this helper — see
    `docs/observability.md` "what is intentionally not logged".
    """
    with _traced_run(
        "retrieval",
        "retriever",
        query_length=query_length,
        conversation_id=conversation_id,
    ):
        yield


@contextmanager
def trace_validation(validation_type: str, *, conversation_id: str | None = None) -> Iterator[None]:
    """Trace one output-validation check.

    See CLAUDE.md "LLM output must be validated before use" — this is how
    that validation step itself becomes visible in a trace.
    """
    with _traced_run(
        f"validation:{validation_type}",
        "chain",
        validation_type=validation_type,
        conversation_id=conversation_id,
    ):
        yield
