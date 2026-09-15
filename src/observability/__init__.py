"""Observability: LangSmith tracing and structured run metadata.

Every agent/tool invocation is expected to be traceable end-to-end in
LangSmith. This package is the only place that configures tracing or calls
the LangSmith SDK — other packages use the `trace_*` helpers re-exported
here instead of importing `langsmith` themselves, so business logic never
becomes tightly coupled to it (see `src.observability.tracing` module
docstring and `docs/observability.md`).
"""

from src.observability.tracing import (
    configure_tracing,
    is_tracing_enabled,
    trace_agent_node,
    trace_conversation,
    trace_retrieval,
    trace_tool_call,
    trace_validation,
)

__all__ = [
    "configure_tracing",
    "is_tracing_enabled",
    "trace_agent_node",
    "trace_conversation",
    "trace_retrieval",
    "trace_tool_call",
    "trace_validation",
]
