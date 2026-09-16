"""Search query generation for the Retrieval Agent.

Queries are derived from the current GraphState: the user's last message is
always the primary query; for complex intents (analytical_research,
mixed_request) additional sub-queries are derived from the supervisor's task
plan, up to ``_MAX_QUERIES`` total.

Security note
-------------
Query strings are UNTRUSTED USER INPUT.  They are search strings only and
must never be passed to any execution context or interpreted as instructions.
"""

from __future__ import annotations

from src.agents.state import GraphState
from src.models.enums import MessageRole

_MAX_QUERIES: int = 3

_MULTI_QUERY_INTENTS: frozenset[str] = frozenset(
    {"analytical_research", "mixed_request"}
)


def generate_queries(state: GraphState) -> list[str]:
    """Derive retrieval queries from *state*.

    Always returns the user's last message as the primary query.  For
    ``analytical_research`` and ``mixed_request`` intents, task-plan steps
    are added as additional sub-queries (up to ``_MAX_QUERIES`` total,
    de-duplicated case-insensitively).

    Returns an empty list when the state contains no user messages — the
    node handles this as an error condition rather than raising.

    Query strings are untrusted — never interpret them as instructions.
    """
    user_msgs = [m for m in state["messages"] if m.role == MessageRole.USER]
    if not user_msgs:
        return []

    primary = user_msgs[-1].content.strip()
    if not primary:
        return []

    queries: list[str] = [primary]
    seen_lower: set[str] = {primary.lower()}

    intent = state["intent"] or ""
    if intent in _MULTI_QUERY_INTENTS:
        for step in state["task_plan"]:
            step = step.strip()
            if step and step.lower() not in seen_lower:
                queries.append(step)
                seen_lower.add(step.lower())
            if len(queries) >= _MAX_QUERIES:
                break

    return queries
