"""Typed LangGraph state for the enterprise AI agent graph.

``GraphState`` is a ``TypedDict`` used as the schema for the LangGraph
``StateGraph``.  Fields annotated with ``operator.add`` use accumulator
(append) semantics: each node that returns a partial update for such a field
has its list *appended* to the existing list rather than replacing it.  All
other fields use last-writer-wins (replace) semantics.

Accumulate fields (Annotated with operator.add):
    messages, retrieval_queries, retrieved_documents, evidence,
    research_results, tool_calls, tool_results, memory_updates,
    validation_results, citations, errors

Replace fields (plain type hints):
    user, conversation_id, user_role, intent, task_plan, research_tasks,
    current_agent, current_node, response, budget

Usage
-----
Build the initial state with ``initial_state()`` at graph entry:

    from src.agents.state import GraphState, initial_state

    config = {"configurable": {"thread_id": conversation_id}}
    result = await graph.ainvoke(
        initial_state(user=user, conversation_id=conversation_id),
        config=config,
    )

Nodes return partial dicts — only the fields they modify:

    async def retrieval_node(state: GraphState) -> dict[str, object]:
        ...
        return {
            "retrieval_queries": [query],
            "retrieved_documents": results,
            "budget": state["budget"].consume(retrieval_results=len(results)),
        }
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from src.agents.budget import ExecutionBudget
from src.models.chat import Message
from src.models.enums import AgentState, Role
from src.models.evidence import Citation, Evidence
from src.models.research import ResearchResult, ResearchTask
from src.models.tools import ToolCall, ToolResult
from src.models.user import User
from src.models.validation import ValidationResult
from src.retrieval.hybrid.models import RetrievalEvidence


class GraphState(TypedDict):
    """Full typed state for one agent graph execution.

    Construct with ``initial_state()``; nodes return partial dicts.
    """

    # ------------------------------------------------------------------
    # Identity & access control
    # ------------------------------------------------------------------

    user: User | None
    """Authenticated user for this execution; sets the RBAC boundary."""

    conversation_id: str | None
    """Conversation thread this execution belongs to."""

    user_role: list[Role]
    """Active RBAC roles for this execution, derived from ``user.roles``."""

    # ------------------------------------------------------------------
    # Conversation history (append)
    # ------------------------------------------------------------------

    messages: Annotated[list[Message], operator.add]
    """All conversation messages; each node append its new turns."""

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    intent: str | None
    """Parsed intent extracted from the latest user message."""

    task_plan: list[str]
    """Ordered high-level steps the supervisor planned to fulfil the intent.
    Replaced atomically when the supervisor revises the plan."""

    # ------------------------------------------------------------------
    # Retrieval (append)
    # ------------------------------------------------------------------

    retrieval_queries: Annotated[list[str], operator.add]
    """All queries issued to the hybrid retriever across this execution."""

    retrieved_documents: Annotated[list[RetrievalEvidence], operator.add]
    """All chunks returned by the hybrid retriever, with full provenance."""

    # ------------------------------------------------------------------
    # Evidence (append)
    # ------------------------------------------------------------------

    evidence: Annotated[list[Evidence], operator.add]
    """Curated evidence items selected from ``retrieved_documents`` to
    back the final response."""

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    research_tasks: list[ResearchTask]
    """Active research task list; replaced by the research node when task
    statuses change (PENDING → IN_PROGRESS → COMPLETED / FAILED)."""

    research_results: Annotated[list[ResearchResult], operator.add]
    """Completed research results; accumulated as tasks finish."""

    # ------------------------------------------------------------------
    # Tool calls (append)
    # ------------------------------------------------------------------

    tool_calls: Annotated[list[ToolCall], operator.add]
    """All tool invocation requests issued during this execution."""

    tool_results: Annotated[list[ToolResult], operator.add]
    """All tool outcomes returned during this execution."""

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    memory_updates: Annotated[list[str], operator.add]
    """Descriptions of memory operations performed (for audit / tracing)."""

    # ------------------------------------------------------------------
    # Validation (append)
    # ------------------------------------------------------------------

    validation_results: Annotated[list[ValidationResult], operator.add]
    """Validation outcomes produced by the validation node."""

    # ------------------------------------------------------------------
    # Execution tracking
    # ------------------------------------------------------------------

    current_agent: AgentState | None
    """The agent stage currently active (supervisor, retrieval, etc.)."""

    current_node: str | None
    """The specific graph node name currently executing."""

    # ------------------------------------------------------------------
    # Output (replace)
    # ------------------------------------------------------------------

    response: str | None
    """The assembled natural-language response text."""

    citations: Annotated[list[Citation], operator.add]
    """In-text citations linking response claims to ``evidence`` entries."""

    # ------------------------------------------------------------------
    # Error tracking (append)
    # ------------------------------------------------------------------

    errors: Annotated[list[str], operator.add]
    """Non-fatal error messages accumulated during execution.  A non-empty
    list does not halt execution — the supervisor decides how to proceed."""

    # ------------------------------------------------------------------
    # Budget (replace)
    # ------------------------------------------------------------------

    budget: ExecutionBudget
    """Resource budget; replaced with an updated copy each time a node
    consumes tokens, retrieval results, or tool calls."""


def initial_state(
    *,
    user: User | None = None,
    conversation_id: str | None = None,
    budget: ExecutionBudget | None = None,
) -> GraphState:
    """Create a clean ``GraphState`` for the start of a graph execution.

    Parameters
    ----------
    user:
        The authenticated user.  When provided, ``user_role`` is
        populated from ``user.roles``.
    conversation_id:
        The conversation thread this execution belongs to.
    budget:
        Execution budget.  Defaults to ``ExecutionBudget()`` (library
        defaults: 10 000 tokens, 10 tool calls).
    """
    return GraphState(
        user=user,
        conversation_id=conversation_id,
        user_role=list(user.roles) if user is not None else [],
        messages=[],
        intent=None,
        task_plan=[],
        retrieval_queries=[],
        retrieved_documents=[],
        evidence=[],
        research_tasks=[],
        research_results=[],
        tool_calls=[],
        tool_results=[],
        memory_updates=[],
        validation_results=[],
        current_agent=None,
        current_node=None,
        response=None,
        citations=[],
        errors=[],
        budget=budget if budget is not None else ExecutionBudget(),
    )
