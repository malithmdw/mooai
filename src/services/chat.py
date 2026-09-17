"""Chat service — orchestration layer between the API and the agent graph.

``ChatService`` builds and owns a compiled LangGraph ``StateGraph`` that
runs the complete supervisor → [retrieval | research | tool_execution] →
response pipeline.  It is the single entry point the ``api`` layer calls.

Graph topology
--------------

    START → supervisor ──┬──► retrieval    → response → END
                         ├──► research     → response → END
                         ├──► tool_execution → response → END
                         ├──► response → END   (unsupported_request)
                         └──► END              (injection-blocked)

Routing is determined by ``_route_after_supervisor`` based on the intent
the supervisor wrote into ``GraphState.intent``:

    KNOWLEDGE_QUESTION  → retrieval
    ANALYTICAL_RESEARCH → research
    TOOL_REQUEST        → tool_execution
    MIXED_REQUEST       → retrieval (supervisor picks the primary branch)
    UNSUPPORTED_REQUEST → response (canned refusal)

Conversation memory
-------------------
A simple in-process dict (``_conversation_history``) stores the last
``_MAX_HISTORY_TURNS`` turns per conversation_id.  This is the POC
substitute for the PostgreSQL-backed ``MemoryService`` — see CLAUDE.md
"What would change for production".

Dependency boundaries (see CLAUDE.md)
--------------------------------------
``services`` may import from ``agents``, ``retrieval``, ``memory``,
``security``, ``observability``, and ``core``.  The ``api`` layer imports
from ``services`` only — never directly from ``agents`` or ``retrieval``.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph

from src.agents import (
    GraphState,
    initial_state,
    make_research_node,
    make_response_node,
    make_retrieval_node,
    make_supervisor_node,
)
from src.agents.supervisor.models import IntentType
from src.core.config import Settings, get_settings
from src.core.logging import get_logger, log_debug, log_error, log_info, log_warning
from src.models.chat import ChatRequest, ChatResponse, Message
from src.models.enums import AgentState, MessageRole, Role
from src.models.tools import ToolCall
from src.models.user import User
from src.retrieval.bm25.corpus import BM25Corpus
from src.retrieval.bm25.service import BM25Service
from src.retrieval.embedding import make_openai_provider
from src.retrieval.hybrid.retriever import HybridRetriever
from src.retrieval.indexing import make_pinecone_service
from src.services.tool_execution import ToolExecutionService

_logger = get_logger(__name__)

_MAX_HISTORY_TURNS: int = 20

# ---------------------------------------------------------------------------
# In-process conversation memory
# ---------------------------------------------------------------------------

_conversation_history: dict[str, list[Message]] = {}


def _get_history(conversation_id: str) -> list[Message]:
    """Return stored conversation history for *conversation_id*, or []."""
    return list(_conversation_history.get(conversation_id, []))


def _update_history(conversation_id: str, messages: list[Message]) -> None:
    """Persist the last ``_MAX_HISTORY_TURNS`` messages for a conversation."""
    _conversation_history[conversation_id] = messages[-_MAX_HISTORY_TURNS:]


# ---------------------------------------------------------------------------
# Tool call inference (heuristic, POC-level)
# ---------------------------------------------------------------------------

_INCIDENT_RE = re.compile(r"\b(INC-\d+)\b", re.IGNORECASE)
_SERVICE_RE = re.compile(r"\b(SVC-\d+)\b", re.IGNORECASE)
_EMPLOYEE_RE = re.compile(r"\b(EMP-\d+)\b", re.IGNORECASE)

_EMPLOYEE_KEYWORDS = (
    "find employee",
    "look up employee",
    "search for employee",
    "who is",
    "find the employee",
    "employee named",
    "lookup employee",
)


def _infer_tool_call(message_text: str, task_plan: list[str]) -> ToolCall | None:
    """Heuristically infer a ToolCall from the user message and supervisor task plan.

    Priority order: incident ID → service ID → employee ID → employee name
    search → default knowledge_search.

    Returns ``None`` only if the message is empty.
    """
    if not message_text.strip():
        return None

    combined = " ".join([message_text] + task_plan)

    if m := _INCIDENT_RE.search(combined):
        return ToolCall(
            tool_call_id=str(uuid.uuid4()),
            tool_name="get_incident",
            arguments={"incident_id": m.group(1).upper()},
        )

    if m := _SERVICE_RE.search(combined):
        return ToolCall(
            tool_call_id=str(uuid.uuid4()),
            tool_name="get_service",
            arguments={"service_id": m.group(1).upper()},
        )

    if m := _EMPLOYEE_RE.search(combined):
        return ToolCall(
            tool_call_id=str(uuid.uuid4()),
            tool_name="find_employee",
            arguments={"query": m.group(1).upper()},
        )

    combined_lower = combined.lower()
    for keyword in _EMPLOYEE_KEYWORDS:
        if keyword in combined_lower:
            idx = combined_lower.find(keyword)
            after = message_text[idx + len(keyword) :].strip()
            name = after.split("\n")[0][:100].strip(" ?.!")
            if name:
                return ToolCall(
                    tool_call_id=str(uuid.uuid4()),
                    tool_name="find_employee",
                    arguments={"query": name},
                )

    return ToolCall(
        tool_call_id=str(uuid.uuid4()),
        tool_name="knowledge_search",
        arguments={"query": message_text[:500]},
    )


# ---------------------------------------------------------------------------
# Tool execution node
# ---------------------------------------------------------------------------


def _make_tool_execution_node(
    tool_service: ToolExecutionService,
) -> Any:
    """Return a LangGraph-compatible async callable for the tool execution node.

    The node infers which tool to call from the user's message and the
    supervisor's task plan, then passes the ToolCall through
    ``ToolExecutionService.execute()`` — the single enforced authorization
    and execution gateway (see CLAUDE.md).
    """

    async def tool_execution_node(state: GraphState) -> dict[str, Any]:
        budget = state["budget"]
        role = state["user_role"][0] if state["user_role"] else Role.VIEWER
        conversation_id = state["conversation_id"]

        if budget.is_exhausted:
            log_warning(
                _logger,
                "tool_execution_node.budget_exhausted",
                "Tool execution skipped — budget exhausted",
                conversation_id=conversation_id,
            )
            return {
                "current_agent": AgentState.TOOL_EXECUTION,
                "current_node": "tool_execution",
                "errors": ["tool_execution skipped: budget exhausted"],
                "budget": budget,
            }

        last_user_message = next(
            (
                m.content
                for m in reversed(list(state["messages"]))
                if m.role == MessageRole.USER
            ),
            "",
        )

        tool_call = _infer_tool_call(last_user_message, list(state["task_plan"]))
        if tool_call is None:
            return {
                "current_agent": AgentState.TOOL_EXECUTION,
                "current_node": "tool_execution",
                "errors": ["tool_execution: could not infer tool call from message"],
                "budget": budget,
            }

        log_info(
            _logger,
            "tool_execution_node.executing",
            "Executing tool call through ToolExecutionService",
            tool_name=tool_call.tool_name,
            role=str(role),
            conversation_id=conversation_id,
        )

        result, new_budget = await tool_service.execute(
            tool_call,
            role=role,
            budget=budget,
            conversation_id=conversation_id,
        )

        if not result.success:
            log_warning(
                _logger,
                "tool_execution_node.tool_failed",
                "Tool call failed",
                tool_name=tool_call.tool_name,
                error_code=result.error_code,
                conversation_id=conversation_id,
            )
            return {
                "current_agent": AgentState.TOOL_EXECUTION,
                "current_node": "tool_execution",
                "tool_calls": [tool_call],
                "tool_results": [result],
                "errors": [
                    f"tool {tool_call.tool_name!r} failed: "
                    f"{result.error_code} — {result.error_message}"
                ],
                "budget": new_budget,
            }

        log_info(
            _logger,
            "tool_execution_node.success",
            "Tool call succeeded",
            tool_name=tool_call.tool_name,
            conversation_id=conversation_id,
        )

        return {
            "current_agent": AgentState.TOOL_EXECUTION,
            "current_node": "tool_execution",
            "tool_calls": [tool_call],
            "tool_results": [result],
            "budget": new_budget,
        }

    return tool_execution_node


# ---------------------------------------------------------------------------
# Graph routing
# ---------------------------------------------------------------------------


def _route_after_supervisor(state: GraphState) -> str:
    """Determine the next node after the supervisor has run.

    Five outcomes:
    - ``END``: the supervisor already set a ``response`` (injection blocked).
    - ``"response"``: intent is ``unsupported_request`` — canned refusal.
    - ``"retrieval"``: KNOWLEDGE_QUESTION or MIXED_REQUEST.
    - ``"research"``: ANALYTICAL_RESEARCH — recursive RLM pipeline.
    - ``"tool_execution"``: TOOL_REQUEST — centralized tool gateway.
    """
    if state["response"] is not None:
        return END

    intent = state.get("intent")

    if intent == IntentType.UNSUPPORTED_REQUEST:
        return "response"
    if intent == IntentType.ANALYTICAL_RESEARCH:
        return "research"
    if intent == IntentType.TOOL_REQUEST:
        return "tool_execution"
    # KNOWLEDGE_QUESTION, MIXED_REQUEST, or unresolved intent all go to retrieval.
    return "retrieval"


# ---------------------------------------------------------------------------
# Agent activity summary (derived from final state, no extra fields needed)
# ---------------------------------------------------------------------------


def _build_activity_summary(state: GraphState) -> list[str]:
    """Derive a human-readable execution trace from the completed graph state."""
    lines: list[str] = []

    intent = state.get("intent")
    if intent:
        lines.append(f"Supervisor → Intent: {intent}")

    task_plan = state.get("task_plan", [])
    if task_plan:
        lines.append(f"Supervisor → Plan: {len(task_plan)} step(s)")

    queries = state.get("retrieval_queries", [])
    retrieved = state.get("retrieved_documents", [])
    if queries:
        lines.append(f"Retrieval Agent → {len(queries)} search query(s)")
    if retrieved:
        lines.append(f"Retrieval Agent → {len(retrieved)} chunks retrieved")

    research_tasks = state.get("research_tasks", [])
    research_results = state.get("research_results", [])
    if research_tasks:
        completed = sum(
            1 for t in research_tasks if getattr(t, "status", None) in ("completed", "COMPLETED")
        )
        lines.append(
            f"Research Agent → {completed}/{len(research_tasks)} task(s) completed"
        )
    if research_results:
        lines.append(f"Research Agent → {len(research_results)} result(s) aggregated")

    tool_calls = state.get("tool_calls", [])
    tool_results = state.get("tool_results", [])
    if tool_calls:
        for call, result in zip(tool_calls, tool_results):
            status = "✓" if getattr(result, "success", False) else "✗"
            lines.append(f"Tool Execution → {status} {getattr(call, 'tool_name', '?')}")

    evidence = state.get("evidence", [])
    citations = state.get("citations", [])
    if evidence:
        lines.append(f"Response Agent → {len(evidence)} evidence item(s) used")
    if citations:
        lines.append(f"Response Agent → {len(citations)} citation(s) verified")

    errors = state.get("errors", [])
    for err in errors:
        lines.append(f"⚠ {err}")

    return lines


# ---------------------------------------------------------------------------
# ChatService
# ---------------------------------------------------------------------------


class ChatService:
    """Compiles and owns the complete LangGraph agent pipeline.

    Build once per process (see ``from_settings`` or inject dependencies
    directly for testing).  ``chat()`` is safe to call concurrently.
    """

    def __init__(
        self,
        *,
        client: anthropic.AsyncAnthropic,
        retriever: HybridRetriever,
        tool_service: ToolExecutionService,
        model: str,
    ) -> None:
        self._model = model
        self._graph: Any = self._build_graph(
            client=client,
            retriever=retriever,
            tool_service=tool_service,
            model=model,
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_graph(
        client: anthropic.AsyncAnthropic,
        retriever: HybridRetriever,
        tool_service: ToolExecutionService,
        model: str,
    ) -> Any:
        """Build and compile the complete StateGraph.  Called once."""
        graph: StateGraph = StateGraph(GraphState)

        graph.add_node("supervisor", make_supervisor_node(client=client, model=model))
        graph.add_node("retrieval", make_retrieval_node(retriever=retriever))
        graph.add_node(
            "research", make_research_node(client=client, retriever=retriever, model=model)
        )
        graph.add_node("tool_execution", _make_tool_execution_node(tool_service))
        graph.add_node("response", make_response_node(client=client, model=model))

        graph.set_entry_point("supervisor")

        graph.add_conditional_edges(
            "supervisor",
            _route_after_supervisor,
            {
                "retrieval": "retrieval",
                "research": "research",
                "tool_execution": "tool_execution",
                "response": "response",
                END: END,
            },
        )
        graph.add_edge("retrieval", "response")
        graph.add_edge("research", "response")
        graph.add_edge("tool_execution", "response")
        graph.add_edge("response", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(self, request: ChatRequest, user: User) -> ChatResponse:
        """Run the agent pipeline for one chat turn.

        Retrieves stored conversation history for ``conversation_id`` (if any),
        prepends it to the graph state so every node has multi-turn context,
        then persists the new turn after the graph completes.

        Returns
        -------
        ChatResponse
            Assembled response with citations and evidence.  When no
            answer can be produced, a canned safe-refusal string is returned.
        """
        conversation_id = request.conversation_id or str(uuid.uuid4())

        history = _get_history(conversation_id)
        new_message = Message(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=request.message,
        )

        state = initial_state(user=user, conversation_id=conversation_id)
        state["messages"] = history + [new_message]

        log_debug(
            _logger,
            "chat_service.invoke_start",
            "Invoking agent graph",
            conversation_id=conversation_id,
            user_id=user.user_id,
            message_length=len(request.message),
            history_turns=len(history),
        )

        try:
            result: GraphState = await self._graph.ainvoke(state)
        except Exception as exc:
            log_error(
                _logger,
                "chat_service.graph_error",
                "Agent graph raised an unhandled exception",
                error=exc,
                conversation_id=conversation_id,
            )
            raise

        response_text: str = result["response"] or _fallback_response()

        assistant_message = Message(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=response_text,
        )
        _update_history(
            conversation_id,
            list(result.get("messages", [])) + [assistant_message],
        )

        evidence = tuple(result["evidence"])
        citations = tuple(result["citations"])
        activity = tuple(_build_activity_summary(result))

        log_info(
            _logger,
            "chat_service.invoke_complete",
            "Agent graph completed",
            conversation_id=conversation_id,
            evidence_count=len(evidence),
            citation_count=len(citations),
            error_count=len(result["errors"]),
            intent=result.get("intent"),
            activity_steps=len(activity),
        )

        return ChatResponse(
            response_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            message=response_text,
            evidence=evidence,
            citations=citations,
            agent_activity=activity,
        )

    # ------------------------------------------------------------------
    # Production factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ChatService":
        """Build a ``ChatService`` wired to real external services.

        BM25 is initialised with an empty corpus — keyword search returns no
        results until documents are indexed, but dense retrieval via Pinecone
        still works.
        """
        cfg = settings or get_settings()
        api_key = cfg.anthropic_api_key.get_secret_value()
        client = anthropic.AsyncAnthropic(
            **({"api_key": api_key} if api_key else {})
        )
        embedder = make_openai_provider()
        pinecone_service = make_pinecone_service()
        bm25_service = BM25Service(BM25Corpus([]))

        retriever = HybridRetriever(
            pinecone_service=pinecone_service,
            embedder=embedder,
            bm25_service=bm25_service,
        )
        tool_service = ToolExecutionService(retriever=retriever)

        return cls(
            client=client,
            retriever=retriever,
            tool_service=tool_service,
            model=cfg.model_name,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fallback_response() -> str:
    return (
        "I was unable to produce a response. "
        "Please try rephrasing your question, or contact your administrator."
    )
