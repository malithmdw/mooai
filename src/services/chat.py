"""Chat service — orchestration layer between the API and the agent graph.

``ChatService`` builds and owns a compiled LangGraph ``StateGraph`` that
runs the supervisor → retrieval → response pipeline.  It is the single
entry point the ``api`` layer is allowed to call for chat requests.

Graph topology
--------------

    START → supervisor ──┬──► retrieval → response → END
                         ├──► response → END   (unsupported_request intent)
                         └──► END              (injection-blocked: response already set)

Dependency boundaries (see CLAUDE.md)
--------------------------------------
``services`` may import from ``agents``, ``retrieval``, ``memory``,
``security``, ``observability``, and ``core``.  The ``api`` layer imports
from ``services`` only — never directly from ``agents`` or ``retrieval``.
"""

from __future__ import annotations

import uuid
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph

from src.agents import (
    GraphState,
    initial_state,
    make_response_node,
    make_retrieval_node,
    make_supervisor_node,
)
from src.agents.supervisor.models import IntentType
from src.core.config import Settings, get_settings
from src.core.logging import get_logger, log_debug, log_error, log_info
from src.models.chat import ChatRequest, ChatResponse, Message
from src.models.enums import MessageRole
from src.models.user import User
from src.retrieval.bm25.corpus import BM25Corpus
from src.retrieval.bm25.service import BM25Service
from src.retrieval.embedding import make_openai_provider
from src.retrieval.hybrid.retriever import HybridRetriever
from src.retrieval.indexing import make_pinecone_service

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Graph routing
# ---------------------------------------------------------------------------


def _route_after_supervisor(state: GraphState) -> str:
    """Determine the next node after the supervisor has run.

    Three outcomes:
    - ``END``: the supervisor already set a ``response`` (e.g. injection
      blocked) — skip all downstream nodes.
    - ``"response"``: intent is ``unsupported_request`` — the response node
      will return the canned refusal without calling the LLM.
    - ``"retrieval"``: all other intents — retrieve evidence first, then
      synthesise a response.
    """
    if state["response"] is not None:
        return END
    if state["intent"] == IntentType.UNSUPPORTED_REQUEST:
        return "response"
    return "retrieval"


# ---------------------------------------------------------------------------
# ChatService
# ---------------------------------------------------------------------------


class ChatService:
    """Compiles and owns the LangGraph agent pipeline.

    Build once per process (see ``from_settings`` or inject dependencies
    directly for testing).  ``chat()`` is safe to call concurrently.
    """

    def __init__(
        self,
        *,
        client: anthropic.AsyncAnthropic,
        retriever: HybridRetriever,
        model: str,
    ) -> None:
        self._model = model
        self._graph: Any = self._build_graph(
            client=client, retriever=retriever, model=model
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_graph(
        client: anthropic.AsyncAnthropic,
        retriever: HybridRetriever,
        model: str,
    ) -> Any:
        """Build and compile the StateGraph.  Called once at construction."""
        graph: StateGraph = StateGraph(GraphState)

        graph.add_node("supervisor", make_supervisor_node(client=client, model=model))
        graph.add_node("retrieval", make_retrieval_node(retriever=retriever))
        graph.add_node("response", make_response_node(client=client, model=model))

        graph.set_entry_point("supervisor")

        graph.add_conditional_edges(
            "supervisor",
            _route_after_supervisor,
            {"retrieval": "retrieval", "response": "response", END: END},
        )
        graph.add_edge("retrieval", "response")
        graph.add_edge("response", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(self, request: ChatRequest, user: User) -> ChatResponse:
        """Run the agent pipeline for one chat turn.

        Parameters
        ----------
        request:
            The inbound ``ChatRequest``.  A new ``conversation_id`` is
            generated when ``request.conversation_id`` is ``None``.
        user:
            The authenticated caller with their RBAC roles.  RBAC filtering
            in the retrieval node is derived from this.

        Returns
        -------
        ChatResponse
            The assembled response with citations and evidence.  When no
            answer can be produced (injection blocked, unsupported intent,
            no evidence, guardrail failure), a canned safe-refusal string
            is returned as ``message``.
        """
        conversation_id = request.conversation_id or str(uuid.uuid4())

        state = initial_state(user=user, conversation_id=conversation_id)
        state["messages"] = [
            Message(
                message_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=request.message,
            )
        ]

        log_debug(
            _logger,
            "chat_service.invoke_start",
            "Invoking agent graph",
            conversation_id=conversation_id,
            user_id=user.user_id,
            message_length=len(request.message),
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

        evidence = tuple(result["evidence"])
        citations = tuple(result["citations"])

        log_info(
            _logger,
            "chat_service.invoke_complete",
            "Agent graph completed",
            conversation_id=conversation_id,
            evidence_count=len(evidence),
            citation_count=len(citations),
            error_count=len(result["errors"]),
        )

        return ChatResponse(
            response_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            message=response_text,
            evidence=evidence,
            citations=citations,
        )

    # ------------------------------------------------------------------
    # Production factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ChatService":
        """Build a ``ChatService`` wired to real external services.

        Reads ``Settings`` (defaulting to ``get_settings()``) to construct
        the Anthropic client, OpenAI embedder, and Pinecone index service.
        BM25 is initialised with an empty corpus — keyword search will
        return no results until documents are indexed, but dense retrieval
        via Pinecone will still work.

        Raises ``ValueError`` in non-local environments when required
        credentials are absent (enforced by ``Settings`` itself).
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
        return cls(client=client, retriever=retriever, model=cfg.model_name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fallback_response() -> str:
    """Canned message used when the graph produced no response at all."""
    return (
        "I was unable to produce a response. "
        "Please try rephrasing your question, or contact your administrator."
    )
