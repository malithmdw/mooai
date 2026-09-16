"""Tests for ``ChatService``.

Coverage
--------
_route_after_supervisor:
  - response already set → END (injection-blocked path)
  - intent unsupported_request → "response"
  - intent knowledge_question → "retrieval"
  - intent None (error path) → "retrieval"

ChatService.chat():
  - full pipeline: graph invoked, ChatResponse assembled from result
  - conversation_id generated when request has none
  - supplied conversation_id is preserved
  - evidence and citations from graph state forwarded to ChatResponse
  - graph error propagates (not swallowed silently)
  - empty response from graph falls back to canned message
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.chat import ChatRequest, ChatResponse
from src.models.enums import Role
from src.models.evidence import Citation, Evidence
from src.models.user import User
from src.agents.state import GraphState, initial_state
from src.services.chat import ChatService, _route_after_supervisor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(roles: tuple[Role, ...] = (Role.ANALYST,)) -> User:
    return User(
        user_id="user-0001",
        username="analyst01",
        display_name="Alice",
        roles=roles,
    )


def _request(
    message: str = "What is the loan default rate?",
    conversation_id: str | None = None,
) -> ChatRequest:
    return ChatRequest(
        request_id=str(uuid.uuid4()),
        user_id="user-0001",
        message=message,
        conversation_id=conversation_id,
    )


def _state(
    response: str | None = None,
    intent: str | None = "knowledge_question",
) -> GraphState:
    state = initial_state(user=_user(), conversation_id="conv-test")
    state["response"] = response
    state["intent"] = intent
    return state


def _mock_service(
    graph_result: dict | None = None,
) -> ChatService:
    """Return a ChatService whose internal graph is replaced with a mock."""
    service = object.__new__(ChatService)
    result = graph_result or {
        "response": "The default rate was 2.3%.",
        "evidence": [],
        "citations": [],
        "errors": [],
    }
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=result)
    service._graph = mock_graph  # type: ignore[attr-defined]
    return service


# ---------------------------------------------------------------------------
# _route_after_supervisor unit tests
# ---------------------------------------------------------------------------


class TestRouteAfterSupervisor:
    def test_response_set_returns_end(self) -> None:
        from langgraph.graph import END

        state = _state(response="Already blocked.")
        assert _route_after_supervisor(state) == END

    def test_unsupported_request_routes_to_response(self) -> None:
        state = _state(intent="unsupported_request")
        assert _route_after_supervisor(state) == "response"

    def test_knowledge_question_routes_to_retrieval(self) -> None:
        state = _state(intent="knowledge_question")
        assert _route_after_supervisor(state) == "retrieval"

    def test_analytical_research_routes_to_retrieval(self) -> None:
        state = _state(intent="analytical_research")
        assert _route_after_supervisor(state) == "retrieval"

    def test_mixed_request_routes_to_retrieval(self) -> None:
        state = _state(intent="mixed_request")
        assert _route_after_supervisor(state) == "retrieval"

    def test_none_intent_routes_to_retrieval(self) -> None:
        state = _state(intent=None)
        assert _route_after_supervisor(state) == "retrieval"

    def test_response_set_takes_priority_over_unsupported_intent(self) -> None:
        """Even with unsupported_request intent, a set response goes to END."""
        from langgraph.graph import END

        state = _state(response="Blocked.", intent="unsupported_request")
        assert _route_after_supervisor(state) == END


# ---------------------------------------------------------------------------
# ChatService.chat() tests
# ---------------------------------------------------------------------------


class TestChatServiceChat:
    async def test_returns_chat_response(self) -> None:
        service = _mock_service()
        response = await service.chat(_request(), _user())
        assert isinstance(response, ChatResponse)
        assert response.message == "The default rate was 2.3%."

    async def test_graph_invoked_once(self) -> None:
        service = _mock_service()
        await service.chat(_request(), _user())
        service._graph.ainvoke.assert_called_once()  # type: ignore[attr-defined]

    async def test_generates_conversation_id_when_none(self) -> None:
        service = _mock_service()
        response = await service.chat(_request(conversation_id=None), _user())
        assert response.conversation_id  # non-empty UUID generated

    async def test_preserves_supplied_conversation_id(self) -> None:
        supplied = "conv-supplied-001"
        service = _mock_service()
        response = await service.chat(_request(conversation_id=supplied), _user())
        assert response.conversation_id == supplied

    async def test_evidence_forwarded_to_response(self) -> None:
        evidence = Evidence(
            evidence_id="DOC-001-chunk-0000",
            document_id="DOC-001",
            excerpt="The default rate was 2.3%.",
            relevance_score=0.9,
        )
        citation = Citation(evidence_id="DOC-001-chunk-0000", reference_number=1)
        service = _mock_service(
            graph_result={
                "response": "The default rate was 2.3%. [1]",
                "evidence": [evidence],
                "citations": [citation],
                "errors": [],
            }
        )
        response = await service.chat(_request(), _user())
        assert len(response.evidence) == 1
        assert len(response.citations) == 1
        assert response.evidence[0].evidence_id == "DOC-001-chunk-0000"
        assert response.citations[0].reference_number == 1

    async def test_empty_response_falls_back_to_canned_message(self) -> None:
        service = _mock_service(
            graph_result={
                "response": None,
                "evidence": [],
                "citations": [],
                "errors": [],
            }
        )
        response = await service.chat(_request(), _user())
        assert response.message  # not empty — canned fallback was used
        assert "unable" in response.message.lower() or "rephrase" in response.message.lower()

    async def test_graph_error_propagates(self) -> None:
        service = object.__new__(ChatService)
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph exploded"))
        service._graph = mock_graph  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="graph exploded"):
            await service.chat(_request(), _user())

    async def test_message_passed_to_graph_state(self) -> None:
        """The user's message appears in the state passed to ainvoke."""
        captured_states: list[GraphState] = []

        service = object.__new__(ChatService)
        mock_graph = MagicMock()

        async def capture_invoke(state: GraphState) -> dict:  # type: ignore[type-arg]
            captured_states.append(state)
            return {"response": "ok", "evidence": [], "citations": [], "errors": []}

        mock_graph.ainvoke = capture_invoke
        service._graph = mock_graph  # type: ignore[attr-defined]

        await service.chat(_request(message="What is the default rate?"), _user())

        assert len(captured_states) == 1
        messages = captured_states[0]["messages"]
        assert any("default rate" in m.content for m in messages)

    async def test_user_roles_passed_to_graph_state(self) -> None:
        """The user's roles appear in the state forwarded to the graph."""
        captured_states: list[GraphState] = []

        service = object.__new__(ChatService)
        mock_graph = MagicMock()

        async def capture_invoke(state: GraphState) -> dict:  # type: ignore[type-arg]
            captured_states.append(state)
            return {"response": "ok", "evidence": [], "citations": [], "errors": []}

        mock_graph.ainvoke = capture_invoke
        service._graph = mock_graph  # type: ignore[attr-defined]

        await service.chat(_request(), _user(roles=(Role.ANALYST, Role.ENGINEER)))

        assert len(captured_states) == 1
        user_role = captured_states[0]["user_role"]
        assert Role.ANALYST in user_role
        assert Role.ENGINEER in user_role
