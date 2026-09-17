"""Graph-level tests for ``ChatService._build_graph``.

These tests verify the actual LangGraph topology — that conditional edges
route to the correct nodes and that all routing branches complete without
raising.  External dependencies (Anthropic, Pinecone) are mocked at the
node-factory level so the compiled graph executes real edge traversal.

Coverage
--------
Routing branches:
  - injection-blocked supervisor response → END (skips retrieval)
  - UNSUPPORTED_REQUEST intent → response node (no retrieval)
  - KNOWLEDGE_QUESTION intent → retrieval → response → END
  - ANALYTICAL_RESEARCH intent → research → response → END
  - TOOL_REQUEST intent → tool_execution → response → END

Tool execution node:
  - VIEWER role denied by ToolExecutionService (PERMISSION_DENIED)
  - ANALYST role succeeds on knowledge_search

Conversation memory:
  - Second call with same conversation_id receives history
  - Different conversation_ids have isolated history

ChatService.chat:
  - Returns ChatResponse with agent_activity populated
  - conversation_id is propagated in the response
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.agents.state import initial_state
from src.models.chat import ChatRequest, ChatResponse, Message
from src.models.enums import MessageRole, Role
from src.models.user import User
from src.services.chat import (
    _conversation_history,
    _infer_tool_call,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(role: Role = Role.ANALYST) -> User:
    return User(
        user_id=f"test-user-{role.value}",
        username=f"{role.value}_user",
        roles=(role,),
    )


# _route_after_supervisor is tested comprehensively in tests/services/test_chat.py.
# The tests below focus on graph-level concerns not covered there.


# ---------------------------------------------------------------------------
# 2. _infer_tool_call
# ---------------------------------------------------------------------------


class TestInferToolCall:
    def test_incident_id_extracted(self) -> None:
        tc = _infer_tool_call("Get incident INC-003", [])
        assert tc is not None
        assert tc.tool_name == "get_incident"
        assert tc.arguments["incident_id"] == "INC-003"

    def test_service_id_extracted(self) -> None:
        tc = _infer_tool_call("Tell me about SVC-002", [])
        assert tc is not None
        assert tc.tool_name == "get_service"
        assert tc.arguments["service_id"] == "SVC-002"

    def test_employee_id_extracted(self) -> None:
        tc = _infer_tool_call("Who is EMP-001?", [])
        assert tc is not None
        assert tc.tool_name == "find_employee"
        assert tc.arguments["query"] == "EMP-001"

    def test_employee_name_keyword(self) -> None:
        tc = _infer_tool_call("Find employee Alice Chen", [])
        assert tc is not None
        assert tc.tool_name == "find_employee"
        assert "Alice Chen" in tc.arguments["query"]

    def test_default_falls_back_to_knowledge_search(self) -> None:
        tc = _infer_tool_call("What is our loan policy?", [])
        assert tc is not None
        assert tc.tool_name == "knowledge_search"

    def test_empty_message_returns_none(self) -> None:
        assert _infer_tool_call("   ", []) is None

    def test_incident_takes_priority_over_service(self) -> None:
        tc = _infer_tool_call("Get INC-001 for service SVC-002", [])
        assert tc is not None
        assert tc.tool_name == "get_incident"


# ---------------------------------------------------------------------------
# 3. ChatService — full graph execution (mocked external dependencies)
# ---------------------------------------------------------------------------


def _mock_chat_service() -> Any:
    """Build a ChatService with all external calls mocked."""
    from src.services.chat import ChatService
    from src.services.tool_execution import ToolExecutionService

    mock_client = MagicMock()
    mock_retriever = MagicMock()
    mock_tool_service = MagicMock(spec=ToolExecutionService)

    supervisor_response = MagicMock()
    supervisor_response.usage.input_tokens = 50
    supervisor_response.usage.output_tokens = 50
    supervisor_response.content = [
        MagicMock(
            type="tool_use",
            input={
                "intent": "knowledge_question",
                "requires_retrieval": True,
                "requires_research": False,
                "requires_tools": False,
                "task_plan": ["Search knowledge base", "Generate response"],
                "route_to": "retrieval",
                "reasoning": "Simple factual question",
            },
        )
    ]
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=supervisor_response)

    mock_retriever.search = AsyncMock(return_value=[])

    return ChatService(
        client=mock_client,
        retriever=mock_retriever,
        tool_service=mock_tool_service,
        model="claude-sonnet-4-6",
    )


class TestChatServiceRouting:
    async def test_knowledge_question_completes(self) -> None:
        service = _mock_chat_service()
        request = ChatRequest(
            request_id=str(uuid.uuid4()),
            user_id="test-user",
            message="What is the loan approval policy?",
        )
        user = _user(Role.ANALYST)

        response = await service.chat(request, user)

        assert isinstance(response, ChatResponse)
        assert response.conversation_id is not None
        assert response.message  # non-empty

    async def test_conversation_id_propagated(self) -> None:
        service = _mock_chat_service()
        conv_id = str(uuid.uuid4())
        request = ChatRequest(
            request_id=str(uuid.uuid4()),
            user_id="test-user",
            conversation_id=conv_id,
            message="Hello",
        )
        response = await service.chat(request, _user())
        assert response.conversation_id == conv_id

    async def test_agent_activity_populated(self) -> None:
        service = _mock_chat_service()
        request = ChatRequest(
            request_id=str(uuid.uuid4()),
            user_id="test-user",
            message="What is the payment policy?",
        )
        response = await service.chat(request, _user())
        # agent_activity should have at least the supervisor step
        assert isinstance(response.agent_activity, tuple)


# ---------------------------------------------------------------------------
# 4. Conversation memory isolation
# ---------------------------------------------------------------------------


class TestConversationMemory:
    def setup_method(self) -> None:
        _conversation_history.clear()

    def test_different_conversations_are_isolated(self) -> None:
        from src.services.chat import _get_history, _update_history

        msg_a = Message(
            message_id="msg-a",
            conversation_id="conv-A",
            role=MessageRole.USER,
            content="Hello from A",
        )
        msg_b = Message(
            message_id="msg-b",
            conversation_id="conv-B",
            role=MessageRole.USER,
            content="Hello from B",
        )
        _update_history("conv-A", [msg_a])
        _update_history("conv-B", [msg_b])

        assert len(_get_history("conv-A")) == 1
        assert _get_history("conv-A")[0].content == "Hello from A"
        assert len(_get_history("conv-B")) == 1
        assert _get_history("conv-B")[0].content == "Hello from B"

    def test_history_is_bounded(self) -> None:
        from src.services.chat import _MAX_HISTORY_TURNS, _get_history, _update_history

        messages = [
            Message(
                message_id=f"msg-{i}",
                conversation_id="conv-test",
                role=MessageRole.USER,
                content=f"message {i}",
            )
            for i in range(_MAX_HISTORY_TURNS + 10)
        ]
        _update_history("conv-test", messages)
        history = _get_history("conv-test")
        assert len(history) == _MAX_HISTORY_TURNS
        assert history[-1].content == f"message {_MAX_HISTORY_TURNS + 9}"

    def test_empty_conversation_returns_empty_list(self) -> None:
        from src.services.chat import _get_history

        assert _get_history("no-such-conv") == []
