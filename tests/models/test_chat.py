"""Tests for `src.models.chat`: `Message`, `Conversation`, `ChatRequest`,
`ChatResponse`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.chat import ChatRequest, ChatResponse, Conversation, Message
from src.models.enums import MessageRole
from src.models.evidence import Citation, Evidence


class TestMessage:
    def test_valid_message_is_immutable(self) -> None:
        message = Message(
            message_id="msg-1",
            conversation_id="conv-1",
            role=MessageRole.USER,
            content="What is our loan approval threshold?",
        )
        with pytest.raises(ValidationError):
            message.content = "edited"  # type: ignore[misc]

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_rejects_empty_content(self, blank: str) -> None:
        with pytest.raises(ValidationError, match="content"):
            Message(
                message_id="msg-1",
                conversation_id="conv-1",
                role=MessageRole.USER,
                content=blank,
            )


class TestChatRequest:
    def test_valid_chat_request(self) -> None:
        request = ChatRequest(request_id="req-1", user_id="user-1", message="Hello")
        assert request.conversation_id is None

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_rejects_empty_user_message(self, blank: str) -> None:
        with pytest.raises(ValidationError, match="message"):
            ChatRequest(request_id="req-1", user_id="user-1", message=blank)


class TestConversation:
    def test_valid_conversation_with_messages(self) -> None:
        message = Message(
            message_id="msg-1",
            conversation_id="conv-1",
            role=MessageRole.USER,
            content="Hello",
        )
        conversation = Conversation(conversation_id="conv-1", user_id="user-1", messages=(message,))
        assert conversation.messages == (message,)
        with pytest.raises(ValidationError):
            conversation.messages = ()  # type: ignore[misc]


class TestChatResponse:
    def test_valid_response_with_matching_citation(self) -> None:
        evidence = Evidence(
            evidence_id="ev-1",
            document_id="doc-1",
            excerpt="Loans over $1M require dual sign-off.",
            relevance_score=0.9,
        )
        response = ChatResponse(
            response_id="resp-1",
            conversation_id="conv-1",
            message="Loans over $1M need dual sign-off [1].",
            evidence=(evidence,),
            citations=(Citation(evidence_id="ev-1", reference_number=1),),
        )
        assert response.citations[0].evidence_id == "ev-1"

    def test_rejects_citation_referencing_unknown_evidence(self) -> None:
        evidence = Evidence(
            evidence_id="ev-1",
            document_id="doc-1",
            excerpt="Loans over $1M require dual sign-off.",
            relevance_score=0.9,
        )
        with pytest.raises(ValidationError, match="evidence not included"):
            ChatResponse(
                response_id="resp-1",
                conversation_id="conv-1",
                message="Loans over $1M need dual sign-off [1].",
                evidence=(evidence,),
                citations=(Citation(evidence_id="ev-does-not-exist", reference_number=1),),
            )

    def test_rejects_empty_response_message(self) -> None:
        with pytest.raises(ValidationError, match="message"):
            ChatResponse(response_id="resp-1", conversation_id="conv-1", message="   ")
