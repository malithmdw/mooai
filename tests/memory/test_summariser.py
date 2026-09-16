"""Tests for Summariser protocol and NullSummariser."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.models.chat import Message
from src.models.enums import MessageRole
from src.memory.models import ConversationSummary
from src.memory.summariser import NullSummariser, Summariser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(index: int, role: MessageRole = MessageRole.USER) -> Message:
    return Message(
        message_id=f"msg-{index:04d}",
        conversation_id="conv-0001",
        role=role,
        content=f"Message content {index}",
        created_at=datetime(2024, 1, 1, 0, index, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# NullSummariser
# ---------------------------------------------------------------------------


class TestNullSummariser:
    async def test_returns_summary_model(self) -> None:
        s = NullSummariser()
        result = await s.summarise("conv-0001", [_msg(0)])
        assert isinstance(result, ConversationSummary)

    async def test_conversation_id_matches(self) -> None:
        result = await NullSummariser().summarise("conv-0001", [_msg(0)])
        assert result.conversation_id == "conv-0001"

    async def test_message_count_matches_input(self) -> None:
        msgs = [_msg(i) for i in range(5)]
        result = await NullSummariser().summarise("conv-0001", msgs)
        assert result.message_count == 5

    async def test_content_references_message_count(self) -> None:
        msgs = [_msg(i) for i in range(3)]
        result = await NullSummariser().summarise("conv-0001", msgs)
        assert "3" in result.content

    async def test_content_includes_message_text(self) -> None:
        msg = _msg(7)
        result = await NullSummariser().summarise("conv-0001", [msg])
        assert "Message content 7" in result.content

    async def test_content_includes_role(self) -> None:
        msg = Message(
            message_id="msg-0001",
            conversation_id="conv-0001",
            role=MessageRole.ASSISTANT,
            content="Assistant reply",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        result = await NullSummariser().summarise("conv-0001", [msg])
        assert "assistant" in result.content

    async def test_summary_id_is_unique(self) -> None:
        s = NullSummariser()
        r1 = await s.summarise("conv-0001", [_msg(0)])
        r2 = await s.summarise("conv-0001", [_msg(0)])
        assert r1.summary_id != r2.summary_id

    async def test_empty_messages_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            await NullSummariser().summarise("conv-0001", [])

    async def test_single_message_succeeds(self) -> None:
        result = await NullSummariser().summarise("conv-0001", [_msg(0)])
        assert result.message_count == 1

    async def test_created_at_is_set(self) -> None:
        result = await NullSummariser().summarise("conv-0001", [_msg(0)])
        assert result.created_at is not None


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestSummariserProtocol:
    def test_null_summariser_satisfies_protocol(self) -> None:
        assert isinstance(NullSummariser(), Summariser)

    def test_arbitrary_object_without_method_fails_protocol(self) -> None:
        class NotASummariser:
            pass

        assert not isinstance(NotASummariser(), Summariser)
