"""Tests for MemoryService.

All external dependencies (store, summariser, checkpoint saver) are replaced
with mocks so no real database or LLM calls are made.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.chat import Conversation, Message
from src.models.enums import MessageRole, Role
from src.memory.models import ConversationSummary, UserContext
from src.memory.service import MemoryService
from src.memory.window import RECENT_WINDOW_SIZE, SUMMARISE_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def _user_ctx() -> UserContext:
    return UserContext(user_id="user-0001", roles=(Role.ENGINEER,))


def _msg(index: int) -> Message:
    return Message(
        message_id=f"msg-{index:04d}",
        conversation_id="conv-0001",
        role=MessageRole.USER,
        content=f"Message {index}",
        created_at=_NOW,
    )


def _conv(n: int) -> Conversation:
    return Conversation(
        conversation_id="conv-0001",
        user_id="user-0001",
        messages=tuple(_msg(i) for i in range(n)),
    )


def _make_service(
    *,
    conv: Conversation | None = None,
    summary: ConversationSummary | None = None,
) -> MemoryService:
    store = AsyncMock()
    store.load_conversation.return_value = conv
    store.load_latest_summary.return_value = summary
    store.append_message.return_value = None
    store.save_summary.return_value = None

    summariser = AsyncMock()
    summariser.summarise.return_value = ConversationSummary(
        summary_id="sum-0001",
        conversation_id="conv-0001",
        content="Auto-generated summary.",
        message_count=SUMMARISE_THRESHOLD,
    )

    checkpoint_saver = MagicMock()

    return MemoryService(
        store=store,
        summariser=summariser,
        checkpoint_saver=checkpoint_saver,
    )


# ---------------------------------------------------------------------------
# get_window
# ---------------------------------------------------------------------------


class TestGetWindow:
    async def test_returns_empty_window_for_new_conversation(self) -> None:
        svc = _make_service(conv=None)
        window = await svc.get_window("conv-0001", _user_ctx())
        assert window.total_message_count == 0
        assert window.recent_messages == ()
        assert window.summary is None

    async def test_window_conversation_id_matches(self) -> None:
        svc = _make_service(conv=None)
        window = await svc.get_window("conv-0001", _user_ctx())
        assert window.conversation_id == "conv-0001"

    async def test_returns_recent_messages_for_existing_conversation(self) -> None:
        conv = _conv(5)
        svc = _make_service(conv=conv)
        window = await svc.get_window("conv-0001", _user_ctx())
        assert len(window.recent_messages) == 5

    async def test_long_conversation_caps_recent_messages(self) -> None:
        conv = _conv(RECENT_WINDOW_SIZE + 5)
        svc = _make_service(conv=conv)
        window = await svc.get_window("conv-0001", _user_ctx())
        assert len(window.recent_messages) == RECENT_WINDOW_SIZE

    async def test_summary_included_when_available(self) -> None:
        conv = _conv(15)
        summary = ConversationSummary(
            summary_id="sum-0001",
            conversation_id="conv-0001",
            content="Prior summary.",
            message_count=5,
        )
        svc = _make_service(conv=conv, summary=summary)
        window = await svc.get_window("conv-0001", _user_ctx())
        assert window.summary is summary

    async def test_user_context_carried_through(self) -> None:
        svc = _make_service(conv=None)
        ctx = _user_ctx()
        window = await svc.get_window("conv-0001", ctx)
        assert window.user_context is ctx


# ---------------------------------------------------------------------------
# append_message
# ---------------------------------------------------------------------------


class TestAppendMessage:
    async def test_returns_updated_conversation(self) -> None:
        conv = _conv(2)
        svc = _make_service()
        new_msg = _msg(99)
        updated = await svc.append_message(new_msg, conv)
        assert len(updated.messages) == 3

    async def test_new_message_is_last_in_tuple(self) -> None:
        conv = _conv(2)
        svc = _make_service()
        new_msg = _msg(99)
        updated = await svc.append_message(new_msg, conv)
        assert updated.messages[-1].message_id == "msg-0099"

    async def test_original_conversation_not_mutated(self) -> None:
        conv = _conv(2)
        svc = _make_service()
        await svc.append_message(_msg(99), conv)
        assert len(conv.messages) == 2

    async def test_store_append_called(self) -> None:
        conv = _conv(2)
        store = AsyncMock()
        svc = MemoryService(store=store, summariser=AsyncMock(), checkpoint_saver=MagicMock())
        msg = _msg(99)
        await svc.append_message(msg, conv)
        store.append_message.assert_called_once_with(msg)


# ---------------------------------------------------------------------------
# maybe_summarise
# ---------------------------------------------------------------------------


class TestMaybeSummarise:
    async def test_returns_none_when_below_threshold(self) -> None:
        svc = _make_service()
        result = await svc.maybe_summarise(_conv(SUMMARISE_THRESHOLD))
        assert result is None

    async def test_returns_none_when_empty_conversation(self) -> None:
        svc = _make_service()
        result = await svc.maybe_summarise(_conv(0))
        assert result is None

    async def test_returns_summary_when_above_threshold(self) -> None:
        svc = _make_service()
        result = await svc.maybe_summarise(_conv(SUMMARISE_THRESHOLD + 1))
        assert result is not None
        assert isinstance(result, ConversationSummary)

    async def test_store_save_summary_called(self) -> None:
        store = AsyncMock()
        store.save_summary.return_value = None
        dummy_summary = ConversationSummary(
            summary_id="sum-0001",
            conversation_id="conv-0001",
            content="Summary.",
            message_count=SUMMARISE_THRESHOLD,
        )
        summariser = AsyncMock()
        summariser.summarise.return_value = dummy_summary

        svc = MemoryService(store=store, summariser=summariser, checkpoint_saver=MagicMock())
        await svc.maybe_summarise(_conv(SUMMARISE_THRESHOLD + 1))
        store.save_summary.assert_called_once_with(dummy_summary)

    async def test_summariser_receives_older_messages(self) -> None:
        summariser = AsyncMock()
        summariser.summarise.return_value = ConversationSummary(
            summary_id="sum-0001",
            conversation_id="conv-0001",
            content="S.",
            message_count=5,
        )
        store = AsyncMock()
        svc = MemoryService(store=store, summariser=summariser, checkpoint_saver=MagicMock())

        total = SUMMARISE_THRESHOLD + 5
        await svc.maybe_summarise(_conv(total))

        call_args = summariser.summarise.call_args
        msgs = call_args[0][1]
        assert len(msgs) == total - RECENT_WINDOW_SIZE


# ---------------------------------------------------------------------------
# get_checkpoint_saver
# ---------------------------------------------------------------------------


class TestGetCheckpointSaver:
    def test_returns_injected_checkpoint_saver(self) -> None:
        checkpoint_saver = MagicMock()
        svc = MemoryService(
            store=AsyncMock(),
            summariser=AsyncMock(),
            checkpoint_saver=checkpoint_saver,
        )
        assert svc.get_checkpoint_saver() is checkpoint_saver
