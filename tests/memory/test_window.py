"""Tests for context-window management (pure functions).

All functions in window.py are pure (no I/O), so tests use deterministic
in-memory inputs and verify exact behaviour.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.models.chat import Conversation, Message
from src.models.enums import MessageRole, Role
from src.memory.models import ConversationSummary, UserContext
from src.memory.window import (
    RECENT_WINDOW_SIZE,
    SUMMARISE_THRESHOLD,
    build_window,
    messages_to_summarise,
    should_summarise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(index: int, role: MessageRole = MessageRole.USER) -> Message:
    return Message(
        message_id=f"msg-{index:04d}",
        conversation_id="conv-0001",
        role=role,
        content=f"Message {index}",
        created_at=datetime(2024, 1, 1, 0, index, tzinfo=UTC),
    )


def _conv(n: int) -> Conversation:
    return Conversation(
        conversation_id="conv-0001",
        user_id="user-0001",
        messages=tuple(_msg(i) for i in range(n)),
    )


def _user_ctx() -> UserContext:
    return UserContext(user_id="user-0001", roles=(Role.ENGINEER,))


def _summary(message_count: int = 5) -> ConversationSummary:
    return ConversationSummary(
        summary_id="sum-0001",
        conversation_id="conv-0001",
        content="Previous context summary.",
        message_count=message_count,
    )


# ---------------------------------------------------------------------------
# build_window
# ---------------------------------------------------------------------------


class TestBuildWindow:
    def test_short_conversation_returns_all_messages(self) -> None:
        conv = _conv(5)
        window = build_window(conv, _user_ctx())
        assert len(window.recent_messages) == 5

    def test_long_conversation_returns_only_recent(self) -> None:
        conv = _conv(RECENT_WINDOW_SIZE + 5)
        window = build_window(conv, _user_ctx())
        assert len(window.recent_messages) == RECENT_WINDOW_SIZE

    def test_recent_messages_are_the_last_n(self) -> None:
        conv = _conv(15)
        window = build_window(conv, _user_ctx())
        expected_ids = {f"msg-{i:04d}" for i in range(5, 15)}
        actual_ids = {m.message_id for m in window.recent_messages}
        assert actual_ids == expected_ids

    def test_exactly_window_size_returns_all(self) -> None:
        conv = _conv(RECENT_WINDOW_SIZE)
        window = build_window(conv, _user_ctx())
        assert len(window.recent_messages) == RECENT_WINDOW_SIZE

    def test_total_message_count_reflects_full_history(self) -> None:
        conv = _conv(25)
        window = build_window(conv, _user_ctx())
        assert window.total_message_count == 25

    def test_summary_included_when_provided(self) -> None:
        conv = _conv(15)
        summary = _summary()
        window = build_window(conv, _user_ctx(), summary=summary)
        assert window.summary is summary

    def test_summary_is_none_when_omitted(self) -> None:
        window = build_window(_conv(5), _user_ctx())
        assert window.summary is None

    def test_user_context_carried_through(self) -> None:
        ctx = _user_ctx()
        window = build_window(_conv(3), ctx)
        assert window.user_context is ctx

    def test_conversation_id_carried_through(self) -> None:
        window = build_window(_conv(3), _user_ctx())
        assert window.conversation_id == "conv-0001"

    def test_empty_conversation_returns_empty_window(self) -> None:
        window = build_window(_conv(0), _user_ctx())
        assert window.recent_messages == ()
        assert window.total_message_count == 0


# ---------------------------------------------------------------------------
# should_summarise
# ---------------------------------------------------------------------------


class TestShouldSummarise:
    def test_below_threshold_returns_false(self) -> None:
        assert not should_summarise(_conv(SUMMARISE_THRESHOLD - 1))

    def test_exactly_at_threshold_returns_false(self) -> None:
        assert not should_summarise(_conv(SUMMARISE_THRESHOLD))

    def test_one_above_threshold_returns_true(self) -> None:
        assert should_summarise(_conv(SUMMARISE_THRESHOLD + 1))

    def test_far_above_threshold_returns_true(self) -> None:
        assert should_summarise(_conv(100))

    def test_empty_conversation_returns_false(self) -> None:
        assert not should_summarise(_conv(0))


# ---------------------------------------------------------------------------
# messages_to_summarise
# ---------------------------------------------------------------------------


class TestMessagesToSummarise:
    def test_short_conversation_returns_empty(self) -> None:
        result = messages_to_summarise(_conv(RECENT_WINDOW_SIZE))
        assert result == ()

    def test_returns_messages_before_recent_window(self) -> None:
        conv = _conv(15)
        result = messages_to_summarise(conv)
        assert len(result) == 5
        expected_ids = {f"msg-{i:04d}" for i in range(5)}
        assert {m.message_id for m in result} == expected_ids

    def test_does_not_include_recent_messages(self) -> None:
        conv = _conv(15)
        result = messages_to_summarise(conv)
        recent_ids = {f"msg-{i:04d}" for i in range(5, 15)}
        returned_ids = {m.message_id for m in result}
        assert returned_ids.isdisjoint(recent_ids)

    def test_custom_keep_recent_reduces_output(self) -> None:
        conv = _conv(20)
        result = messages_to_summarise(conv, keep_recent=5)
        assert len(result) == 15

    def test_custom_keep_recent_larger_than_history_returns_empty(self) -> None:
        conv = _conv(5)
        result = messages_to_summarise(conv, keep_recent=10)
        assert result == ()

    def test_exactly_window_size_returns_empty(self) -> None:
        result = messages_to_summarise(_conv(RECENT_WINDOW_SIZE))
        assert result == ()

    def test_order_preserved(self) -> None:
        conv = _conv(25)
        result = messages_to_summarise(conv)
        indices = [int(m.message_id.split("-")[1]) for m in result]
        assert indices == sorted(indices)
