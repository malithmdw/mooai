"""Tests for ConversationStore.

The SQLAlchemy async engine is mocked so no real database is needed.
Tests verify that the store correctly translates between domain models and
the SQL layer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.chat import Conversation, Message
from src.models.enums import MessageRole
from src.memory.models import ConversationSummary
from src.memory.store import ConversationStore


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _cursor(
    rows: list[dict] | None = None,
    scalar_value: int = 0,
) -> MagicMock:
    """Return a mock CursorResult whose mappings() returns the given rows."""
    cursor = MagicMock()
    row_list = rows or []
    mappings = MagicMock()
    mappings.first.return_value = row_list[0] if row_list else None
    mappings.all.return_value = row_list
    cursor.mappings.return_value = mappings
    cursor.scalar.return_value = scalar_value
    return cursor


def _make_engine(cursors: list) -> tuple[MagicMock, AsyncMock]:
    """Return (engine, conn) where conn.execute returns cursors in sequence."""
    pending = list(cursors)

    conn = AsyncMock()

    async def _execute(stmt, *args, **kwargs):  # noqa: ARG001
        return pending.pop(0) if pending else _cursor()

    conn.execute.side_effect = _execute
    conn.run_sync = AsyncMock()

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=None)

    engine = MagicMock()
    engine.connect.return_value = ctx
    engine.begin.return_value = ctx

    return engine, conn


_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _conv_row(conv_id: str = "conv-0001") -> dict:
    return {
        "conversation_id": conv_id,
        "user_id": "user-0001",
        "title": None,
        "created_at": _NOW,
    }


def _msg_row(msg_id: str = "msg-0001", conv_id: str = "conv-0001") -> dict:
    return {
        "message_id": msg_id,
        "conversation_id": conv_id,
        "role": "user",
        "content": "Hello",
        "created_at": _NOW,
    }


def _sum_row(sum_id: str = "sum-0001", conv_id: str = "conv-0001") -> dict:
    return {
        "summary_id": sum_id,
        "conversation_id": conv_id,
        "content": "A summary.",
        "message_count": 5,
        "created_at": _NOW,
    }


# ---------------------------------------------------------------------------
# load_conversation
# ---------------------------------------------------------------------------


class TestLoadConversation:
    async def test_returns_none_when_not_found(self) -> None:
        engine, _ = _make_engine([_cursor(rows=None), _cursor(rows=[])])
        store = ConversationStore(engine)
        result = await store.load_conversation("conv-0001")
        assert result is None

    async def test_returns_conversation_when_found(self) -> None:
        engine, _ = _make_engine([_cursor([_conv_row()]), _cursor([])])
        store = ConversationStore(engine)
        result = await store.load_conversation("conv-0001")
        assert result is not None
        assert result.conversation_id == "conv-0001"
        assert result.user_id == "user-0001"

    async def test_messages_attached_to_conversation(self) -> None:
        engine, _ = _make_engine([_cursor([_conv_row()]), _cursor([_msg_row()])])
        store = ConversationStore(engine)
        result = await store.load_conversation("conv-0001")
        assert result is not None
        assert len(result.messages) == 1
        assert result.messages[0].message_id == "msg-0001"

    async def test_multiple_messages_ordered_correctly(self) -> None:
        msgs = [_msg_row(f"msg-{i:04d}") for i in range(3)]
        engine, _ = _make_engine([_cursor([_conv_row()]), _cursor(msgs)])
        store = ConversationStore(engine)
        result = await store.load_conversation("conv-0001")
        assert result is not None
        assert [m.message_id for m in result.messages] == [f"msg-{i:04d}" for i in range(3)]

    async def test_message_role_parsed_correctly(self) -> None:
        msg_row = _msg_row()
        msg_row["role"] = "assistant"
        engine, _ = _make_engine([_cursor([_conv_row()]), _cursor([msg_row])])
        store = ConversationStore(engine)
        result = await store.load_conversation("conv-0001")
        assert result is not None
        assert result.messages[0].role == MessageRole.ASSISTANT

    async def test_title_preserved(self) -> None:
        row = _conv_row()
        row["title"] = "My Chat"
        engine, _ = _make_engine([_cursor([row]), _cursor([])])
        store = ConversationStore(engine)
        result = await store.load_conversation("conv-0001")
        assert result is not None
        assert result.title == "My Chat"


# ---------------------------------------------------------------------------
# save_conversation
# ---------------------------------------------------------------------------


class TestSaveConversation:
    async def test_inserts_new_conversation(self) -> None:
        # exists check returns 0 → triggers INSERT
        engine, conn = _make_engine([_cursor(scalar_value=0)])
        store = ConversationStore(engine)
        conv = Conversation(
            conversation_id="conv-0001",
            user_id="user-0001",
        )
        await store.save_conversation(conv)
        assert conn.execute.call_count == 2  # count check + INSERT

    async def test_updates_existing_conversation(self) -> None:
        # exists check returns 1 → triggers UPDATE
        engine, conn = _make_engine([_cursor(scalar_value=1)])
        store = ConversationStore(engine)
        conv = Conversation(
            conversation_id="conv-0001",
            user_id="user-0001",
            title="Updated title",
        )
        await store.save_conversation(conv)
        assert conn.execute.call_count == 2  # count check + UPDATE


# ---------------------------------------------------------------------------
# append_message / load_messages
# ---------------------------------------------------------------------------


class TestMessages:
    async def test_append_message_calls_execute(self) -> None:
        engine, conn = _make_engine([_cursor()])
        store = ConversationStore(engine)
        msg = Message(
            message_id="msg-0001",
            conversation_id="conv-0001",
            role=MessageRole.USER,
            content="Hello",
        )
        await store.append_message(msg)
        conn.execute.assert_called_once()

    async def test_load_messages_returns_empty_list_when_none(self) -> None:
        engine, _ = _make_engine([_cursor(rows=[])])
        store = ConversationStore(engine)
        result = await store.load_messages("conv-0001")
        assert result == []

    async def test_load_messages_returns_list_of_message(self) -> None:
        engine, _ = _make_engine([_cursor([_msg_row()])])
        store = ConversationStore(engine)
        result = await store.load_messages("conv-0001")
        assert len(result) == 1
        assert result[0].message_id == "msg-0001"


# ---------------------------------------------------------------------------
# save_summary / load_latest_summary
# ---------------------------------------------------------------------------


class TestSummaries:
    async def test_save_summary_calls_execute(self) -> None:
        engine, conn = _make_engine([_cursor()])
        store = ConversationStore(engine)
        summary = ConversationSummary(
            summary_id="sum-0001",
            conversation_id="conv-0001",
            content="A summary.",
            message_count=5,
        )
        await store.save_summary(summary)
        conn.execute.assert_called_once()

    async def test_load_latest_summary_returns_none_when_not_found(self) -> None:
        engine, _ = _make_engine([_cursor(rows=[])])
        store = ConversationStore(engine)
        result = await store.load_latest_summary("conv-0001")
        assert result is None

    async def test_load_latest_summary_returns_summary(self) -> None:
        engine, _ = _make_engine([_cursor([_sum_row()])])
        store = ConversationStore(engine)
        result = await store.load_latest_summary("conv-0001")
        assert result is not None
        assert result.summary_id == "sum-0001"
        assert result.message_count == 5
        assert result.content == "A summary."

    async def test_load_latest_summary_parses_conversation_id(self) -> None:
        engine, _ = _make_engine([_cursor([_sum_row(conv_id="conv-9999")])])
        store = ConversationStore(engine)
        result = await store.load_latest_summary("conv-9999")
        assert result is not None
        assert result.conversation_id == "conv-9999"
