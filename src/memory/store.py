"""Persistent conversation storage backed by PostgreSQL.

Uses SQLAlchemy Core (not ORM) to manage three tables:
  - conversations             one row per conversation thread
  - messages                  one row per message turn
  - conversation_summaries    compressed summaries of older turns

Connections are opened and closed per operation; the caller is responsible
for constructing the store once and sharing it across requests.
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncConnection, create_async_engine

from src.core.logging import get_logger, log_debug
from src.models.chat import Conversation, Message
from src.models.common import EntityId
from src.models.enums import MessageRole
from src.memory.models import ConversationSummary

logger: logging.Logger = get_logger(__name__)

_metadata = sa.MetaData()

_conversations = sa.Table(
    "conversations",
    _metadata,
    sa.Column("conversation_id", sa.String(128), primary_key=True),
    sa.Column("user_id", sa.String(128), nullable=False),
    sa.Column("title", sa.Text, nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

_messages = sa.Table(
    "messages",
    _metadata,
    sa.Column("message_id", sa.String(128), primary_key=True),
    sa.Column(
        "conversation_id",
        sa.String(128),
        sa.ForeignKey("conversations.conversation_id"),
        nullable=False,
        index=True,
    ),
    sa.Column("role", sa.String(16), nullable=False),
    sa.Column("content", sa.Text, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

_summaries = sa.Table(
    "conversation_summaries",
    _metadata,
    sa.Column("summary_id", sa.String(128), primary_key=True),
    sa.Column(
        "conversation_id",
        sa.String(128),
        sa.ForeignKey("conversations.conversation_id"),
        nullable=False,
        index=True,
    ),
    sa.Column("content", sa.Text, nullable=False),
    sa.Column("message_count", sa.Integer, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)


class ConversationStore:
    """Async SQLAlchemy store for conversations, messages, and summaries."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @classmethod
    def from_url(cls, url: str) -> "ConversationStore":
        """Create a store from a SQLAlchemy async database URL."""
        return cls(create_async_engine(url))

    async def create_tables(self) -> None:
        """Create all tables if they do not already exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all, checkfirst=True)

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def save_conversation(self, conversation: Conversation) -> None:
        """Persist conversation metadata (not messages — use append_message)."""
        async with self._engine.begin() as conn:
            exists = (
                await conn.execute(
                    sa.select(sa.func.count())
                    .select_from(_conversations)
                    .where(_conversations.c.conversation_id == conversation.conversation_id)
                )
            ).scalar() or 0

            if exists:
                await conn.execute(
                    sa.update(_conversations)
                    .where(_conversations.c.conversation_id == conversation.conversation_id)
                    .values(title=conversation.title)
                )
            else:
                await conn.execute(
                    sa.insert(_conversations).values(
                        conversation_id=conversation.conversation_id,
                        user_id=conversation.user_id,
                        title=conversation.title,
                        created_at=conversation.created_at,
                    )
                )

        log_debug(
            logger,
            "memory.store.conversation_saved",
            "Conversation saved",
            conversation_id=conversation.conversation_id,
        )

    async def load_conversation(self, conversation_id: EntityId) -> Conversation | None:
        """Load a conversation and all its messages, or None if not found."""
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    sa.select(_conversations).where(
                        _conversations.c.conversation_id == conversation_id
                    )
                )
            ).mappings().first()

            if row is None:
                return None

            messages = await self._load_messages_conn(conn, conversation_id)

        return Conversation(
            conversation_id=str(row["conversation_id"]),
            user_id=str(row["user_id"]),
            title=row["title"],
            messages=tuple(messages),
            created_at=row["created_at"],
        )

    async def _load_messages_conn(
        self, conn: AsyncConnection, conversation_id: str
    ) -> list[Message]:
        rows = (
            await conn.execute(
                sa.select(_messages)
                .where(_messages.c.conversation_id == conversation_id)
                .order_by(_messages.c.created_at.asc())
            )
        ).mappings().all()
        return [
            Message(
                message_id=str(r["message_id"]),
                conversation_id=str(r["conversation_id"]),
                role=MessageRole(str(r["role"])),
                content=str(r["content"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def append_message(self, message: Message) -> None:
        """Persist a single message."""
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.insert(_messages).values(
                    message_id=message.message_id,
                    conversation_id=message.conversation_id,
                    role=message.role.value,
                    content=message.content,
                    created_at=message.created_at,
                )
            )

    async def load_messages(
        self,
        conversation_id: EntityId,
        *,
        limit: int | None = None,
    ) -> list[Message]:
        """Load messages for a conversation, oldest first."""
        stmt = (
            sa.select(_messages)
            .where(_messages.c.conversation_id == conversation_id)
            .order_by(_messages.c.created_at.asc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)

        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()

        return [
            Message(
                message_id=str(r["message_id"]),
                conversation_id=str(r["conversation_id"]),
                role=MessageRole(str(r["role"])),
                content=str(r["content"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    async def save_summary(self, summary: ConversationSummary) -> None:
        """Persist a conversation summary."""
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.insert(_summaries).values(
                    summary_id=summary.summary_id,
                    conversation_id=summary.conversation_id,
                    content=summary.content,
                    message_count=summary.message_count,
                    created_at=summary.created_at,
                )
            )

    async def load_latest_summary(
        self, conversation_id: EntityId
    ) -> ConversationSummary | None:
        """Load the most recently created summary for a conversation."""
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    sa.select(_summaries)
                    .where(_summaries.c.conversation_id == conversation_id)
                    .order_by(_summaries.c.created_at.desc())
                    .limit(1)
                )
            ).mappings().first()

        if row is None:
            return None

        return ConversationSummary(
            summary_id=str(row["summary_id"]),
            conversation_id=str(row["conversation_id"]),
            content=str(row["content"]),
            message_count=int(row["message_count"]),
            created_at=row["created_at"],
        )
