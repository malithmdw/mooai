"""Memory domain models — UserContext, ConversationSummary, MemoryWindow."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.models.chat import Message
from src.models.common import EntityId, NonEmptyStr, utcnow
from src.models.enums import Role


class UserContext(BaseModel):
    """Contextual information about the requesting user for memory operations."""

    model_config = ConfigDict(frozen=True)

    user_id: EntityId
    roles: tuple[Role, ...] = Field(default_factory=tuple)
    display_name: str | None = None


class ConversationSummary(BaseModel):
    """A compressed summary of older conversation turns.

    Generated when a conversation exceeds SUMMARISE_THRESHOLD messages so
    that the full history is never blindly forwarded to the LLM.  The
    ``message_count`` field records how many messages from the start of the
    conversation are covered by this summary.
    """

    model_config = ConfigDict(frozen=True)

    summary_id: EntityId
    conversation_id: EntityId
    content: NonEmptyStr
    message_count: int = Field(gt=0)
    created_at: datetime = Field(default_factory=utcnow)


class MemoryWindow(BaseModel):
    """The resolved memory context to inject into an LLM turn.

    Combines a compressed summary of older turns with the verbatim recent
    message window.  Downstream consumers (agents, prompt builders) should
    use ``recent_messages`` and ``summary`` rather than the raw conversation
    history.
    """

    model_config = ConfigDict(frozen=True)

    conversation_id: EntityId
    user_context: UserContext
    recent_messages: tuple[Message, ...]
    summary: ConversationSummary | None = None
    total_message_count: int = Field(ge=0)
