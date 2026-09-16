"""Context-window management for conversations.

Pure functions — no I/O.  The window builder decides which messages to send
verbatim and which to replace with a summary, keeping the LLM context lean
without losing conversational coherence.
"""

from __future__ import annotations

from src.models.chat import Conversation, Message
from src.memory.models import ConversationSummary, MemoryWindow, UserContext

RECENT_WINDOW_SIZE: int = 10
"""Number of most-recent messages kept verbatim in the context window."""

SUMMARISE_THRESHOLD: int = 20
"""Total message count at which older messages should be collapsed into a summary."""


def build_window(
    conversation: Conversation,
    user_context: UserContext,
    *,
    summary: ConversationSummary | None = None,
) -> MemoryWindow:
    """Build the context window for a conversation turn.

    Returns the last ``RECENT_WINDOW_SIZE`` messages verbatim plus any
    existing summary covering older turns.
    """
    messages = conversation.messages
    recent = messages[-RECENT_WINDOW_SIZE:] if len(messages) > RECENT_WINDOW_SIZE else messages
    return MemoryWindow(
        conversation_id=conversation.conversation_id,
        user_context=user_context,
        recent_messages=recent,
        summary=summary,
        total_message_count=len(messages),
    )


def should_summarise(conversation: Conversation) -> bool:
    """Return True when the conversation is long enough to warrant summarisation."""
    return len(conversation.messages) > SUMMARISE_THRESHOLD


def messages_to_summarise(
    conversation: Conversation,
    *,
    keep_recent: int = RECENT_WINDOW_SIZE,
) -> tuple[Message, ...]:
    """Return the messages that should be compressed into a summary.

    Excludes the most-recent ``keep_recent`` messages, which are kept
    verbatim in the context window.  Returns an empty tuple when the
    conversation is too short to have messages outside the recent window.
    """
    all_messages = conversation.messages
    cutoff = max(0, len(all_messages) - keep_recent)
    return all_messages[:cutoff]
