"""MemoryService — orchestrates conversation persistence and context-window management.

The service is the single entry point for all memory operations.  It wires
together the ``ConversationStore`` (persistence), the ``Summariser``
(compression), and the ``PostgresCheckpointSaver`` (LangGraph state) so
that callers never need to interact with those components directly.
"""

from __future__ import annotations

import logging

from src.core.logging import get_logger, log_debug, log_info
from src.models.chat import Conversation, Message
from src.models.common import EntityId
from src.memory.checkpoint import PostgresCheckpointSaver
from src.memory.models import ConversationSummary, MemoryWindow, UserContext
from src.memory.store import ConversationStore
from src.memory.summariser import Summariser
from src.memory.window import build_window, messages_to_summarise, should_summarise

logger: logging.Logger = get_logger(__name__)


class MemoryService:
    """Orchestrates conversation persistence and context-window management.

    Inject this service into request handlers and agent nodes rather than
    accessing the store or summariser directly.
    """

    def __init__(
        self,
        store: ConversationStore,
        summariser: Summariser,
        checkpoint_saver: PostgresCheckpointSaver,
    ) -> None:
        self._store = store
        self._summariser = summariser
        self._checkpoint_saver = checkpoint_saver

    async def get_window(
        self,
        conversation_id: EntityId,
        user_context: UserContext,
    ) -> MemoryWindow:
        """Return the context window for a conversation turn.

        Loads the conversation and its latest summary from the store and
        combines them into a ``MemoryWindow`` with the most-recent messages
        verbatim and older context compressed.  Returns an empty window for
        new conversations.
        """
        conversation = await self._store.load_conversation(conversation_id)
        if conversation is None:
            log_debug(
                logger,
                "memory.service.window_empty",
                "No conversation found; returning empty window",
                conversation_id=conversation_id,
            )
            return MemoryWindow(
                conversation_id=conversation_id,
                user_context=user_context,
                recent_messages=(),
                summary=None,
                total_message_count=0,
            )

        summary = await self._store.load_latest_summary(conversation_id)
        window = build_window(conversation, user_context, summary=summary)

        log_debug(
            logger,
            "memory.service.window_built",
            "Memory window built",
            conversation_id=conversation_id,
            total_messages=window.total_message_count,
            recent_messages=len(window.recent_messages),
            has_summary=summary is not None,
        )
        return window

    async def append_message(
        self,
        message: Message,
        conversation: Conversation,
    ) -> Conversation:
        """Persist ``message`` and return an updated ``Conversation`` snapshot.

        The store is updated and a new immutable ``Conversation`` is returned
        with the message appended to its ``messages`` tuple.
        """
        await self._store.append_message(message)
        return conversation.model_copy(
            update={"messages": (*conversation.messages, message)}
        )

    async def maybe_summarise(
        self,
        conversation: Conversation,
    ) -> ConversationSummary | None:
        """Summarise older turns if the conversation has grown long enough.

        Returns the new ``ConversationSummary`` if summarisation happened,
        or ``None`` if the conversation is below the threshold or there is
        nothing to summarise.
        """
        if not should_summarise(conversation):
            return None

        msgs = messages_to_summarise(conversation)
        if not msgs:
            return None

        summary = await self._summariser.summarise(
            conversation.conversation_id,
            msgs,
        )
        await self._store.save_summary(summary)

        log_info(
            logger,
            "memory.service.summarised",
            "Conversation summarised",
            conversation_id=conversation.conversation_id,
            messages_summarised=len(msgs),
            summary_id=summary.summary_id,
        )
        return summary

    def get_checkpoint_saver(self) -> PostgresCheckpointSaver:
        """Return the LangGraph checkpoint saver for agent graph wiring."""
        return self._checkpoint_saver
