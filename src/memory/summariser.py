"""Summariser protocol and NullSummariser (no-op default).

A ``Summariser`` compresses older conversation turns into a
``ConversationSummary``.  The ``NullSummariser`` is suitable for testing
and local development; a real LLM-backed implementation would replace it in
production.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable
from uuid import uuid4

from src.models.chat import Message
from src.models.common import EntityId
from src.memory.models import ConversationSummary


@runtime_checkable
class Summariser(Protocol):
    """Structural protocol for conversation summarisers."""

    async def summarise(
        self,
        conversation_id: EntityId,
        messages: Sequence[Message],
    ) -> ConversationSummary:
        """Compress ``messages`` into a ``ConversationSummary``.

        Parameters
        ----------
        conversation_id:
            The conversation these messages belong to.
        messages:
            The messages to summarise.  Must be non-empty.

        Raises
        ------
        ValueError
            If ``messages`` is empty.
        """
        ...


class NullSummariser:
    """No-op summariser that concatenates message content.

    Suitable for testing and local development.  In production, replace with
    an LLM-backed implementation that generates a coherent prose summary.
    """

    async def summarise(
        self,
        conversation_id: EntityId,
        messages: Sequence[Message],
    ) -> ConversationSummary:
        if not messages:
            raise ValueError("Cannot summarise an empty message sequence")

        lines = [
            f"[{m.role.value}] {m.content[:200]}"
            for m in messages
        ]
        content = f"Summary of {len(messages)} message(s):\n" + "\n".join(lines)

        return ConversationSummary(
            summary_id=str(uuid4()),
            conversation_id=conversation_id,
            content=content,
            message_count=len(messages),
        )
