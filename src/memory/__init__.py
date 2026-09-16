"""Conversational memory: context-window management and persistent storage.

Public surface
--------------
Models
- ``UserContext``            — user identity carried through memory operations.
- ``ConversationSummary``    — compressed summary of older conversation turns.
- ``MemoryWindow``           — resolved context window for one LLM turn.

Window management (pure functions)
- ``RECENT_WINDOW_SIZE``     — verbatim message count (default 10).
- ``SUMMARISE_THRESHOLD``    — total messages before summarisation (default 20).
- ``build_window``           — builds a ``MemoryWindow`` from a ``Conversation``.
- ``should_summarise``       — True when conversation exceeds the threshold.
- ``messages_to_summarise``  — messages outside the recent window.

Persistence
- ``ConversationStore``      — async SQLAlchemy store (conversations/messages/summaries).

Summarisation
- ``Summariser``             — Protocol for summary generators.
- ``NullSummariser``         — No-op default (plain text concatenation).

LangGraph checkpointing
- ``PostgresCheckpointSaver`` — ``BaseCheckpointSaver`` backed by PostgreSQL.
- ``make_memory_saver``      — factory that accepts a database URL.

Orchestration
- ``MemoryService``          — single entry point wiring all components together.
"""

from src.memory.checkpoint import PostgresCheckpointSaver, make_memory_saver
from src.memory.models import ConversationSummary, MemoryWindow, UserContext
from src.memory.service import MemoryService
from src.memory.store import ConversationStore
from src.memory.summariser import NullSummariser, Summariser
from src.memory.window import (
    RECENT_WINDOW_SIZE,
    SUMMARISE_THRESHOLD,
    build_window,
    messages_to_summarise,
    should_summarise,
)

__all__ = [
    # models
    "UserContext",
    "ConversationSummary",
    "MemoryWindow",
    # window
    "RECENT_WINDOW_SIZE",
    "SUMMARISE_THRESHOLD",
    "build_window",
    "should_summarise",
    "messages_to_summarise",
    # store
    "ConversationStore",
    # summariser
    "Summariser",
    "NullSummariser",
    # checkpoint
    "PostgresCheckpointSaver",
    "make_memory_saver",
    # service
    "MemoryService",
]
