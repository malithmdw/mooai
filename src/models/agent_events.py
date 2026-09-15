"""Observability events emitted as an agent moves through execution stages."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from src.models.common import EntityId, NonEmptyStr, utcnow
from src.models.enums import AgentState


class AgentEvent(BaseModel):
    """One observable step of agent execution, for transparent tracing.

    Intended to be emitted to LangSmith (once observability is
    implemented) and to any other execution-trace consumer — see
    CLAUDE.md project purpose, "transparent, inspectable agent execution".
    """

    model_config = ConfigDict(frozen=True)

    event_id: EntityId
    conversation_id: EntityId
    state: AgentState
    message: NonEmptyStr
    timestamp: datetime = Field(default_factory=utcnow)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
