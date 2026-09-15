"""Chat request/response payloads and the conversation they belong to."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.common import EntityId, NonEmptyStr, utcnow
from src.models.enums import MessageRole
from src.models.evidence import Citation, Evidence


class Message(BaseModel):
    """One turn in a `Conversation`."""

    model_config = ConfigDict(frozen=True)

    message_id: EntityId
    conversation_id: EntityId
    role: MessageRole
    content: NonEmptyStr
    created_at: datetime = Field(default_factory=utcnow)


class Conversation(BaseModel):
    """An ordered thread of messages belonging to one user.

    Immutable: appending a message means constructing a new `Conversation`
    (e.g. via `conversation.model_copy(update={...})`) rather than mutating
    one in place, so every snapshot stays an independent, auditable record.
    """

    model_config = ConfigDict(frozen=True)

    conversation_id: EntityId
    user_id: EntityId
    title: str | None = None
    messages: tuple[Message, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=utcnow)


class ChatRequest(BaseModel):
    """An inbound chat request from a user."""

    model_config = ConfigDict(frozen=True)

    request_id: EntityId
    conversation_id: EntityId | None = None
    user_id: EntityId
    message: NonEmptyStr
    requested_at: datetime = Field(default_factory=utcnow)


class ChatResponse(BaseModel):
    """An assistant reply, with the evidence and citations backing it.

    Every `Citation.evidence_id` must reference an `Evidence` entry present
    in this same response — see CLAUDE.md "invalid citation references".
    """

    model_config = ConfigDict(frozen=True)

    response_id: EntityId
    conversation_id: EntityId
    message: NonEmptyStr
    evidence: tuple[Evidence, ...] = Field(default_factory=tuple)
    citations: tuple[Citation, ...] = Field(default_factory=tuple)
    generated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _citations_reference_known_evidence(self) -> ChatResponse:
        known_evidence_ids = {evidence.evidence_id for evidence in self.evidence}
        unknown = sorted(
            {
                citation.evidence_id
                for citation in self.citations
                if citation.evidence_id not in known_evidence_ids
            }
        )
        if unknown:
            raise ValueError(
                f"citation(s) reference evidence not included in this response: {unknown}"
            )
        return self
