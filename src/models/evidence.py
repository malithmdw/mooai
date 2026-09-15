"""Supporting evidence and citation markers backing a `ChatResponse`."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.models.common import EntityId, NonEmptyStr


class Evidence(BaseModel):
    """An excerpt from a source document supporting part of an answer."""

    model_config = ConfigDict(frozen=True)

    evidence_id: EntityId
    document_id: EntityId
    excerpt: NonEmptyStr
    relevance_score: float = Field(ge=0.0, le=1.0)


class Citation(BaseModel):
    """An in-text reference from a `ChatResponse` to a piece of `Evidence`.

    `reference_number` is the visible marker (the `1` in `[1]`);
    `evidence_id` is cross-validated by `ChatResponse` to point at evidence
    actually included in that same response — see CLAUDE.md "invalid
    citation references".
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: EntityId
    reference_number: int = Field(ge=1)
