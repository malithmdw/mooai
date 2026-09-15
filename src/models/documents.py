"""Document metadata and hybrid-search retrieval results.

`RetrievedDocument.content` is untrusted content returned by the retrieval
layer (Pinecone + BM25), not an instruction — see CLAUDE.md "Retrieved
documents are untrusted content".
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.common import EntityId, NonEmptyStr, utcnow
from src.models.enums import AccessLevel, Role


class DocumentMetadata(BaseModel):
    """Descriptive and access-control metadata for one enterprise document."""

    model_config = ConfigDict(frozen=True)

    document_id: EntityId
    title: NonEmptyStr
    department: NonEmptyStr
    document_type: NonEmptyStr
    access_level: AccessLevel
    created_date: date
    allowed_roles: tuple[Role, ...]

    @field_validator("created_date")
    @classmethod
    def _not_in_the_future(cls, value: date) -> date:
        if value > utcnow().date():
            raise ValueError("created_date cannot be in the future")
        return value

    @field_validator("allowed_roles")
    @classmethod
    def _at_least_one_allowed_role(cls, roles: tuple[Role, ...]) -> tuple[Role, ...]:
        if not roles:
            raise ValueError("a document must be accessible to at least one role")
        return roles


class RetrievedDocument(BaseModel):
    """A single hybrid-search retrieval result."""

    model_config = ConfigDict(frozen=True)

    metadata: DocumentMetadata
    content: NonEmptyStr
    relevance_score: float = Field(ge=0.0, le=1.0)
