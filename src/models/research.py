"""Multi-step research task tracking and its final synthesized result."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.models.common import EntityId, NonEmptyStr, utcnow
from src.models.enums import ResearchStatus
from src.models.evidence import Evidence


class ResearchTask(BaseModel):
    """One unit of multi-step research work, tracked across its lifecycle.

    Unlike most models in this package, `ResearchTask` is intentionally
    mutable: its `status` changes as research proceeds from `PENDING`
    through to `COMPLETED`/`FAILED`. `validate_assignment=True` keeps every
    mutation as strongly typed as construction.
    """

    model_config = ConfigDict(validate_assignment=True)

    task_id: EntityId
    conversation_id: EntityId
    objective: NonEmptyStr
    status: ResearchStatus = ResearchStatus.PENDING
    created_at: datetime = Field(default_factory=utcnow)


class ResearchResult(BaseModel):
    """The final, synthesized outcome of a completed `ResearchTask`."""

    model_config = ConfigDict(frozen=True)

    task_id: EntityId
    summary: NonEmptyStr
    evidence: tuple[Evidence, ...] = Field(default_factory=tuple)
    completed_at: datetime = Field(default_factory=utcnow)
