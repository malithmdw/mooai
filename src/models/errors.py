"""Structured error payload returned to API callers."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.models.common import EntityId, NonEmptyStr, utcnow

ErrorCode = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$")]
"""An upper-snake-case machine-readable error code, e.g. `NOT_FOUND`."""


class ErrorResponse(BaseModel):
    """A structured, machine-parseable error returned to an API caller."""

    model_config = ConfigDict(frozen=True)

    error_code: ErrorCode
    message: NonEmptyStr
    request_id: EntityId | None = None
    occurred_at: datetime = Field(default_factory=utcnow)
