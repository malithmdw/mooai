"""Outcome of validating untrusted content before it reaches a sink with
side effects — see CLAUDE.md "LLM output must be validated before use".
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.common import utcnow


class ValidationResult(BaseModel):
    """Whether a piece of content passed validation, and why not if it didn't."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    errors: tuple[str, ...] = Field(default_factory=tuple)
    validated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _errors_consistent_with_validity(self) -> ValidationResult:
        if self.is_valid and self.errors:
            raise ValueError("a valid result cannot carry validation errors")
        if not self.is_valid and not self.errors:
            raise ValueError("an invalid result must include at least one error")
        return self
