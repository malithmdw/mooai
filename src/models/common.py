"""Shared primitive types used across domain models.

Centralizing these here keeps validation rules (what counts as a malformed
identifier, how timestamps default) consistent across every model in
`src.models`, rather than duplicating `Field(...)` constraints in each
file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import StringConstraints

EntityId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?$",
    ),
]
"""A validated identifier: 1-128 characters, alphanumeric with internal
`-`/`_`, no leading/trailing whitespace or internal whitespace. Covers
UUIDs, ULIDs, and simple slugs; rejects empty strings, whitespace-only
strings, and IDs containing stray punctuation or spaces.
"""

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""A string that must contain at least one non-whitespace character."""


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC `datetime`.

    Centralized so every model's `default_factory` timestamp uses the same
    clock semantics (timezone-aware, never naive).
    """
    return datetime.now(UTC)
