"""Shared sensitive-field redaction.

Used by both `src.core.logging` and `src.observability.tracing` so "never
log/trace secrets or full document content" (see CLAUDE.md "Logging
requirements" and "Security principles") is enforced identically wherever
structured, caller-supplied fields leave the process — one rule, one
implementation, not two copies that can drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping

_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "token",
    "authorization",
    "credential",
)
_CONTENT_KEY_MARKERS = ("content", "document_text", "excerpt", "body", "raw_text")
_MAX_CONTENT_PREVIEW_CHARS = 200
_REDACTED = "***REDACTED***"


def sanitize_fields(fields: Mapping[str, object]) -> dict[str, object]:
    """Redact secrets and truncate document-like content in `fields`.

    Any key whose name contains a secret marker (`api_key`, `password`,
    `token`, `authorization`, ...) is fully redacted. Any string-valued key
    whose name suggests document content is truncated to a bounded preview
    — never logged/traced in full, per CLAUDE.md "Never log ... complete
    confidential document contents".
    """
    sanitized: dict[str, object] = {}
    for key, value in fields.items():
        lowered = key.lower()
        if any(marker in lowered for marker in _SECRET_KEY_MARKERS):
            sanitized[key] = _REDACTED
        elif (
            any(marker in lowered for marker in _CONTENT_KEY_MARKERS)
            and isinstance(value, str)
            and len(value) > _MAX_CONTENT_PREVIEW_CHARS
        ):
            omitted = len(value) - _MAX_CONTENT_PREVIEW_CHARS
            sanitized[key] = f"{value[:_MAX_CONTENT_PREVIEW_CHARS]}…[{omitted} chars truncated]"
        else:
            sanitized[key] = value
    return sanitized
