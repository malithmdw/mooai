"""Tests for `src.core.redaction.sanitize_fields`.

Also exercised indirectly via `src.core.logging` (as `_sanitize_extra`) and
`src.observability.tracing` — this module is the single source of truth
for what "never log/trace secrets or document content" means in practice.
"""

from __future__ import annotations

import pytest

from src.core.redaction import sanitize_fields


@pytest.mark.parametrize(
    "key",
    ["api_key", "API_KEY", "password", "authorization", "access_token", "client_secret"],
)
def test_redacts_secret_like_keys(key: str) -> None:
    sanitized = sanitize_fields({key: "super-secret-value"})
    assert sanitized[key] == "***REDACTED***"


def test_truncates_long_document_content() -> None:
    long_text = "x" * 500
    sanitized = sanitize_fields({"document_content": long_text})
    assert len(sanitized["document_content"]) < 500  # type: ignore[arg-type]
    assert "truncated" in sanitized["document_content"]  # type: ignore[operator]


def test_leaves_short_content_untouched() -> None:
    sanitized = sanitize_fields({"excerpt": "short"})
    assert sanitized["excerpt"] == "short"


def test_passes_through_ordinary_fields() -> None:
    sanitized = sanitize_fields({"document_id": "doc-1", "latency_ms": 42})
    assert sanitized == {"document_id": "doc-1", "latency_ms": 42}


def test_non_string_values_with_secret_keys_are_still_redacted() -> None:
    sanitized = sanitize_fields({"api_key": None})
    assert sanitized["api_key"] == "***REDACTED***"
