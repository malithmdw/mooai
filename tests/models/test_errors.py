"""Tests for `src.models.errors.ErrorResponse`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.errors import ErrorResponse


def test_valid_error_response() -> None:
    error = ErrorResponse(error_code="NOT_FOUND", message="Document not found.")
    assert error.request_id is None


@pytest.mark.parametrize("bad_code", ["not_found", "notFound", "404", ""])
def test_rejects_malformed_error_code(bad_code: str) -> None:
    with pytest.raises(ValidationError, match="error_code"):
        ErrorResponse(error_code=bad_code, message="Document not found.")


def test_rejects_empty_message() -> None:
    with pytest.raises(ValidationError, match="message"):
        ErrorResponse(error_code="NOT_FOUND", message="   ")
