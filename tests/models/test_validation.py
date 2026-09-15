"""Tests for `src.models.validation.ValidationResult`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.validation import ValidationResult


def test_valid_result_with_no_errors() -> None:
    result = ValidationResult(is_valid=True)
    assert result.errors == ()


def test_invalid_result_with_errors() -> None:
    result = ValidationResult(is_valid=False, errors=("missing citation",))
    assert result.errors == ("missing citation",)


def test_rejects_valid_result_carrying_errors() -> None:
    with pytest.raises(ValidationError, match="cannot carry validation errors"):
        ValidationResult(is_valid=True, errors=("unexpected",))


def test_rejects_invalid_result_without_errors() -> None:
    with pytest.raises(ValidationError, match="must include at least one error"):
        ValidationResult(is_valid=False)
