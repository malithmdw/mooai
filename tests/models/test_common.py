"""Tests for the shared `EntityId`/`NonEmptyStr` primitive types.

Exercised indirectly through `User`, the simplest model that uses both.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.enums import Role
from src.models.user import User


def _user(**overrides: object) -> dict[str, object]:
    return {
        "user_id": "user-123",
        "username": "jsmith",
        "roles": (Role.ANALYST,),
        **overrides,
    }


class TestEntityIdValidation:
    """Malformed IDs are rejected."""

    @pytest.mark.parametrize(
        "malformed_id",
        [
            "",
            "   ",
            "has spaces",
            "-leading-hyphen",
            "trailing-hyphen-",
            "a" * 129,
        ],
    )
    def test_rejects_malformed_ids(self, malformed_id: str) -> None:
        with pytest.raises(ValidationError, match="user_id"):
            User.model_validate(_user(user_id=malformed_id))

    @pytest.mark.parametrize(
        "valid_id",
        ["a", "user-123", "3fa85f64-5717-4562-b3fc-2c963f66afa6", "user_123"],
    )
    def test_accepts_well_formed_ids(self, valid_id: str) -> None:
        user = User.model_validate(_user(user_id=valid_id))
        assert user.user_id == valid_id


class TestNonEmptyStrValidation:
    """Empty/whitespace-only strings are rejected."""

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_rejects_blank_username(self, blank: str) -> None:
        with pytest.raises(ValidationError, match="username"):
            User.model_validate(_user(username=blank))

    def test_strips_surrounding_whitespace(self) -> None:
        user = User.model_validate(_user(username="  jsmith  "))
        assert user.username == "jsmith"
