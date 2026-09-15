"""Tests for `src.models.user.User`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.enums import Role
from src.models.user import User


def test_valid_user_is_immutable() -> None:
    user = User(
        user_id="user-1",
        username="jsmith",
        roles=(Role.VIEWER, Role.ANALYST),
    )

    assert user.roles == (Role.VIEWER, Role.ANALYST)
    assert user.is_active is True
    with pytest.raises(ValidationError):
        user.username = "someone-else"  # type: ignore[misc]


def test_rejects_empty_roles() -> None:
    with pytest.raises(ValidationError, match="at least one role"):
        User(user_id="user-1", username="jsmith", roles=())


def test_rejects_invalid_role_value() -> None:
    with pytest.raises(ValidationError):
        User(user_id="user-1", username="jsmith", roles=("STAFF",))  # type: ignore[arg-type]


@pytest.mark.parametrize("role", list(Role))
def test_accepts_every_defined_role(role: Role) -> None:
    user = User(user_id="user-1", username="jsmith", roles=(role,))
    assert user.roles == (role,)


def test_role_values_are_exactly_the_three_specified() -> None:
    assert {role.value for role in Role} == {"VIEWER", "ANALYST", "ADMINISTRATOR"}
