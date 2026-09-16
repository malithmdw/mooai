"""Tests for `src.security.poc_users`."""

from __future__ import annotations

import pytest

from src.models.enums import Role
from src.security.poc_users import find_poc_user


@pytest.mark.parametrize(
    ("username", "expected_role"),
    [
        ("viewer01", Role.VIEWER),
        ("analyst01", Role.ANALYST),
        ("admin01", Role.ADMINISTRATOR),
    ],
)
def test_each_example_user_has_the_expected_role(username: str, expected_role: Role) -> None:
    user = find_poc_user(username)

    assert user is not None
    assert user.username == username
    assert user.role is expected_role
    assert user.user_id


def test_unknown_username_returns_none() -> None:
    assert find_poc_user("nobody") is None


@pytest.mark.parametrize("username", ["viewer01", "analyst01", "admin01"])
def test_salt_and_hash_are_well_formed_hex(username: str) -> None:
    user = find_poc_user(username)
    assert user is not None

    # 16-byte salt, 32-byte PBKDF2-HMAC-SHA256 digest — see src.security.auth.
    assert len(bytes.fromhex(user.salt_hex)) == 16
    assert len(bytes.fromhex(user.password_hash_hex)) == 32
