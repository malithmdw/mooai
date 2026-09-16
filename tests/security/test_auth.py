"""Tests for `src.security.auth`.

The parametrized `test_authenticate_succeeds_for_each_example_user` case
is the load-bearing test here: it round-trips each hardcoded user's known
plaintext password through `hash_password` and compares against the hash
stored in `src.security.poc_users` — if that hash (or its salt) were ever
transcribed incorrectly, this is what would catch it.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

from src.models.enums import Role
from src.security.auth import AuthenticatedUser, authenticate, get_current_user, hash_password
from src.security.poc_users import find_poc_user

_EXAMPLE_CREDENTIALS = [
    ("viewer01", "Viewer01#Poc2026", Role.VIEWER),
    ("analyst01", "Analyst01#Poc2026", Role.ANALYST),
    ("admin01", "Admin01#Poc2026", Role.ADMINISTRATOR),
]


class TestAuthenticate:
    @pytest.mark.parametrize(("username", "password", "role"), _EXAMPLE_CREDENTIALS)
    def test_succeeds_for_each_example_user(self, username: str, password: str, role: Role) -> None:
        user = authenticate(username, password)

        assert user == AuthenticatedUser(
            user_id=find_poc_user(username).user_id,  # type: ignore[union-attr]
            username=username,
            role=role,
        )

    def test_rejects_wrong_password(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            authenticate("admin01", "not-the-right-password")

        assert exc_info.value.status_code == 401

    def test_rejects_unknown_username(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            authenticate("nobody", "irrelevant")

        assert exc_info.value.status_code == 401

    def test_unknown_username_and_wrong_password_give_identical_error(self) -> None:
        """An error message difference would let a caller enumerate usernames."""
        with pytest.raises(HTTPException) as unknown_user_exc:
            authenticate("nobody", "irrelevant")
        with pytest.raises(HTTPException) as wrong_password_exc:
            authenticate("admin01", "not-the-right-password")

        assert unknown_user_exc.value.status_code == wrong_password_exc.value.status_code
        assert unknown_user_exc.value.detail == wrong_password_exc.value.detail

    def test_authenticated_user_has_exactly_user_id_username_role(self) -> None:
        user = authenticate("viewer01", "Viewer01#Poc2026")

        assert vars(user).keys() == {"user_id", "username", "role"}


class TestHashPassword:
    def test_is_deterministic_for_the_same_salt(self) -> None:
        salt_hex = "00" * 16
        assert hash_password("hunter2", salt_hex=salt_hex) == hash_password(
            "hunter2", salt_hex=salt_hex
        )

    def test_differs_for_different_salts(self) -> None:
        first = hash_password("hunter2", salt_hex="00" * 16)
        second = hash_password("hunter2", salt_hex="ff" * 16)

        assert first != second

    def test_differs_for_different_passwords(self) -> None:
        salt_hex = "00" * 16
        assert hash_password("hunter2", salt_hex=salt_hex) != hash_password(
            "hunter3", salt_hex=salt_hex
        )


class TestGetCurrentUser:
    def test_returns_authenticated_user_for_valid_credentials(self) -> None:
        credentials = HTTPBasicCredentials(username="analyst01", password="Analyst01#Poc2026")

        user = get_current_user(credentials)

        assert user.username == "analyst01"
        assert user.role is Role.ANALYST

    def test_raises_401_for_missing_credentials(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(None)

        assert exc_info.value.status_code == 401

    def test_raises_401_for_invalid_credentials(self) -> None:
        credentials = HTTPBasicCredentials(username="analyst01", password="wrong")

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials)

        assert exc_info.value.status_code == 401
