"""Hardcoded POC user directory.

**This is not a real user store.** See the `src.security.auth` module
docstring and `docs/authentication.md` for what this is (and explicitly
is not). Passwords are never held in plaintext here — only a salted
PBKDF2-HMAC-SHA256 hash per user, computed once via
`src.security.auth.hash_password`. The corresponding plaintext example
passwords are documented in `docs/authentication.md`, not in source, so
they can be tested/used without ever appearing in application logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.enums import Role


@dataclass(frozen=True)
class PocUserRecord:
    """One row of the hardcoded POC user directory."""

    user_id: str
    username: str
    role: Role
    salt_hex: str
    password_hash_hex: str


_POC_USERS: tuple[PocUserRecord, ...] = (
    PocUserRecord(
        user_id="poc-user-viewer01",
        username="viewer01",
        role=Role.VIEWER,
        salt_hex="a83229b7267819da98f0e6934913c91d",
        password_hash_hex="3f23a973460ee2d85660c661d1f35389d2659c878de82a0591e05ffbb72b7b07",
    ),
    PocUserRecord(
        user_id="poc-user-analyst01",
        username="analyst01",
        role=Role.ANALYST,
        salt_hex="21bf65062abc6c770edbbc1a546b0123",
        password_hash_hex="3321510f7c2fad6da399eea1dfc56c03785c3cd749938ec52e90bce154006e2f",
    ),
    PocUserRecord(
        user_id="poc-user-admin01",
        username="admin01",
        role=Role.ADMINISTRATOR,
        salt_hex="86a150ceb31b7d4475fa9bd34d248788",
        password_hash_hex="da6e869f0ce84c3b8dc8af1f483df90543744df3949e14780fe55d4d645e78e3",
    ),
)

_POC_USERS_BY_USERNAME: dict[str, PocUserRecord] = {user.username: user for user in _POC_USERS}


def find_poc_user(username: str) -> PocUserRecord | None:
    """Look up a hardcoded POC user by username, or `None` if unknown."""
    return _POC_USERS_BY_USERNAME.get(username)
