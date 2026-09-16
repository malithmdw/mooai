"""Hardcoded POC authentication.

**POC-ONLY — see `docs/authentication.md`.** This is intentionally NOT a
real authentication system: no session management, no token issuance, no
real identity provider, no credential rotation, no lockout policy, no
MFA. It exists so that downstream RBAC and audit-logging code has a
stable interface (`AuthenticatedUser`, `get_current_user`) to build
against, backed by three fixed example users
(`src.security.poc_users`). Replacing this module with Keycloak/OAuth is
a tracked future milestone — see README.md "Roadmap"; that future
dependency should be a drop-in replacement for `get_current_user` with
the same `AuthenticatedUser` shape, so callers don't need to change.

Passwords are never compared in plaintext: each hardcoded user has a
salted PBKDF2-HMAC-SHA256 hash (see `src.security.poc_users`), verified
here via `hash_password`/`_verify_password`. That said, a fixed salt baked
into source control and no real credential lifecycle would never be
acceptable outside a disposable proof of concept — this hashing exists to
demonstrate the right *pattern*, not to claim production-grade security.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from src.core.logging import get_logger, log_info, log_warning
from src.models.enums import Role
from src.security.poc_users import PocUserRecord, find_poc_user

logger = get_logger(__name__)

# PBKDF2 parameters used for every hash in `src.security.poc_users`. 260_000
# iterations of SHA-256 matches common current guidance (e.g. Django's
# default) for this algorithm.
_HASH_NAME = "sha256"
_ITERATIONS = 260_000
_DKLEN = 32

_INVALID_CREDENTIALS_DETAIL = "Invalid username or password."

_basic_auth = HTTPBasic(auto_error=False)
_BasicAuthDep = Annotated[HTTPBasicCredentials | None, Depends(_basic_auth)]


@dataclass(frozen=True)
class AuthenticatedUser:
    """The caller identity resolved for the duration of a request.

    `role` drives RBAC decisions made elsewhere in `src.security` — see
    CLAUDE.md "Agents must never bypass application authorization". An
    agent's plan or a tool's own judgment is never a substitute for
    checking this.
    """

    user_id: str
    username: str
    role: Role


def hash_password(plaintext_password: str, *, salt_hex: str) -> str:
    """Derive the PBKDF2-HMAC-SHA256 hash used to populate `poc_users.py`.

    Not called at request-handling time — this is the function used
    (once, offline) to generate the hashes hardcoded in
    `src.security.poc_users`, so those hashes are reproducible from a
    documented algorithm rather than opaque magic strings. `_verify_password`
    calls it again at request time to compute the candidate hash to compare.
    """
    salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac(
        _HASH_NAME, plaintext_password.encode("utf-8"), salt, _ITERATIONS, _DKLEN
    )
    return digest.hex()


def _verify_password(plaintext_password: str, user: PocUserRecord) -> bool:
    """Constant-time comparison of a plaintext password against its hash."""
    candidate_hash = hash_password(plaintext_password, salt_hex=user.salt_hex)
    return hmac.compare_digest(candidate_hash, user.password_hash_hex)


def authenticate(username: str, password: str) -> AuthenticatedUser:
    """Verify a username/password pair against the hardcoded POC directory.

    Raises `HTTPException(401)` with an identical, generic message for
    both an unknown username and a wrong password — so a caller can't use
    the error response to enumerate which usernames exist.
    """
    user = find_poc_user(username)
    if user is None or not _verify_password(password, user):
        log_warning(logger, "auth.login_failed", "authentication failed", username=username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID_CREDENTIALS_DETAIL,
            headers={"WWW-Authenticate": "Basic"},
        )

    log_info(
        logger,
        "auth.login_succeeded",
        "authentication succeeded",
        user_id=user.user_id,
        username=user.username,
    )
    return AuthenticatedUser(user_id=user.user_id, username=user.username, role=user.role)


def get_current_user(credentials: _BasicAuthDep) -> AuthenticatedUser:
    """FastAPI authentication dependency: resolve the caller from HTTP Basic credentials.

    Routes depend on this (directly, or via `src.api.dependencies.CurrentUserDep`)
    to require authentication — see CLAUDE.md "clean dependency injection".
    A future Keycloak/OAuth dependency is meant to be a drop-in replacement
    with the same signature and `AuthenticatedUser` return type.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return authenticate(credentials.username, credentials.password)
