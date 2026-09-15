"""Hardcoded authentication for the POC.

This is intentionally NOT a real authentication system. It exists so that
downstream RBAC and audit-logging code has a stable interface to build
against. Replacing this module with Keycloak/OAuth is a tracked future
milestone — see README.md "Roadmap". Do not extend this module with real
credential storage, token issuance, or password hashing; that work belongs
to the future identity-provider integration, not to this stub.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthenticatedUser:
    """Represents the caller identity for the duration of a request.

    `roles` drives RBAC decisions made in `src.security.rbac` (to be added
    when authorization enforcement is implemented).
    """

    username: str
    roles: tuple[str, ...]


def get_current_user() -> AuthenticatedUser:
    """Return the hardcoded POC user.

    Placeholder for a FastAPI dependency that will later resolve the caller
    from a validated session/token. No real functionality is implemented
    yet — this establishes the seam, not the behavior.
    """
    raise NotImplementedError(
        "Hardcoded POC authentication is not yet wired up. This is a foundation-only placeholder."
    )
