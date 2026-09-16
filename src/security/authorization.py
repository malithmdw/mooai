"""Centralized RBAC authorization policy.

This is the **only** place in the codebase that decides whether a role
may perform an action. It is a pure, deterministic mapping from `Role` to
the `Permission`s that role holds — no model call, no prompt, no agent
state, no network I/O. See CLAUDE.md security principles:

- "Agents must never bypass application authorization" — an agent's plan
  or a tool's own judgment is never a substitute for calling
  `AuthorizationPolicy.authorize`.
- "The LLM must NEVER be treated as the authority for permissions" — this
  module has no dependency on `anthropic`, `openai`, `langgraph`, or any
  other LLM/agent library, by construction (see the import list below).
  Every tool must eventually call this policy before executing (see
  `docs/authorization.md`); none exist yet to wire it into (`src.tools`
  is still an empty seam), so this ships as the policy engine + its tests
  ahead of that integration, per CLAUDE.md "Foundation before features".

`Role` already exists in `src.models.enums` (shared domain vocabulary,
used well beyond authorization); it is reused here, not redefined.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from functools import lru_cache

from src.models.enums import Role


class Permission(StrEnum):
    """A discrete, checkable capability an authenticated caller may hold."""

    CHAT = "chat"
    KNOWLEDGE_SEARCH = "knowledge_search"
    ANALYTICS = "analytics"
    MCP_TOOLS = "mcp_tools"
    ADMIN_OPERATIONS = "admin_operations"


class AuthorizationError(Exception):
    """Base class for every error raised by this module."""


class PermissionDeniedError(AuthorizationError):
    """Raised when `role` does not hold `permission`.

    Carries both fields so a caller (eventually the API layer, once real
    tool-calling exists) can build a specific, auditable error response
    without re-deriving what was denied.
    """

    def __init__(self, *, role: Role, permission: Permission) -> None:
        self.role = role
        self.permission = permission
        super().__init__(f"role {role.value!r} does not have permission {permission.value!r}")


_DEFAULT_ROLE_PERMISSIONS: Mapping[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset({Permission.CHAT, Permission.KNOWLEDGE_SEARCH}),
    Role.ANALYST: frozenset(
        {
            Permission.CHAT,
            Permission.KNOWLEDGE_SEARCH,
            Permission.ANALYTICS,
            Permission.MCP_TOOLS,
        }
    ),
    # "all tools" for ADMINISTRATOR: every `Permission` that exists, not a
    # hand-maintained list that could silently drift as `Permission` grows.
    Role.ADMINISTRATOR: frozenset(Permission),
}


class AuthorizationPolicy:
    """The centralized role → permission mapping and its one real check.

    Stateless and side-effect-free: constructing one just captures a
    mapping (the hardcoded default, unless a caller supplies their own —
    tests use this to exercise the generic lookup logic independently of
    the specific default mapping above).
    """

    def __init__(
        self, role_permissions: Mapping[Role, frozenset[Permission]] | None = None
    ) -> None:
        self._role_permissions: Mapping[Role, frozenset[Permission]] = (
            dict(role_permissions)
            if role_permissions is not None
            else dict(_DEFAULT_ROLE_PERMISSIONS)
        )

    def permissions_for(self, role: Role) -> frozenset[Permission]:
        """Every `Permission` granted to `role` (empty if the role is undefined)."""
        return self._role_permissions.get(role, frozenset())

    def is_allowed(self, role: Role, permission: Permission) -> bool:
        """Pure boolean check: does `role` hold `permission`?

        This — not an LLM's stated intent, not an agent's plan — is the
        authority for permission decisions in this system.
        """
        return permission in self.permissions_for(role)

    def authorize(self, role: Role, permission: Permission) -> None:
        """Raise `PermissionDeniedError` if `role` lacks `permission`.

        This is the call every tool must make before executing — see
        module docstring.
        """
        if not self.is_allowed(role, permission):
            raise PermissionDeniedError(role=role, permission=permission)


@lru_cache
def get_authorization_policy() -> AuthorizationPolicy:
    """Return the process-wide `AuthorizationPolicy` singleton.

    Mirrors `src.core.config.get_settings`'s caching pattern — construct
    `AuthorizationPolicy(...)` directly in tests to exercise a specific
    mapping instead.
    """
    return AuthorizationPolicy()
