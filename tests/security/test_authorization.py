"""Tests for `src.security.authorization`.

Covers the exact permission matrix specified for this RBAC policy:
VIEWER = chat + knowledge_search; ANALYST = VIEWER's permissions +
analytics + MCP tools; ADMINISTRATOR = every permission that exists.
"""

from __future__ import annotations

import pytest

from src.models.enums import Role
from src.security.authorization import (
    AuthorizationPolicy,
    Permission,
    PermissionDeniedError,
    get_authorization_policy,
)

policy = AuthorizationPolicy()


class TestViewer:
    def test_allowed_knowledge_search(self) -> None:
        assert policy.is_allowed(Role.VIEWER, Permission.KNOWLEDGE_SEARCH) is True

    def test_allowed_chat(self) -> None:
        assert policy.is_allowed(Role.VIEWER, Permission.CHAT) is True

    def test_denied_analytics(self) -> None:
        assert policy.is_allowed(Role.VIEWER, Permission.ANALYTICS) is False

    def test_denied_mcp_tools(self) -> None:
        assert policy.is_allowed(Role.VIEWER, Permission.MCP_TOOLS) is False

    def test_denied_admin_operations(self) -> None:
        assert policy.is_allowed(Role.VIEWER, Permission.ADMIN_OPERATIONS) is False

    def test_authorize_raises_for_a_denied_permission(self) -> None:
        with pytest.raises(PermissionDeniedError) as exc_info:
            policy.authorize(Role.VIEWER, Permission.ANALYTICS)

        assert exc_info.value.role is Role.VIEWER
        assert exc_info.value.permission is Permission.ANALYTICS


class TestAnalyst:
    def test_allowed_analytics(self) -> None:
        assert policy.is_allowed(Role.ANALYST, Permission.ANALYTICS) is True

    def test_allowed_mcp_tools(self) -> None:
        assert policy.is_allowed(Role.ANALYST, Permission.MCP_TOOLS) is True

    def test_allowed_chat_and_knowledge_search(self) -> None:
        assert policy.is_allowed(Role.ANALYST, Permission.CHAT) is True
        assert policy.is_allowed(Role.ANALYST, Permission.KNOWLEDGE_SEARCH) is True

    def test_denied_admin_only_operation(self) -> None:
        assert policy.is_allowed(Role.ANALYST, Permission.ADMIN_OPERATIONS) is False

    def test_authorize_raises_for_admin_only_operation(self) -> None:
        with pytest.raises(PermissionDeniedError):
            policy.authorize(Role.ANALYST, Permission.ADMIN_OPERATIONS)


class TestAdministrator:
    @pytest.mark.parametrize("permission", list(Permission))
    def test_allowed_everything(self, permission: Permission) -> None:
        assert policy.is_allowed(Role.ADMINISTRATOR, permission) is True

    def test_authorize_never_raises(self) -> None:
        for permission in Permission:
            policy.authorize(Role.ADMINISTRATOR, permission)  # must not raise


class TestPermissionsFor:
    def test_viewer_permission_set_is_exactly_chat_and_search(self) -> None:
        assert policy.permissions_for(Role.VIEWER) == {
            Permission.CHAT,
            Permission.KNOWLEDGE_SEARCH,
        }

    def test_analyst_permission_set_excludes_admin_operations(self) -> None:
        assert Permission.ADMIN_OPERATIONS not in policy.permissions_for(Role.ANALYST)

    def test_administrator_permission_set_is_every_permission(self) -> None:
        assert policy.permissions_for(Role.ADMINISTRATOR) == frozenset(Permission)


class TestCustomPolicy:
    """The lookup logic itself, independent of the hardcoded default mapping."""

    def test_unknown_role_in_a_custom_mapping_has_no_permissions(self) -> None:
        empty_policy = AuthorizationPolicy(role_permissions={})

        assert empty_policy.permissions_for(Role.ADMINISTRATOR) == frozenset()
        assert empty_policy.is_allowed(Role.ADMINISTRATOR, Permission.CHAT) is False

    def test_custom_mapping_is_respected(self) -> None:
        restrictive_policy = AuthorizationPolicy(
            role_permissions={Role.ADMINISTRATOR: frozenset({Permission.CHAT})}
        )

        assert restrictive_policy.is_allowed(Role.ADMINISTRATOR, Permission.CHAT) is True
        assert restrictive_policy.is_allowed(Role.ADMINISTRATOR, Permission.ANALYTICS) is False


def test_get_authorization_policy_returns_a_singleton() -> None:
    assert get_authorization_policy() is get_authorization_policy()
