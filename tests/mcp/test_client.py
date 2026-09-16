"""Tests for ``EnterpriseDataClient`` — the in-process MCP client adapter.

Coverage
--------
RBAC (all four roles against each method):
  - VIEWER and ENGINEER are denied (PermissionDeniedError)
  - ANALYST and ADMINISTRATOR are allowed

find_employee:
  - returns employee dict for valid query
  - raises MCPClientError(NOT_FOUND) for unknown query
  - raises MCPClientError(INVALID_INPUT) for blank query
  - result contains expected fields

get_service:
  - returns service dict for valid ID
  - raises MCPClientError(NOT_FOUND) for unknown ID
  - raises MCPClientError(INVALID_INPUT) for blank ID

get_incident:
  - returns incident dict for valid ID
  - raises MCPClientError(NOT_FOUND) for unknown ID
  - raises MCPClientError(INVALID_INPUT) for blank ID

list_* methods:
  - list_employees, list_services, list_incidents return non-empty lists
  - denied for VIEWER

Timeout:
  - TimeoutError propagates when asyncio.to_thread is slow
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from src.mcp.client import EnterpriseDataClient, MCPClientError
from src.mcp.data import employee_count, incident_count, service_count
from src.models.enums import Role
from src.security.authorization import PermissionDeniedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> EnterpriseDataClient:
    return EnterpriseDataClient()


# ---------------------------------------------------------------------------
# RBAC — find_employee
# ---------------------------------------------------------------------------


class TestFindEmployeeRBAC:
    async def test_viewer_denied(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(PermissionDeniedError):
            await client.find_employee("Alice", role=Role.VIEWER)

    async def test_engineer_denied(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(PermissionDeniedError):
            await client.find_employee("Alice", role=Role.ENGINEER)

    async def test_analyst_allowed(self, client: EnterpriseDataClient) -> None:
        result = await client.find_employee("Alice", role=Role.ANALYST)
        assert result["name"] == "Alice Chen"

    async def test_administrator_allowed(self, client: EnterpriseDataClient) -> None:
        result = await client.find_employee("EMP-001", role=Role.ADMINISTRATOR)
        assert result["employee_id"] == "EMP-001"


# ---------------------------------------------------------------------------
# RBAC — get_service
# ---------------------------------------------------------------------------


class TestGetServiceRBAC:
    async def test_viewer_denied(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(PermissionDeniedError):
            await client.get_service("SVC-001", role=Role.VIEWER)

    async def test_analyst_allowed(self, client: EnterpriseDataClient) -> None:
        result = await client.get_service("SVC-001", role=Role.ANALYST)
        assert result["service_id"] == "SVC-001"


# ---------------------------------------------------------------------------
# RBAC — get_incident
# ---------------------------------------------------------------------------


class TestGetIncidentRBAC:
    async def test_viewer_denied(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(PermissionDeniedError):
            await client.get_incident("INC-001", role=Role.VIEWER)

    async def test_analyst_allowed(self, client: EnterpriseDataClient) -> None:
        result = await client.get_incident("INC-001", role=Role.ANALYST)
        assert result["incident_id"] == "INC-001"


# ---------------------------------------------------------------------------
# find_employee behaviour
# ---------------------------------------------------------------------------


class TestFindEmployee:
    async def test_find_by_id(self, client: EnterpriseDataClient) -> None:
        result = await client.find_employee("EMP-001", role=Role.ANALYST)
        assert result["employee_id"] == "EMP-001"
        assert result["name"] == "Alice Chen"

    async def test_find_by_partial_name(self, client: EnterpriseDataClient) -> None:
        result = await client.find_employee("Bob", role=Role.ANALYST)
        assert "Bob" in result["name"]

    async def test_not_found_raises_mcp_error(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(MCPClientError) as exc_info:
            await client.find_employee("EMP-999", role=Role.ANALYST)
        assert exc_info.value.code == "NOT_FOUND"

    async def test_blank_query_raises_invalid_input(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(MCPClientError) as exc_info:
            await client.find_employee("   ", role=Role.ANALYST)
        assert exc_info.value.code == "INVALID_INPUT"

    async def test_result_contains_expected_fields(self, client: EnterpriseDataClient) -> None:
        result = await client.find_employee("EMP-001", role=Role.ANALYST)
        for field in ("employee_id", "name", "title", "department", "email"):
            assert field in result


# ---------------------------------------------------------------------------
# get_service behaviour
# ---------------------------------------------------------------------------


class TestGetService:
    async def test_get_by_id(self, client: EnterpriseDataClient) -> None:
        result = await client.get_service("SVC-001", role=Role.ANALYST)
        assert result["service_id"] == "SVC-001"
        assert result["name"] == "Core Banking API"

    async def test_case_insensitive_id(self, client: EnterpriseDataClient) -> None:
        result = await client.get_service("svc-001", role=Role.ANALYST)
        assert result["service_id"] == "SVC-001"

    async def test_not_found_raises_mcp_error(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(MCPClientError) as exc_info:
            await client.get_service("SVC-999", role=Role.ANALYST)
        assert exc_info.value.code == "NOT_FOUND"

    async def test_blank_id_raises_invalid_input(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(MCPClientError) as exc_info:
            await client.get_service("  ", role=Role.ANALYST)
        assert exc_info.value.code == "INVALID_INPUT"

    async def test_result_has_sla_field(self, client: EnterpriseDataClient) -> None:
        result = await client.get_service("SVC-001", role=Role.ANALYST)
        assert "sla_uptime_pct" in result
        assert isinstance(result["sla_uptime_pct"], float)

    async def test_all_services_retrievable(self, client: EnterpriseDataClient) -> None:
        services = await client.list_services(role=Role.ANALYST)
        for svc in services:
            got = await client.get_service(svc["service_id"], role=Role.ANALYST)
            assert got["service_id"] == svc["service_id"]


# ---------------------------------------------------------------------------
# get_incident behaviour
# ---------------------------------------------------------------------------


class TestGetIncident:
    async def test_get_by_id(self, client: EnterpriseDataClient) -> None:
        result = await client.get_incident("INC-001", role=Role.ANALYST)
        assert result["incident_id"] == "INC-001"

    async def test_case_insensitive_id(self, client: EnterpriseDataClient) -> None:
        result = await client.get_incident("inc-001", role=Role.ANALYST)
        assert result["incident_id"] == "INC-001"

    async def test_not_found_raises_mcp_error(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(MCPClientError) as exc_info:
            await client.get_incident("INC-999", role=Role.ANALYST)
        assert exc_info.value.code == "NOT_FOUND"

    async def test_blank_id_raises_invalid_input(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(MCPClientError) as exc_info:
            await client.get_incident("", role=Role.ANALYST)
        assert exc_info.value.code == "INVALID_INPUT"

    async def test_severity_field_present(self, client: EnterpriseDataClient) -> None:
        result = await client.get_incident("INC-001", role=Role.ANALYST)
        assert result["severity"] in ("P1", "P2", "P3", "P4")

    async def test_all_incidents_retrievable(self, client: EnterpriseDataClient) -> None:
        incidents = await client.list_incidents(role=Role.ANALYST)
        for inc in incidents:
            got = await client.get_incident(inc["incident_id"], role=Role.ANALYST)
            assert got["incident_id"] == inc["incident_id"]


# ---------------------------------------------------------------------------
# List methods
# ---------------------------------------------------------------------------


class TestListMethods:
    async def test_list_employees_returns_all(self, client: EnterpriseDataClient) -> None:
        result = await client.list_employees(role=Role.ANALYST)
        assert len(result) == employee_count()

    async def test_list_services_returns_all(self, client: EnterpriseDataClient) -> None:
        result = await client.list_services(role=Role.ANALYST)
        assert len(result) == service_count()

    async def test_list_incidents_returns_all(self, client: EnterpriseDataClient) -> None:
        result = await client.list_incidents(role=Role.ANALYST)
        assert len(result) == incident_count()

    async def test_list_employees_viewer_denied(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(PermissionDeniedError):
            await client.list_employees(role=Role.VIEWER)

    async def test_list_services_viewer_denied(self, client: EnterpriseDataClient) -> None:
        with pytest.raises(PermissionDeniedError):
            await client.list_services(role=Role.VIEWER)

    async def test_list_incidents_administrator_allowed(
        self, client: EnterpriseDataClient
    ) -> None:
        result = await client.list_incidents(role=Role.ADMINISTRATOR)
        assert len(result) > 0

    async def test_list_employees_sorted_by_id(self, client: EnterpriseDataClient) -> None:
        employees = await client.list_employees(role=Role.ANALYST)
        ids = [e["employee_id"] for e in employees]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_timeout_raises_timeout_error(self) -> None:
        async def _slow(fn: object, *args: object, **kwargs: object) -> None:
            await asyncio.sleep(10)

        client = EnterpriseDataClient(timeout=0.001)
        with patch("asyncio.to_thread", new=_slow):
            with pytest.raises(TimeoutError):
                await client.find_employee("Alice", role=Role.ANALYST)

    async def test_custom_timeout_accepted(self) -> None:
        client = EnterpriseDataClient(timeout=60.0)
        result = await client.find_employee("Alice", role=Role.ANALYST)
        assert result is not None


# ---------------------------------------------------------------------------
# MCPClientError
# ---------------------------------------------------------------------------


class TestMCPClientError:
    def test_code_and_message_accessible(self) -> None:
        err = MCPClientError("NOT_FOUND", "nothing here")
        assert err.code == "NOT_FOUND"
        assert err.message == "nothing here"
        assert str(err) == "nothing here"

    def test_repr(self) -> None:
        err = MCPClientError("INVALID_INPUT", "bad")
        assert "INVALID_INPUT" in repr(err)
        assert "bad" in repr(err)

    def test_is_exception(self) -> None:
        err = MCPClientError("TIMEOUT", "slow")
        assert isinstance(err, Exception)
