"""In-process MCP client adapter for the enterprise data server.

``EnterpriseDataClient`` exposes the same data as the standalone MCP server
(``src.mcp.server``) but without subprocess overhead — it calls the shared
data layer (``src.mcp.data``) directly in the same Python process.

This is the component the rest of the application should import when it
needs to query the enterprise data.  The MCP server in ``server.py`` is the
externally-runnable transport wrapper over the same data; both use identical
business logic.

Design notes
------------
- Every method checks ``Permission.MCP_TOOLS`` via ``AuthorizationPolicy``
  before touching data.  ``PermissionDeniedError`` is deliberately not caught
  here — callers must handle it, matching the same pattern every other tool
  in ``src.tools`` follows.
- Data lookups run inside ``asyncio.to_thread`` wrapped by
  ``asyncio.wait_for``.  The operations are entirely in-memory so the timeout
  is a safety net rather than an expected trigger in normal usage.
- ``MCPClientError`` carries a machine-readable ``code`` alongside the
  human-readable message so the calling agent can route errors without
  string-matching.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from src.core.logging import get_logger, log_debug, log_info, log_warning
from src.mcp import data as _data
from src.models.enums import Role
from src.security.authorization import AuthorizationPolicy, Permission

_logger = get_logger(__name__)

_T = TypeVar("_T")

DEFAULT_TIMEOUT: float = 10.0

_policy = AuthorizationPolicy()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class MCPClientError(Exception):
    """Raised when a data lookup fails for a business-logic reason.

    ``code`` is one of ``"NOT_FOUND"``, ``"INVALID_INPUT"``.
    ``PermissionDeniedError`` is raised directly for RBAC failures (not
    wrapped here) so callers can distinguish the two cleanly.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    def __repr__(self) -> str:
        return f"MCPClientError(code={self.code!r}, message={self.message!r})"


# ---------------------------------------------------------------------------
# Client adapter
# ---------------------------------------------------------------------------


class EnterpriseDataClient:
    """Async client for the enterprise data server, usable from agent nodes.

    Build one instance at startup and reuse it across requests — it is
    stateless and safe for concurrent use.
    """

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run(self, fn: Callable[[], _T]) -> _T:
        """Execute a synchronous data function in a thread with a timeout."""
        return await asyncio.wait_for(asyncio.to_thread(fn), timeout=self._timeout)

    def _check(self, role: Role) -> None:
        """Raise ``PermissionDeniedError`` if the role lacks MCP_TOOLS."""
        _policy.authorize(role, Permission.MCP_TOOLS)

    # ------------------------------------------------------------------
    # Tool-equivalent methods
    # ------------------------------------------------------------------

    async def find_employee(self, query: str, *, role: Role) -> dict[str, Any]:
        """Find an employee by name or ID.

        Parameters
        ----------
        query:
            Employee ID (e.g. ``"EMP-001"``) or partial name
            (case-insensitive).
        role:
            Authenticated caller's RBAC role.

        Raises
        ------
        PermissionDeniedError
            Role lacks ``MCP_TOOLS``.
        MCPClientError(code="INVALID_INPUT")
            ``query`` is blank.
        MCPClientError(code="NOT_FOUND")
            No matching employee.
        """
        self._check(role)
        if not query.strip():
            raise MCPClientError("INVALID_INPUT", "query must not be blank")

        log_debug(
            _logger,
            "mcp_client.find_employee",
            "Looking up employee",
            query=query,
            role=str(role),
        )

        result = await self._run(lambda: _data.lookup_employee(query))
        if result is None:
            log_warning(
                _logger,
                "mcp_client.find_employee.not_found",
                "Employee not found",
                query=query,
            )
            raise MCPClientError("NOT_FOUND", f"No employee matching {query!r}")

        log_info(
            _logger,
            "mcp_client.find_employee.ok",
            "Employee found",
            employee_id=result["employee_id"],
            role=str(role),
        )
        return result

    async def get_service(self, service_id: str, *, role: Role) -> dict[str, Any]:
        """Fetch a service catalog entry by ID (e.g. ``"SVC-001"``).

        Raises
        ------
        PermissionDeniedError
            Role lacks ``MCP_TOOLS``.
        MCPClientError(code="INVALID_INPUT")
            ``service_id`` is blank.
        MCPClientError(code="NOT_FOUND")
            No service with that ID.
        """
        self._check(role)
        if not service_id.strip():
            raise MCPClientError("INVALID_INPUT", "service_id must not be blank")

        log_debug(
            _logger,
            "mcp_client.get_service",
            "Looking up service",
            service_id=service_id,
            role=str(role),
        )

        result = await self._run(lambda: _data.lookup_service(service_id))
        if result is None:
            raise MCPClientError("NOT_FOUND", f"No service {service_id!r}")

        log_info(
            _logger,
            "mcp_client.get_service.ok",
            "Service found",
            service_id=result["service_id"],
            role=str(role),
        )
        return result

    async def get_incident(self, incident_id: str, *, role: Role) -> dict[str, Any]:
        """Fetch an incident record by ID (e.g. ``"INC-001"``).

        Raises
        ------
        PermissionDeniedError
            Role lacks ``MCP_TOOLS``.
        MCPClientError(code="INVALID_INPUT")
            ``incident_id`` is blank.
        MCPClientError(code="NOT_FOUND")
            No incident with that ID.
        """
        self._check(role)
        if not incident_id.strip():
            raise MCPClientError("INVALID_INPUT", "incident_id must not be blank")

        log_debug(
            _logger,
            "mcp_client.get_incident",
            "Looking up incident",
            incident_id=incident_id,
            role=str(role),
        )

        result = await self._run(lambda: _data.lookup_incident(incident_id))
        if result is None:
            raise MCPClientError("NOT_FOUND", f"No incident {incident_id!r}")

        log_info(
            _logger,
            "mcp_client.get_incident.ok",
            "Incident found",
            incident_id=result["incident_id"],
            role=str(role),
        )
        return result

    # ------------------------------------------------------------------
    # Resource-equivalent methods
    # ------------------------------------------------------------------

    async def list_employees(self, *, role: Role) -> list[dict[str, Any]]:
        """Return the full employee directory."""
        self._check(role)
        return await self._run(_data.list_employees)

    async def list_services(self, *, role: Role) -> list[dict[str, Any]]:
        """Return the full service catalog."""
        self._check(role)
        return await self._run(_data.list_services)

    async def list_incidents(self, *, role: Role) -> list[dict[str, Any]]:
        """Return all incident records."""
        self._check(role)
        return await self._run(_data.list_incidents)
