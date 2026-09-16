"""Standalone enterprise MCP server.

Exposes synthetic enterprise data via the Model Context Protocol (MCP)
over stdio transport.  This server is independently runnable — the main
application does not need to be running for it to work:

    python -m src.mcp.server          # run the server (stdio)
    python src/mcp/server.py          # equivalent

Tools
-----
``find_employee``    Search the employee directory by name or ID.
``get_service``      Look up a service catalog entry by ID.
``get_incident``     Retrieve an incident record by ID.

Resources
---------
``enterprise://employees``   Full employee directory (JSON).
``enterprise://services``    Service catalog (JSON).
``enterprise://incidents``   Incident records (JSON).

Authorization
-------------
Every tool call carries an optional ``caller_role`` argument (defaults to
``"VIEWER"`` — minimum privilege).  The server enforces ``MCP_TOOLS``
permission before returning data.  Denied calls receive a structured error
response rather than an exception.

Tool response shape
-------------------
Every tool returns a JSON-encoded object:

    {
        "success": true | false,
        "data": { ... },           # present when success is true
        "error": {                 # present when success is false
            "code": "NOT_FOUND" | "PERMISSION_DENIED" | "INVALID_INPUT",
            "message": "..."
        },
        "_meta": {
            "authorization": {
                "permission_required": "mcp_tools",
                "caller_role": "ANALYST",
                "granted": true | false
            }
        }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from src.mcp import data as _data
from src.models.enums import Role
from src.security.authorization import (
    AuthorizationPolicy,
    Permission,
    PermissionDeniedError,
)

# mcp stubs are absent; mypy ignores these imports (see pyproject.toml)
try:
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    raise SystemExit("The 'mcp' package is required. Install with: pip install mcp>=1.0")

_logger = logging.getLogger(__name__)

_policy = AuthorizationPolicy()

TOOL_TIMEOUT_SECONDS: float = 10.0

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "enterprise-data-server",
    instructions=(
        "Access Novus Corp synthetic enterprise data — employee directory, "
        "service catalog, and incident records.  All data is fictional.  "
        "Pass caller_role matching the authenticated user's RBAC role."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_role(role_str: str) -> Role:
    """Parse a role string; fall back to VIEWER on invalid input."""
    try:
        return Role(role_str.upper())
    except (ValueError, AttributeError):
        return Role.VIEWER


def _auth_meta(role: Role, *, granted: bool) -> dict[str, Any]:
    return {
        "_meta": {
            "authorization": {
                "permission_required": Permission.MCP_TOOLS.value,
                "caller_role": role.value,
                "granted": granted,
            }
        }
    }


def _ok(data: Any, role: Role) -> str:
    return json.dumps({"success": True, "data": data, **_auth_meta(role, granted=True)})


def _err(code: str, message: str, role: Role) -> str:
    return json.dumps(
        {
            "success": False,
            "error": {"code": code, "message": message},
            **_auth_meta(role, granted=(code != "PERMISSION_DENIED")),
        }
    )


async def _run_with_timeout(fn: Any, *args: Any) -> Any:
    """Execute a synchronous lookup in a thread with a fixed timeout."""
    return await asyncio.wait_for(
        asyncio.to_thread(fn, *args),
        timeout=TOOL_TIMEOUT_SECONDS,
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def find_employee(query: str, caller_role: str = "VIEWER") -> str:
    """Find an employee by name or employee ID.

    Pass query as a partial name (e.g. 'Alice') or an exact ID
    (e.g. 'EMP-001').  Requires MCP_TOOLS permission (ANALYST or
    ADMINISTRATOR).
    """
    role = _resolve_role(caller_role)
    _logger.debug("find_employee query=%r role=%s", query, role)

    try:
        _policy.authorize(role, Permission.MCP_TOOLS)
    except PermissionDeniedError as exc:
        _logger.warning("find_employee denied role=%s", role)
        return _err("PERMISSION_DENIED", str(exc), role)

    if not query.strip():
        return _err("INVALID_INPUT", "query must not be blank", role)

    try:
        result = await _run_with_timeout(_data.lookup_employee, query)
    except TimeoutError:
        _logger.error("find_employee timed out query=%r", query)
        return _err("TIMEOUT", "lookup timed out — retry later", role)

    if result is None:
        return _err("NOT_FOUND", f"No employee matching {query!r}", role)

    _logger.info("find_employee hit employee_id=%s", result["employee_id"])
    return _ok(result, role)


@mcp.tool()
async def get_service(service_id: str, caller_role: str = "VIEWER") -> str:
    """Retrieve a service catalog entry by service ID (e.g. 'SVC-001').

    Returns name, description, SLA, tier, owner team, on-call contact,
    and dependency list.  Requires MCP_TOOLS permission.
    """
    role = _resolve_role(caller_role)
    _logger.debug("get_service service_id=%r role=%s", service_id, role)

    try:
        _policy.authorize(role, Permission.MCP_TOOLS)
    except PermissionDeniedError as exc:
        return _err("PERMISSION_DENIED", str(exc), role)

    if not service_id.strip():
        return _err("INVALID_INPUT", "service_id must not be blank", role)

    try:
        result = await _run_with_timeout(_data.lookup_service, service_id)
    except TimeoutError:
        return _err("TIMEOUT", "lookup timed out — retry later", role)

    if result is None:
        return _err("NOT_FOUND", f"No service {service_id!r}", role)

    _logger.info("get_service hit service_id=%s", result["service_id"])
    return _ok(result, role)


@mcp.tool()
async def get_incident(incident_id: str, caller_role: str = "VIEWER") -> str:
    """Retrieve an incident record by incident ID (e.g. 'INC-001').

    Returns severity, status, root cause, affected service, assignee,
    timeline, and description.  Requires MCP_TOOLS permission.
    """
    role = _resolve_role(caller_role)
    _logger.debug("get_incident incident_id=%r role=%s", incident_id, role)

    try:
        _policy.authorize(role, Permission.MCP_TOOLS)
    except PermissionDeniedError as exc:
        return _err("PERMISSION_DENIED", str(exc), role)

    if not incident_id.strip():
        return _err("INVALID_INPUT", "incident_id must not be blank", role)

    try:
        result = await _run_with_timeout(_data.lookup_incident, incident_id)
    except TimeoutError:
        return _err("TIMEOUT", "lookup timed out — retry later", role)

    if result is None:
        return _err("NOT_FOUND", f"No incident {incident_id!r}", role)

    _logger.info("get_incident hit incident_id=%s", result["incident_id"])
    return _ok(result, role)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("enterprise://employees")
def employee_directory() -> str:
    """Full employee directory — all 12 synthetic employees as JSON."""
    return json.dumps(
        {
            "resource": "enterprise://employees",
            "count": _data.employee_count(),
            "employees": _data.list_employees(),
        }
    )


@mcp.resource("enterprise://services")
def service_catalog() -> str:
    """Service catalog — all 8 synthetic services as JSON."""
    return json.dumps(
        {
            "resource": "enterprise://services",
            "count": _data.service_count(),
            "services": _data.list_services(),
        }
    )


@mcp.resource("enterprise://incidents")
def incident_records() -> str:
    """Incident records — all 10 synthetic incidents as JSON."""
    return json.dumps(
        {
            "resource": "enterprise://incidents",
            "count": _data.incident_count(),
            "incidents": _data.list_incidents(),
        }
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """Run the server over stdio (default FastMCP transport)."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
