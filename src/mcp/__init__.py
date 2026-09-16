"""MCP integration — enterprise data server and in-process client adapter.

``src.mcp`` contains two components that share the same synthetic data layer
(``src.mcp.data``):

``server``
    A standalone FastMCP server runnable over stdio:
        python -m src.mcp.server
    Exposes ``find_employee``, ``get_service``, ``get_incident`` as MCP tools
    and the three collections as MCP resources.

``client``
    ``EnterpriseDataClient`` — an in-process adapter that calls the data
    layer directly (no subprocess).  This is what the application's agent
    nodes should import.  It enforces the same RBAC policy as the server.

Dependency boundary
-------------------
``src.mcp`` may import from ``src.core``, ``src.models``, and
``src.security``.  The ``src.api`` and ``src.agents`` layers import from
``src.mcp``, never the other way round.
"""

from src.mcp.client import DEFAULT_TIMEOUT, EnterpriseDataClient, MCPClientError

__all__ = [
    "DEFAULT_TIMEOUT",
    "EnterpriseDataClient",
    "MCPClientError",
]
