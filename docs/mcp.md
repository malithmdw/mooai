# MCP integration — enterprise data server

## What is MCP?

The **Model Context Protocol** (MCP) is an open standard that lets AI
applications (hosts) connect to external data sources and tools through a
uniform interface.  An MCP *server* exposes *tools* (callable operations) and
*resources* (readable data).  An MCP *client* (typically an AI host like Claude
Desktop, or the application itself) connects to one or more servers and
exposes their capabilities to the model.

```
┌─────────────────────────────────┐
│  AI Host / MCP Client           │
│  (Claude, application agent)    │
│                                 │
│  "Call find_employee('Alice')"  │
└───────────────┬─────────────────┘
                │  MCP protocol (stdio / HTTP)
                ▼
┌─────────────────────────────────┐
│  MCP Server                     │
│  src/mcp/server.py              │
│                                 │
│  Tools:     find_employee       │
│             get_service         │
│             get_incident        │
│                                 │
│  Resources: enterprise://       │
│             employees           │
│             services            │
│             incidents           │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Data layer                     │
│  src/mcp/data.py                │
│  (in-memory, synthetic only)    │
└─────────────────────────────────┘
```

---

## Components

### `src/mcp/data.py` — shared data layer

All synthetic enterprise data lives here.  **No real people, services, or
incidents are stored.**  The module provides:

| Function | Returns |
|---|---|
| `lookup_employee(query)` | `dict \| None` — match by ID or partial name |
| `lookup_service(service_id)` | `dict \| None` — match by ID |
| `lookup_incident(incident_id)` | `dict \| None` — match by ID |
| `list_employees()` | `list[dict]` — all employees, sorted by ID |
| `list_services()` | `list[dict]` — all services, sorted by ID |
| `list_incidents()` | `list[dict]` — all incidents, sorted by ID |

Both the server and the client adapter call these functions directly so
business logic is never duplicated.

### `src/mcp/server.py` — standalone FastMCP server

The server is built with **FastMCP** (the high-level Python MCP SDK API).  It
is independently runnable:

```bash
python -m src.mcp.server     # stdio transport (default)
python src/mcp/server.py     # equivalent
```

**Use case:** connecting external MCP hosts such as Claude Desktop or a
separate agent process to the enterprise data.  The server is *not* required
for the main application to function — the client adapter (below) provides
equivalent access in-process.

### `src/mcp/client.py` — in-process client adapter

`EnterpriseDataClient` calls `data.py` functions directly in the same Python
process.  It is the component **agent nodes should import** when they need
enterprise data.

```python
from src.mcp import EnterpriseDataClient
from src.models.enums import Role

client = EnterpriseDataClient()

# In an async agent node:
employee = await client.find_employee("Alice Chen", role=user_role)
service  = await client.get_service("SVC-001", role=user_role)
incident = await client.get_incident("INC-001", role=user_role)
```

The adapter offers the same interface as a real MCP client would, without the
subprocess/transport overhead.  In a future production build, this adapter
could be replaced by a real MCP client that communicates with the server over
HTTP/SSE.

---

## Data sets

### Employee directory (12 records)

| ID | Name | Title | Department |
|---|---|---|---|
| EMP-001 | Alice Chen | Senior Software Engineer | Platform Engineering |
| EMP-002 | Bob Martinez | Product Manager | Digital Banking |
| EMP-003 | Carol Smith | Data Scientist | Analytics & Data |
| EMP-004 | David Johnson | DevOps Engineer | Infrastructure |
| EMP-005 | Emma Williams | Security Engineer | Cybersecurity |
| EMP-006 | Frank Brown | Business Analyst | Lending & Credit |
| EMP-007 | Grace Davis | QA Lead | Quality Assurance |
| EMP-008 | Henry Wilson | Enterprise Architect | Architecture |
| EMP-009 | Isabella Taylor | Chief Compliance Officer | Risk & Compliance |
| EMP-010 | James Anderson | Engineering Tech Lead | Core Banking |
| EMP-011 | Karen Lee | UI/UX Designer | Product |
| EMP-012 | Liam Nguyen | Database Administrator | Infrastructure |

### Service catalog (8 records)

| ID | Name | Tier | SLA |
|---|---|---|---|
| SVC-001 | Core Banking API | Tier 1 | 99.95 % |
| SVC-002 | Payment Gateway | Tier 1 | 99.99 % |
| SVC-003 | Customer Identity Service | Tier 1 | 99.99 % |
| SVC-004 | Loan Origination System | Tier 2 | 99.9 % |
| SVC-005 | Mobile Banking Platform | Tier 2 | 99.9 % |
| SVC-006 | Enterprise Data Warehouse | Tier 2 | 99.5 % |
| SVC-007 | Fraud Detection Service | Tier 1 | 99.95 % |
| SVC-008 | Notification Service | Tier 3 | 99.5 % |

### Incident records (10 records)

Severities P1–P4; statuses `resolved`, `open`, `investigating`.

---

## Authorization

### Server (standalone)

Every tool call accepts an optional `caller_role` argument
(`"VIEWER"`, `"ENGINEER"`, `"ANALYST"`, `"ADMINISTRATOR"`).  The server
resolves this to a `Role` enum, then calls `AuthorizationPolicy.authorize`
before touching any data.

| Permission required | Allowed roles |
|---|---|
| `MCP_TOOLS` | ANALYST, ADMINISTRATOR |

Denied calls receive a structured JSON error response (`"PERMISSION_DENIED"`)
rather than an exception, so MCP hosts can handle the error gracefully.

### Client adapter

`EnterpriseDataClient` accepts the caller's `Role` on every method.  Denied
calls raise `PermissionDeniedError` (the same exception raised throughout the
rest of the application), so the agent pipeline handles them uniformly.

In both cases the LLM's stated intent or a tool's own judgment is **never**
the authority — `AuthorizationPolicy.authorize` is the only thing that grants
access (see `docs/authorization.md`).

---

## Tool response format (server)

```json
{
    "success": true,
    "data": { "employee_id": "EMP-001", "name": "Alice Chen", "..." : "..." },
    "_meta": {
        "authorization": {
            "permission_required": "mcp_tools",
            "caller_role": "ANALYST",
            "granted": true
        }
    }
}
```

On error:

```json
{
    "success": false,
    "error": {
        "code": "NOT_FOUND",
        "message": "No employee matching 'xyz'"
    },
    "_meta": {
        "authorization": {
            "permission_required": "mcp_tools",
            "caller_role": "ANALYST",
            "granted": true
        }
    }
}
```

Error codes: `NOT_FOUND`, `INVALID_INPUT`, `PERMISSION_DENIED`, `TIMEOUT`.

---

## Timeout handling

Both the server and the client adapter run data lookups inside
`asyncio.to_thread` wrapped by `asyncio.wait_for`.  The default timeout is
**10 seconds**.  Because all lookups are in-memory, a timeout in the current
implementation indicates a system-level problem (e.g. thread pool starvation)
rather than a slow query.

---

## Connecting Claude Desktop (optional)

Add the server to `claude_desktop_config.json`:

```json
{
    "mcpServers": {
        "enterprise-data": {
            "command": "python",
            "args": ["-m", "src.mcp.server"],
            "cwd": "/path/to/mooai"
        }
    }
}
```

Claude Desktop will start the server automatically and present its tools to
the model.

---

## File inventory

| Path | Purpose |
|---|---|
| `src/mcp/data.py` | Synthetic data store + lookup functions |
| `src/mcp/server.py` | Standalone FastMCP server (independently runnable) |
| `src/mcp/client.py` | In-process adapter for use by agent nodes |
| `src/mcp/__init__.py` | Public exports (`EnterpriseDataClient`, `MCPClientError`) |
| `tests/mcp/test_data.py` | Data layer + integrity tests |
| `tests/mcp/test_client.py` | Client adapter RBAC + behaviour tests |
| `docs/mcp.md` | This document |
