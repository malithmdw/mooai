# Authorization (RBAC)

This document explains the centralized role-based access control (RBAC)
policy implemented in `src/security/authorization.py`: the permission
model, why the engine is independent of any LLM, and the rule every
future tool integration must follow.

## The model

Three types, two of which are new here:

- **`Role`** (`src.models.enums.Role`) — already existed as shared domain
  vocabulary (`VIEWER`, `ANALYST`, `ADMINISTRATOR`); reused here, not
  redefined.
- **`Permission`** (`src.security.authorization.Permission`) — a
  discrete, checkable capability: `CHAT`, `KNOWLEDGE_SEARCH`,
  `ANALYTICS`, `MCP_TOOLS`, `ADMIN_OPERATIONS`.
- **`AuthorizationPolicy`** (`src.security.authorization.AuthorizationPolicy`)
  — the policy engine: a `Role` → `frozenset[Permission]` mapping plus
  `is_allowed`/`authorize`, the only functions in the codebase that decide
  whether a role may do something.

## The permission matrix

| Permission | VIEWER | ANALYST | ADMINISTRATOR |
|---|---|---|---|
| `CHAT` | ✅ | ✅ | ✅ |
| `KNOWLEDGE_SEARCH` | ✅ | ✅ | ✅ |
| `ANALYTICS` | ❌ | ✅ | ✅ |
| `MCP_TOOLS` | ❌ | ✅ | ✅ |
| `ADMIN_OPERATIONS` | ❌ | ❌ | ✅ |

`ADMINISTRATOR`'s permission set is computed as *every member of
`Permission`* (`frozenset(Permission)`), not a hand-maintained list — "all
tools" stays true even as `Permission` grows, with no risk of a new
permission being silently forgotten for the administrator role.

## Using it

```python
from src.security.authorization import Permission, get_authorization_policy

policy = get_authorization_policy()

if policy.is_allowed(current_user.role, Permission.ANALYTICS):
    ...

# or, to fail loudly with a typed error instead of branching:
policy.authorize(current_user.role, Permission.MCP_TOOLS)  # raises PermissionDeniedError
```

`current_user` here is the `AuthenticatedUser` resolved by
`src.security.auth.get_current_user` (see `docs/authentication.md`) —
`AuthenticatedUser.role` is exactly the `Role` this policy checks against.

## Why the engine is independent of the LLM

`src/security/authorization.py` imports nothing from `anthropic`,
`openai`, `langgraph`, `langchain`, or any other model/agent library —
`is_allowed`/`authorize` are pure functions over a `dict` literal. No
prompt is constructed, no model is called, and no agent state is read to
answer "is this allowed?". This is deliberate, not incidental — see
CLAUDE.md:

> The LLM must NEVER be treated as the authority for permissions.

An agent asking an LLM "should I call this tool?" and getting back "yes"
is not authorization — it's the LLM's opinion. The only thing that
authorizes a tool call is `AuthorizationPolicy.authorize` returning
without raising, evaluated against the real `Role` of the real
authenticated caller.

## The invariant every tool must follow

> Every tool must eventually call the authorization policy before
> execution.

Concretely: once `src.tools` contains real tool implementations (it is
still an empty seam — see CLAUDE.md "Foundation before features"), each
one must call `AuthorizationPolicy.authorize(role, permission)` (or check
`is_allowed`) **before** doing any work, using the caller's actual
resolved role — never a role or permission an LLM's output claims should
apply, and never skipped because "the agent already decided to call it."
This is the same rule CLAUDE.md states more generally: "Agents must never
bypass application authorization" — an agent's plan or a tool's own
judgment is never a substitute for this check.

`PermissionDeniedError` (raised by `authorize`) is a plain Python
exception, not an HTTP-specific one — `security` has no dependency on
`api` (see CLAUDE.md "Dependency boundaries") and tool calls need not be
HTTP-bound. Translating it into a `403` `ErrorResponse` at the API
boundary (mirroring how `src.api.exception_handlers` already normalizes
other errors) is future work, once a real endpoint actually triggers a
tool call.

## Current scope

This ships the policy engine, its full test matrix (every role/permission
combination in the table above, plus the generic lookup logic exercised
independently of the hardcoded default mapping), and this document. It is
not yet wired into any endpoint or tool — there are no tools to wire it
into yet. `AuthorizationPolicy` also does not yet support
resource-/attribute-level scoping (e.g. "analyst may see analytics only
for their own department"); today's model is role → permission only, matching
what was specified.
