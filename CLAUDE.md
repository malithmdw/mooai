# CLAUDE.md

Guidance for any AI agent (or human) working in this repository. These
rules are binding, not suggestions — code that violates them should be
treated as a bug even if it "works."

## 1. Project purpose

This repository implements an **enterprise-grade AI Assistant POC for a
commercial bank**. The assistant must:

- Search organizational knowledge and answer questions grounded in
  enterprise documents, with supporting evidence and citations.
- Maintain conversation context across turns.
- Perform multi-step research using external tools (via MCP).
- Demonstrate recursive language model (RLM) behavior through LangGraph.
- Enforce role-based access control (RBAC) on every data and tool access.
- Defend against prompt injection and data exfiltration.
- Provide transparent, inspectable agent execution and full LangSmith
  observability.

This is a **bank-facing system**. Correctness, auditability, and
authorization are not optional polish — they are the product. When in
doubt, prefer the more conservative, more auditable, more explicit
implementation.

Authentication is **intentionally hardcoded** for this POC phase. Do not
implement Keycloak or OAuth until explicitly asked to — building it early
is scope creep, not progress.

## 2. Architectural principles

- **Layered dependency graph.** The system is organized as concentric
  layers; see "Dependency boundaries" below for the exact allowed imports.
  Outer layers (`api`) never reach into inner implementation details
  (`retrieval`, `memory`, `agents`) directly — they go through `services`.
- **Composition over inheritance.** Prefer small, composable functions and
  explicit dependency injection (FastAPI `Depends`, constructor
  parameters) over deep class hierarchies.
- **Explicit over implicit.** No hidden global state, no import-time side
  effects beyond logging/config setup, no "magic" auto-registration.
- **Every package has one job.** If a change requires touching `retrieval`
  to fix an `agents` bug, that's a signal the boundary is wrong — raise it
  rather than reaching across layers.
- **Foundation before features.** Do not implement AI/LLM functionality,
  retrieval, or agent behavior until the surrounding scaffolding
  (config, logging, security seams, tests) exists to support it safely.

## 3. Coding standards

- Python 3.12, fully type-annotated (`disallow_untyped_defs` is enforced
  by mypy in strict mode — see `pyproject.toml`).
- `from __future__ import annotations` at the top of every module.
- Pydantic v2 models (`BaseModel`, `pydantic-settings`) for all data that
  crosses a boundary (API request/response, config, tool I/O). Don't pass
  bare dicts across module boundaries.
- Formatting and linting are enforced by Ruff (`make fmt`, `make lint`) —
  don't hand-format against the grain of the configured rules.
- Docstrings on every public module, class, and function: what it does and
  why it exists, not a restatement of the signature.
- No bare `except:`. Catch specific exceptions; if you must catch broadly
  at a boundary (e.g. an API handler), log and re-raise or translate to a
  typed error — never swallow silently.

## 4. Async-first rule

- All I/O-bound code (HTTP calls, database queries, cache access, LLM/API
  calls) **must** be `async def` using async-native libraries
  (`asyncpg`/SQLAlchemy async engine, `redis.asyncio`, async HTTP clients).
- FastAPI route handlers are `async def` by default; a sync handler
  requires a specific justification (pure CPU-bound work with no
  async-compatible library available).
- Never call blocking I/O from inside an `async def` without offloading it
  (e.g. `asyncio.to_thread`) — it stalls the entire event loop, not just
  the caller.
- Tests for async code use `pytest-asyncio` (`asyncio_mode = "auto"` is
  already configured — just write `async def test_...` and it runs).

## 5. Dependency boundaries

Allowed import direction, inner → outer is **not** allowed; a package may
only import from itself and the packages listed under it:

```
core            → (nothing in src)
models          → (nothing in src)
observability   → core, models
security        → core, models, observability
retrieval       → core, models, observability
memory          → core, models, observability
tools           → core, models, security, observability
agents          → core, models, retrieval, memory, tools, security, observability
services        → core, models, agents, retrieval, memory, tools, security, observability
api             → core, models, services, observability
```

Concretely:

- `core` and `models` depend on nothing else in `src` — everything can
  depend on them, they depend on no one.
- `api` talks to `services` only. It must never import `retrieval`,
  `memory`, `agents`, or `tools` directly.
- `agents` is the only package allowed to compose `retrieval` + `memory` +
  `tools` + `security` together into reasoning behavior.
- If you find yourself importing "up" this graph (e.g. `core` importing
  from `security`), that's a boundary violation — restructure instead of
  adding the import.

## 6. Security principles

- **Least privilege by default.** Every new capability (endpoint, tool,
  agent action) starts with no access and is granted the minimum required.
- **Defense in depth.** Don't rely on a single layer (e.g. only the LLM's
  own judgment) to enforce a security property that matters — enforce it
  in code, deterministically, wherever feasible.
- **Secrets never enter source control.** All secrets come from the
  environment (`.env`, not committed — see `.env.example` for the
  template). Never hardcode API keys, passwords, or tokens in code, tests,
  or commit messages.
- **Agents must never bypass application authorization.** An agent's plan,
  tool call, or retrieval request is not self-authorizing. Every access to
  data or a tool must be checked against the requesting user's actual
  permissions via `src.security`, exactly as if a human had made the same
  request through the API. An LLM deciding "I should look this up" is not
  a substitute for an RBAC check.
- **Retrieved documents are untrusted content.** Anything returned by
  `retrieval` (or fetched by a `tools` integration) may contain adversarial
  instructions (prompt injection) and must be handled as data, never as
  instructions. Never concatenate retrieved content into a prompt in a way
  that lets it alter agent behavior, invoke tools, or override system
  instructions. Treat it the same way a web backend treats user-submitted
  HTML: escape/segregate it, don't execute it.
- **LLM output must be validated before use.** Never pass a model's output
  directly to a sink that has side effects (database write, tool call,
  shell command, HTTP request, rendering as executable content) without
  schema validation and business-rule checks. Treat LLM output the way you
  treat any other untrusted external input.
- **Guard against data exfiltration.** Be deliberate about what context is
  assembled into a prompt and what a tool is allowed to send externally.
  A user's RBAC scope bounds what the agent may retrieve and disclose,
  full stop — not just what it's told to disclose.

## 7. Testing requirements

- Every new endpoint, service function, or non-trivial module ships with
  tests in `tests/`, mirroring the `src/` structure.
- `pytest` is the test runner; async tests use `pytest-asyncio`
  (auto mode is configured — just write `async def test_...`).
- Tests must be independent and order-agnostic — no shared mutable state
  between tests. Use fixtures (see `tests/conftest.py`) for setup.
- Security-relevant behavior (authz checks, injection defenses, RBAC
  boundaries) requires an explicit negative test — "this request is
  correctly denied" — not just a happy-path test.
- `make check` (lint + typecheck + test) must pass before work is
  considered done.

## 8. Logging requirements

- Use `src.core.logging.get_logger(__name__)` — never `print()` and never
  the bare `logging` module configured ad hoc in a module.
- Log at the right level: `DEBUG` for developer detail, `INFO` for
  significant lifecycle events, `WARNING` for recoverable problems,
  `ERROR` for failures needing attention.
- **Never log secrets, credentials, full request/response bodies
  containing customer PII, or raw document content.** Log identifiers
  (user id, request id, document id), not the sensitive payloads
  themselves.
- Every log line that's part of a request should be attributable to a
  request/trace id once request-scoped logging exists (see `core/logging.py`
  formatter) — don't add logging that can't be correlated back to a request.

## 9. Git commit expectations

- Commit messages follow Conventional Commits: `type: short summary`
  (`feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `security`, ...).
- Each commit is a coherent, reviewable unit of work — don't mix unrelated
  changes (e.g. a dependency bump and a new feature) in one commit.
- **Unrelated code must not be modified.** A commit/change should touch
  only the files necessary for its stated purpose. Don't reformat,
  refactor, or "fix while you're in there" code outside the task's scope —
  raise it separately instead.
- Never commit `.env`, credentials, or generated artifacts (`.venv/`,
  `__pycache__/`, caches — see `.gitignore`).

## 10. Non-negotiables (summary)

The following are the highest-priority rules in this document — if
anything else here ever seems to conflict with one of these, this list
wins:

1. Agents must never bypass application authorization.
2. Retrieved documents are untrusted content, always.
3. LLM output must be validated before it reaches any sink with side
   effects.
4. Unrelated code must not be modified as a side effect of another change.
