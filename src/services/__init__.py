"""Services: application/orchestration layer consumed by the API.

`services` composes `agents`, `retrieval`, `memory`, `tools`, `security`,
and `observability` into use cases, and is the only layer the `api` package
is allowed to call directly. Keeping this boundary means `api` never talks
to `retrieval`, `memory`, or `agents` directly — see CLAUDE.md
"Dependency boundaries".
"""
