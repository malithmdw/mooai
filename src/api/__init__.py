"""API: FastAPI HTTP interface.

`api` is the outermost layer. It may depend only on `services`, `core`,
`models`, and `observability` — never directly on `agents`, `retrieval`,
`memory`, or `tools`. See CLAUDE.md "Dependency boundaries".
"""
