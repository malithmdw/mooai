"""Shared Pydantic schemas and data models.

`models` defines data shapes only — no business logic, no I/O. It may be
imported by any other `src` package; it must not import from `api`,
`agents`, `retrieval`, `memory`, `tools`, `security`, `observability`, or
`services` (no upward or lateral dependencies).
"""
