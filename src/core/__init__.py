"""Core cross-cutting concerns: configuration and logging.

`core` has no dependency on any other `src` package. Every other package may
depend on `core`, but `core` must never import from `api`, `agents`,
`retrieval`, `memory`, `tools`, `security`, `observability`, or `services`.
"""
