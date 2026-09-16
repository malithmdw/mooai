"""Services: application/orchestration layer consumed by the API.

`services` composes `agents`, `retrieval`, `memory`, `tools`, `security`,
and `observability` into use cases, and is the only layer the `api` package
is allowed to call directly. Keeping this boundary means `api` never talks
to `retrieval`, `memory`, or `agents` directly — see CLAUDE.md
"Dependency boundaries".

Public surface
--------------
- ``ChatService``    — orchestrates the full agent pipeline for one chat turn.
"""

from src.services.chat import ChatService

__all__ = ["ChatService"]
