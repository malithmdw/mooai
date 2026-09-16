"""Services: application/orchestration layer consumed by the API.

`services` composes `agents`, `retrieval`, `memory`, `tools`, `security`,
and `observability` into use cases, and is the only layer the `api` package
is allowed to call directly. Keeping this boundary means `api` never talks
to `retrieval`, `memory`, or `agents` directly — see CLAUDE.md
"Dependency boundaries".

Public surface
--------------
- ``ChatService``          — orchestrates the full agent pipeline for one
  chat turn.
- ``ToolExecutionService``  — the single enforced gateway every tool
  invocation (knowledge search, Python analysis, MCP enterprise data) must
  pass through; see `src.services.tool_execution` for the pipeline.
- ``ToolExecutionError``    — internal structured-rejection error raised
  (and always caught) inside `ToolExecutionService.execute`.
"""

from src.services.chat import ChatService
from src.services.tool_execution import ToolExecutionError, ToolExecutionService

__all__ = ["ChatService", "ToolExecutionError", "ToolExecutionService"]
