"""Tool invocation records — the request/response pair for one MCP tool call.

A tool's output is external, untrusted data: the same discipline that
applies to retrieved documents applies to `ToolResult.output` — see
CLAUDE.md "Retrieved documents are untrusted content" and "LLM output must
be validated before use".
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from src.models.common import EntityId, NonEmptyStr, utcnow


class ToolCall(BaseModel):
    """A request to invoke one external tool, issued by an agent."""

    model_config = ConfigDict(frozen=True)

    tool_call_id: EntityId
    tool_name: NonEmptyStr
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    requested_at: datetime = Field(default_factory=utcnow)


class ToolResult(BaseModel):
    """The outcome of one `ToolCall`."""

    model_config = ConfigDict(frozen=True)

    tool_call_id: EntityId
    success: bool
    output: JsonValue = None
    error_message: str | None = None
    completed_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _error_message_required_on_failure(self) -> ToolResult:
        if not self.success and not (self.error_message and self.error_message.strip()):
            raise ValueError("error_message is required when success is False")
        return self
