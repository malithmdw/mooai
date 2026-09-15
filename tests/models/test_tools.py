"""Tests for `src.models.tools.ToolCall` and `ToolResult`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.tools import ToolCall, ToolResult


def test_valid_tool_call_is_immutable() -> None:
    call = ToolCall(
        tool_call_id="call-1",
        tool_name="knowledge_search",
        arguments={"query": "loan approval threshold"},
    )
    with pytest.raises(ValidationError):
        call.tool_name = "other_tool"  # type: ignore[misc]


def test_tool_call_rejects_empty_tool_name() -> None:
    with pytest.raises(ValidationError, match="tool_name"):
        ToolCall(tool_call_id="call-1", tool_name="   ")


def test_successful_tool_result_does_not_require_error_message() -> None:
    result = ToolResult(tool_call_id="call-1", success=True, output={"hits": 3})
    assert result.error_message is None


def test_failed_tool_result_requires_error_message() -> None:
    with pytest.raises(ValidationError, match="error_message is required"):
        ToolResult(tool_call_id="call-1", success=False)


def test_failed_tool_result_rejects_blank_error_message() -> None:
    with pytest.raises(ValidationError, match="error_message is required"):
        ToolResult(tool_call_id="call-1", success=False, error_message="   ")


def test_failed_tool_result_with_error_message_is_valid() -> None:
    result = ToolResult(tool_call_id="call-1", success=False, error_message="timed out")
    assert result.success is False
