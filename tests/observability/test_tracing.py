"""Tests for `src.observability.tracing`, using a mocked LangSmith client.

No test in this file makes a real network call to LangSmith: `_get_client`
is always monkeypatched, and tests that exercise the "tracing disabled"
path assert it is never even called.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from src.core.config import get_settings
from src.observability import tracing


def _enable_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    get_settings.cache_clear()


def _disable_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_configure_flag() -> Iterator[None]:
    """Isolate `configure_tracing`'s idempotency flag between tests."""
    tracing._CONFIGURED = False
    yield
    tracing._CONFIGURED = False


@pytest.fixture
def mock_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """A mock LangSmith client installed in place of `_get_client`."""
    client = MagicMock()
    monkeypatch.setattr(tracing, "_get_client", lambda: client)
    return client


class TestIsTracingEnabled:
    def test_true_when_flag_set_and_api_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_tracing(monkeypatch)
        assert tracing.is_tracing_enabled() is True

    def test_false_when_flag_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
        get_settings.cache_clear()
        assert tracing.is_tracing_enabled() is False

    def test_false_when_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _disable_tracing(monkeypatch)
        assert tracing.is_tracing_enabled() is False


class TestConfigureTracing:
    def test_idempotent_and_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _disable_tracing(monkeypatch)
        tracing.configure_tracing()
        tracing.configure_tracing()  # second call must be a no-op, not an error

    def test_does_not_construct_a_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_tracing(monkeypatch)
        monkeypatch.setattr(
            tracing,
            "_get_client",
            lambda: (_ for _ in ()).throw(AssertionError("should not construct a client")),
        )
        tracing.configure_tracing()  # only logs; must not touch the client


class TestTracedRunDisabled:
    """Business code keeps working, with zero LangSmith interaction."""

    def test_block_still_executes_and_no_client_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_tracing(monkeypatch)
        monkeypatch.setattr(
            tracing,
            "_get_client",
            lambda: (_ for _ in ()).throw(AssertionError("should not construct a client")),
        )

        executed = False
        with tracing.trace_conversation("conv-1", user_id="user-1"):
            executed = True

        assert executed is True

    def test_exceptions_from_the_block_still_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_tracing(monkeypatch)

        with pytest.raises(ValueError, match="boom"), tracing.trace_tool_call("search"):
            raise ValueError("boom")


class TestTracedRunEnabled:
    def test_trace_conversation_creates_and_ends_run(
        self, monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock
    ) -> None:
        _enable_tracing(monkeypatch)

        with tracing.trace_conversation("conv-1", user_id="user-1"):
            pass

        assert mock_client.create_run.call_count == 1
        _, kwargs = mock_client.create_run.call_args
        assert kwargs["name"] == "conversation"
        assert kwargs["run_type"] == "chain"
        assert kwargs["project_name"] == get_settings().langsmith_project
        assert kwargs["extra"] == {"metadata": {"conversation_id": "conv-1", "user_id": "user-1"}}

        assert mock_client.update_run.call_count == 1
        run_id_arg, update_kwargs = (
            mock_client.update_run.call_args[0][0],
            mock_client.update_run.call_args[1],
        )
        assert run_id_arg == kwargs["id"]
        assert update_kwargs["error"] is None

    def test_reraises_block_exception_and_records_error(
        self, monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock
    ) -> None:
        _enable_tracing(monkeypatch)

        with (
            pytest.raises(ValueError, match="boom"),
            tracing.trace_tool_call("search", conversation_id="conv-1"),
        ):
            raise ValueError("boom")

        _, update_kwargs = mock_client.update_run.call_args
        assert update_kwargs["error"] == "boom"

    def test_swallows_create_run_failure_and_still_runs_block(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _enable_tracing(monkeypatch)
        mock_client.create_run.side_effect = RuntimeError("langsmith unreachable")

        executed = False
        with (
            caplog.at_level("WARNING", logger="src.observability.tracing"),
            tracing.trace_retrieval(query_length=42),
        ):
            executed = True

        assert executed is True
        assert mock_client.update_run.call_count == 0  # no run_id to end
        assert any(
            record.event_type == "observability.trace_start_failed" for record in caplog.records
        )

    def test_swallows_update_run_failure(
        self, monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock
    ) -> None:
        _enable_tracing(monkeypatch)
        mock_client.update_run.side_effect = RuntimeError("langsmith unreachable")

        executed = False
        with tracing.trace_validation("citation_check"):
            executed = True

        assert executed is True  # no exception escaped despite update_run failing

    def test_traced_run_sanitizes_metadata(
        self, monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock
    ) -> None:
        _enable_tracing(monkeypatch)

        with tracing._traced_run("custom", "chain", api_key="sk-secret", count=3):
            pass

        _, kwargs = mock_client.create_run.call_args
        assert kwargs["extra"]["metadata"] == {"api_key": "***REDACTED***", "count": 3}


class TestHelperNamingConventions:
    def test_trace_agent_node_metadata(
        self, monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock
    ) -> None:
        _enable_tracing(monkeypatch)

        with tracing.trace_agent_node("supervisor", conversation_id="conv-1"):
            pass

        _, kwargs = mock_client.create_run.call_args
        assert kwargs["name"] == "agent_node:supervisor"
        assert kwargs["run_type"] == "chain"
        assert kwargs["extra"]["metadata"] == {
            "node_name": "supervisor",
            "conversation_id": "conv-1",
        }

    def test_trace_tool_call_metadata(
        self, monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock
    ) -> None:
        _enable_tracing(monkeypatch)

        with tracing.trace_tool_call("knowledge_search", conversation_id="conv-1"):
            pass

        _, kwargs = mock_client.create_run.call_args
        assert kwargs["name"] == "tool:knowledge_search"
        assert kwargs["run_type"] == "tool"

    def test_trace_retrieval_never_carries_raw_query_text(
        self, monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock
    ) -> None:
        _enable_tracing(monkeypatch)

        with tracing.trace_retrieval(query_length=17, conversation_id="conv-1"):
            pass

        _, kwargs = mock_client.create_run.call_args
        assert kwargs["name"] == "retrieval"
        assert kwargs["run_type"] == "retriever"
        metadata = kwargs["extra"]["metadata"]
        assert metadata == {"query_length": 17, "conversation_id": "conv-1"}
        assert "query" not in metadata

    def test_trace_validation_metadata(
        self, monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock
    ) -> None:
        _enable_tracing(monkeypatch)

        with tracing.trace_validation("citation_check", conversation_id="conv-1"):
            pass

        _, kwargs = mock_client.create_run.call_args
        assert kwargs["name"] == "validation:citation_check"
        assert kwargs["run_type"] == "chain"
