"""Tests for `src.core.logging`."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Iterator
from uuid import uuid4

import pytest

from src.core.logging import (
    _JsonFormatter,
    _sanitize_extra,
    bind_log_context,
    get_agent_name,
    get_conversation_id,
    get_request_id,
    get_user_id,
    log_debug,
    log_error,
    log_event,
    log_info,
    log_warning,
)


class _ListHandler(logging.Handler):
    """Collects emitted records in-memory, for direct assertion in tests."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def collector() -> Iterator[tuple[logging.Logger, list[logging.LogRecord]]]:
    """An isolated logger (not the app's root logger) with a list-backed handler."""
    logger = logging.getLogger(f"test.logging.{uuid4()}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = _ListHandler()
    logger.addHandler(handler)
    yield logger, handler.records
    logger.removeHandler(handler)


class TestJsonFormatter:
    def test_includes_required_fields(self) -> None:
        logger = logging.getLogger(f"test.formatter.{uuid4()}")
        record = logger.makeRecord(logger.name, logging.INFO, __file__, 1, "hello", (), None)
        record.event_type = "chat.received"
        record.request_id = "req-1"
        record.conversation_id = "conv-1"
        record.user_id = "user-1"
        record.agent_name = "supervisor"

        payload = json.loads(_JsonFormatter(service_name="ai-assistant").format(record))

        assert payload["level"] == "INFO"
        assert payload["service"] == "ai-assistant"
        assert payload["logger"] == logger.name
        assert payload["message"] == "hello"
        assert payload["event_type"] == "chat.received"
        assert payload["request_id"] == "req-1"
        assert payload["conversation_id"] == "conv-1"
        assert payload["user_id"] == "user-1"
        assert payload["agent_name"] == "supervisor"
        assert "timestamp" in payload

    def test_missing_context_fields_are_null(self) -> None:
        logger = logging.getLogger(f"test.formatter.{uuid4()}")
        record = logger.makeRecord(logger.name, logging.INFO, __file__, 1, "hi", (), None)

        payload = json.loads(_JsonFormatter(service_name="svc").format(record))

        assert payload["request_id"] is None
        assert payload["conversation_id"] is None
        assert payload["user_id"] is None
        assert payload["agent_name"] is None
        assert "error" not in payload

    def test_includes_structured_error_info(self) -> None:
        logger = logging.getLogger(f"test.formatter.{uuid4()}")
        try:
            raise ValueError("boom")
        except ValueError:
            record = logger.makeRecord(
                logger.name, logging.ERROR, __file__, 1, "tool failed", (), sys.exc_info()
            )

        payload = json.loads(_JsonFormatter(service_name="svc").format(record))

        assert payload["error"]["type"] == "ValueError"
        assert payload["error"]["message"] == "boom"
        assert "ValueError: boom" in payload["error"]["traceback"]

    def test_omits_fields_key_when_no_structured_fields(self) -> None:
        logger = logging.getLogger(f"test.formatter.{uuid4()}")
        record = logger.makeRecord(logger.name, logging.INFO, __file__, 1, "hi", (), None)
        record.fields = {}

        payload = json.loads(_JsonFormatter(service_name="svc").format(record))

        assert "fields" not in payload


class TestSanitizeExtra:
    @pytest.mark.parametrize(
        "key",
        ["api_key", "API_KEY", "password", "authorization", "access_token", "client_secret"],
    )
    def test_redacts_secret_like_keys(self, key: str) -> None:
        sanitized = _sanitize_extra({key: "super-secret-value"})
        assert sanitized[key] == "***REDACTED***"

    def test_truncates_long_document_content(self) -> None:
        long_text = "x" * 500
        sanitized = _sanitize_extra({"document_content": long_text})
        assert len(sanitized["document_content"]) < 500  # type: ignore[arg-type]
        assert "truncated" in sanitized["document_content"]  # type: ignore[operator]

    def test_leaves_short_content_untouched(self) -> None:
        sanitized = _sanitize_extra({"excerpt": "short"})
        assert sanitized["excerpt"] == "short"

    def test_passes_through_ordinary_fields(self) -> None:
        sanitized = _sanitize_extra({"document_id": "doc-1", "latency_ms": 42})
        assert sanitized == {"document_id": "doc-1", "latency_ms": 42}


class TestBindLogContext:
    def test_sets_and_resets_context(self) -> None:
        assert get_request_id() is None

        with bind_log_context(request_id="req-1", conversation_id="conv-1", user_id="user-1"):
            assert get_request_id() == "req-1"
            assert get_conversation_id() == "conv-1"
            assert get_user_id() == "user-1"

        assert get_request_id() is None
        assert get_conversation_id() is None
        assert get_user_id() is None

    def test_resets_on_exception(self) -> None:
        with pytest.raises(RuntimeError), bind_log_context(request_id="req-1"):
            raise RuntimeError("boom")

        assert get_request_id() is None

    def test_nested_context_overrides_only_provided_fields(self) -> None:
        with bind_log_context(request_id="req-1", conversation_id="conv-1"):
            with bind_log_context(agent_name="retrieval"):
                assert get_request_id() == "req-1"
                assert get_conversation_id() == "conv-1"
                assert get_agent_name() == "retrieval"

            assert get_agent_name() is None
            assert get_request_id() == "req-1"

    async def test_context_propagates_across_await_and_tasks(self) -> None:
        async def read_context() -> str | None:
            await asyncio.sleep(0)
            return get_request_id()

        with bind_log_context(request_id="req-async"):
            direct = await read_context()
            task_result = await asyncio.create_task(read_context())

        assert direct == "req-async"
        assert task_result == "req-async"
        assert get_request_id() is None


class TestLogHelpers:
    def test_log_event_attaches_context_and_event_type(
        self, collector: tuple[logging.Logger, list[logging.LogRecord]]
    ) -> None:
        logger, records = collector

        with bind_log_context(request_id="req-1", conversation_id="conv-1", user_id="user-1"):
            log_event(logger, logging.INFO, "retrieval.completed", "found 3 documents", hits=3)

        assert len(records) == 1
        record = records[0]
        assert record.event_type == "retrieval.completed"
        assert record.request_id == "req-1"
        assert record.conversation_id == "conv-1"
        assert record.user_id == "user-1"
        assert record.fields == {"hits": 3}

    def test_log_event_sanitizes_fields(
        self, collector: tuple[logging.Logger, list[logging.LogRecord]]
    ) -> None:
        logger, records = collector

        log_event(logger, logging.INFO, "tool_call.issued", "calling tool", api_key="sk-secret")

        assert records[0].fields == {"api_key": "***REDACTED***"}

    @pytest.mark.parametrize(
        ("helper", "expected_level"),
        [
            (log_debug, logging.DEBUG),
            (log_info, logging.INFO),
            (log_warning, logging.WARNING),
        ],
    )
    def test_level_helpers_use_correct_level(
        self,
        collector: tuple[logging.Logger, list[logging.LogRecord]],
        helper: object,
        expected_level: int,
    ) -> None:
        logger, records = collector

        helper(logger, "generic.event", "something happened")  # type: ignore[operator]

        assert records[0].levelno == expected_level

    def test_log_error_attaches_exception_info(
        self, collector: tuple[logging.Logger, list[logging.LogRecord]]
    ) -> None:
        logger, records = collector
        error = ValueError("tool timed out")

        log_error(logger, "tool_call.failed", "tool call failed", error=error)

        record = records[0]
        assert record.levelno == logging.ERROR
        assert record.exc_info is not None
        assert record.exc_info[0] is ValueError
        assert record.exc_info[1] is error

    def test_log_error_without_exception_has_no_exc_info(
        self, collector: tuple[logging.Logger, list[logging.LogRecord]]
    ) -> None:
        logger, records = collector

        log_error(logger, "validation.failed", "validation failed")

        assert records[0].exc_info is None
