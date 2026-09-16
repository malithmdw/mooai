"""Tests for `src.api.exception_handlers`.

`test_404_...`/`test_validation_error_...`/`test_error_response_includes_request_id`
exercise the handlers as wired into the real app via `async_client`.
`unhandled_exception_handler` has no reachable route that triggers it by
design (nothing in this codebase is supposed to raise an unhandled
exception) — it earns a direct unit test instead, since it's the one
handler with a security-relevant behavior to verify: internal exception
detail must never reach the caller.
"""

from __future__ import annotations

import json

import httpx
from starlette.requests import Request

from src.api.exception_handlers import unhandled_exception_handler


def _request(request_id: str | None = None) -> Request:
    request = Request({"type": "http", "method": "GET", "path": "/test", "headers": []})
    if request_id is not None:
        request.state.request_id = request_id
    return request


async def test_404_uses_error_response_shape(async_client: httpx.AsyncClient) -> None:
    response = await async_client.get("/this-route-does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "NOT_FOUND"
    assert body["message"]
    assert "occurred_at" in body


async def test_error_response_includes_the_request_id(async_client: httpx.AsyncClient) -> None:
    response = await async_client.get(
        "/this-route-does-not-exist", headers={"X-Request-ID": "req-xyz"}
    )

    assert response.json()["request_id"] == "req-xyz"


async def test_validation_error_uses_error_response_shape(async_client: httpx.AsyncClient) -> None:
    response = await async_client.post("/api/v1/chat", json={})

    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"


class TestUnhandledExceptionHandler:
    """Direct unit tests — see module docstring for why."""

    async def test_hides_internal_exception_details(self) -> None:
        response = await unhandled_exception_handler(
            _request(request_id="req-1"), RuntimeError("db password is hunter2")
        )

        assert response.status_code == 500
        body = json.loads(response.body)
        assert body["error_code"] == "INTERNAL_ERROR"
        assert "hunter2" not in body["message"]
        assert "RuntimeError" not in body["message"]
        assert body["request_id"] == "req-1"

    async def test_handles_a_request_with_no_bound_request_id(self) -> None:
        response = await unhandled_exception_handler(_request(), ValueError("boom"))

        body = json.loads(response.body)
        assert body["request_id"] is None
