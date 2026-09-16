"""Integration tests for `GET /api/v1/conversations/{conversation_id}`."""

from __future__ import annotations

import httpx


async def test_get_conversation_reports_not_implemented(async_client: httpx.AsyncClient) -> None:
    response = await async_client.get("/api/v1/conversations/conv-123")

    assert response.status_code == 501
    body = response.json()
    assert body["error_code"] == "NOT_IMPLEMENTED"
    assert "conv-123" in body["message"]


async def test_get_conversation_rejects_malformed_id(async_client: httpx.AsyncClient) -> None:
    response = await async_client.get("/api/v1/conversations/has%20spaces")

    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"
