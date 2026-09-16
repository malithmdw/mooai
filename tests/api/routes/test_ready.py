"""Integration tests for `GET /ready`."""

from __future__ import annotations

import httpx


async def test_ready_returns_ready(async_client: httpx.AsyncClient) -> None:
    response = await async_client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["service"] == "enterprise-ai-assistant"
