"""Integration tests for `src.api.middleware.RequestContextMiddleware`."""

from __future__ import annotations

import httpx


async def test_response_includes_generated_request_id(async_client: httpx.AsyncClient) -> None:
    response = await async_client.get("/health")

    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


async def test_response_echoes_supplied_request_id(async_client: httpx.AsyncClient) -> None:
    response = await async_client.get("/health", headers={"X-Request-ID": "client-supplied-id"})

    assert response.headers["x-request-id"] == "client-supplied-id"


async def test_each_request_without_a_supplied_id_gets_a_distinct_one(
    async_client: httpx.AsyncClient,
) -> None:
    first = await async_client.get("/health")
    second = await async_client.get("/health")

    assert first.headers["x-request-id"] != second.headers["x-request-id"]
