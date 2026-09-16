"""Integration tests for CORS configuration."""

from __future__ import annotations

import httpx


async def test_allows_configured_origin(async_client: httpx.AsyncClient) -> None:
    response = await async_client.get("/health", headers={"Origin": "http://localhost:8501"})

    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"


async def test_does_not_reflect_unconfigured_origin(async_client: httpx.AsyncClient) -> None:
    response = await async_client.get("/health", headers={"Origin": "http://evil.example.com"})

    assert "access-control-allow-origin" not in response.headers
