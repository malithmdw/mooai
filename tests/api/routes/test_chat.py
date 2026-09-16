"""Integration tests for `POST /api/v1/chat`."""

from __future__ import annotations

import httpx


async def test_chat_returns_placeholder_response(async_client: httpx.AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/chat",
        json={"request_id": "req-1", "user_id": "user-1", "message": "What is our loan policy?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"]
    assert body["response_id"]
    assert body["conversation_id"]
    assert body["evidence"] == []
    assert body["citations"] == []


async def test_chat_generates_conversation_id_when_absent(async_client: httpx.AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/chat", json={"request_id": "req-1", "user_id": "user-1", "message": "Hello"}
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"]


async def test_chat_preserves_supplied_conversation_id(async_client: httpx.AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/chat",
        json={
            "request_id": "req-1",
            "user_id": "user-1",
            "conversation_id": "conv-123",
            "message": "Hello",
        },
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "conv-123"


async def test_chat_rejects_empty_message(async_client: httpx.AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/chat", json={"request_id": "req-1", "user_id": "user-1", "message": "   "}
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"


async def test_chat_rejects_missing_required_field(async_client: httpx.AsyncClient) -> None:
    response = await async_client.post("/api/v1/chat", json={"message": "Hello"})

    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"
