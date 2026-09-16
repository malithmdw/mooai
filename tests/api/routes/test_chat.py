"""Integration tests for `POST /api/v1/chat`.

All tests mock ``ChatService`` via ``app.dependency_overrides`` so they
exercise the HTTP layer (auth, request validation, response shape) without
requiring real Pinecone / Anthropic credentials or a running agent graph.

POC credentials used (see docs/authentication.md):
  analyst01 / Analyst01#Poc2026
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.api.dependencies import get_chat_service
from src.models.chat import ChatResponse
from src.models.evidence import Citation, Evidence

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

_AUTH = ("analyst01", "Analyst01#Poc2026")
"""POC analyst credentials used across all chat tests."""


def _mock_service(response_message: str = "The default rate was 2.3%.") -> MagicMock:
    """Return a mock ``ChatService`` whose ``chat()`` returns a fixed response."""
    service = MagicMock()
    service.chat = AsyncMock(
        return_value=ChatResponse(
            response_id=str(uuid.uuid4()),
            conversation_id="conv-test-001",
            message=response_message,
        )
    )
    return service


# ---------------------------------------------------------------------------
# Basic endpoint behaviour
# ---------------------------------------------------------------------------


async def test_chat_returns_response(async_client: httpx.AsyncClient) -> None:
    from src.api.main import create_app

    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: _mock_service()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/chat",
            json={"request_id": "req-1", "user_id": "user-1", "message": "What is our loan policy?"},
            auth=_AUTH,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "The default rate was 2.3%."
    assert body["response_id"]
    assert body["conversation_id"]
    assert body["evidence"] == []
    assert body["citations"] == []


async def test_chat_generates_conversation_id_when_absent(async_client: httpx.AsyncClient) -> None:
    from src.api.main import create_app

    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: _mock_service()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/chat",
            json={"request_id": "req-1", "user_id": "user-1", "message": "Hello"},
            auth=_AUTH,
        )

    assert response.status_code == 200
    assert response.json()["conversation_id"]


async def test_chat_preserves_supplied_conversation_id() -> None:
    from src.api.main import create_app

    supplied_id = "conv-123"

    service = MagicMock()
    service.chat = AsyncMock(
        return_value=ChatResponse(
            response_id=str(uuid.uuid4()),
            conversation_id=supplied_id,
            message="Hello",
        )
    )

    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/chat",
            json={
                "request_id": "req-1",
                "user_id": "user-1",
                "conversation_id": supplied_id,
                "message": "Hello",
            },
            auth=_AUTH,
        )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == supplied_id


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


async def test_chat_rejects_empty_message() -> None:
    from src.api.main import create_app

    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: _mock_service()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/chat",
            json={"request_id": "req-1", "user_id": "user-1", "message": "   "},
            auth=_AUTH,
        )

    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"


async def test_chat_rejects_missing_required_field() -> None:
    from src.api.main import create_app

    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: _mock_service()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
            auth=_AUTH,
        )

    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def test_chat_requires_authentication() -> None:
    """Unauthenticated request must receive 401."""
    from src.api.main import create_app

    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: _mock_service()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/chat",
            json={"request_id": "req-1", "user_id": "user-1", "message": "Hello"},
            # No auth header
        )

    assert response.status_code == 401


async def test_chat_rejects_wrong_password() -> None:
    from src.api.main import create_app

    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: _mock_service()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/chat",
            json={"request_id": "req-1", "user_id": "user-1", "message": "Hello"},
            auth=("analyst01", "wrong-password"),
        )

    assert response.status_code == 401


async def test_all_poc_users_can_call_chat() -> None:
    """All three POC users (VIEWER, ANALYST, ADMINISTRATOR) can access the endpoint."""
    from src.api.main import create_app

    credentials = [
        ("viewer01", "Viewer01#Poc2026"),
        ("analyst01", "Analyst01#Poc2026"),
        ("admin01", "Admin01#Poc2026"),
    ]

    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: _mock_service()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for username, password in credentials:
            response = await client.post(
                "/api/v1/chat",
                json={"request_id": "req-1", "user_id": "user-1", "message": "Hello"},
                auth=(username, password),
            )
            assert response.status_code == 200, (
                f"{username} got {response.status_code}: {response.text}"
            )


# ---------------------------------------------------------------------------
# Service delegation
# ---------------------------------------------------------------------------


async def test_chat_delegates_request_and_user_to_service() -> None:
    """Route handler passes a User with the authenticated role to ChatService.chat."""
    from src.api.main import create_app
    from src.models.enums import Role

    received_args: list[tuple] = []

    service = MagicMock()

    async def capture_chat(request, user):  # type: ignore[no-untyped-def]
        received_args.append((request, user))
        return ChatResponse(
            response_id=str(uuid.uuid4()),
            conversation_id="conv-0001",
            message="Captured.",
        )

    service.chat = capture_chat

    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        await client.post(
            "/api/v1/chat",
            json={"request_id": "req-1", "user_id": "user-1", "message": "What is the policy?"},
            auth=("analyst01", "Analyst01#Poc2026"),
        )

    assert len(received_args) == 1
    _, user = received_args[0]
    assert user.username == "analyst01"
    assert Role.ANALYST in user.roles


# ---------------------------------------------------------------------------
# Response with evidence and citations
# ---------------------------------------------------------------------------


async def test_chat_response_includes_evidence_and_citations() -> None:
    """Evidence and citations from the service are forwarded in the HTTP response."""
    from src.api.main import create_app

    evidence = Evidence(
        evidence_id="DOC-001-chunk-0000",
        document_id="DOC-001",
        excerpt="The loan default rate was 2.3%.",
        relevance_score=0.9,
    )
    citation = Citation(evidence_id="DOC-001-chunk-0000", reference_number=1)

    service = MagicMock()
    service.chat = AsyncMock(
        return_value=ChatResponse(
            response_id=str(uuid.uuid4()),
            conversation_id="conv-0001",
            message="The default rate was 2.3%. [1]",
            evidence=(evidence,),
            citations=(citation,),
        )
    )

    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/chat",
            json={"request_id": "req-1", "user_id": "user-1", "message": "What is the default rate?"},
            auth=_AUTH,
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["evidence"]) == 1
    assert len(body["citations"]) == 1
    assert body["evidence"][0]["evidence_id"] == "DOC-001-chunk-0000"
    assert body["citations"][0]["reference_number"] == 1
