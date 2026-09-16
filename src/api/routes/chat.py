"""Chat endpoint.

Accepts a ``ChatRequest`` from an authenticated user, runs it through the
``ChatService`` (which invokes the full LangGraph agent pipeline), and
returns a ``ChatResponse`` with an evidence-grounded answer and citations.

Authentication
--------------
Every request must carry HTTP Basic credentials.  ``CurrentUserDep`` either
resolves to an ``AuthenticatedUser`` or raises ``401``.  The resolved user
drives RBAC filtering throughout the retrieval pipeline.

Error handling
--------------
Unexpected errors from the agent graph propagate to the global exception
handler in ``src.api.exception_handlers``, which returns a structured
``ErrorResponse`` rather than exposing a raw traceback.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import ChatServiceDep, CurrentUserDep
from src.core.logging import get_logger, log_info
from src.models.chat import ChatRequest, ChatResponse
from src.models.user import User
from src.security.auth import AuthenticatedUser

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _to_user(auth_user: AuthenticatedUser) -> User:
    """Convert an ``AuthenticatedUser`` (single role) to the domain ``User`` model."""
    return User(
        user_id=auth_user.user_id,
        username=auth_user.username,
        roles=(auth_user.role,),
    )


@router.post("/chat", response_model=ChatResponse)
async def post_chat(
    chat_request: ChatRequest,
    current_user: CurrentUserDep,
    chat_service: ChatServiceDep,
) -> ChatResponse:
    """Accept a chat request and return an agent-generated response.

    The agent pipeline:

    1. ``InputGuard`` validates the message for prompt-injection patterns.
    2. ``SupervisorAgent`` classifies intent and determines the retrieval route.
    3. ``RetrievalAgent`` fetches relevant document chunks from the knowledge base,
       applying RBAC filters derived from the caller's roles.
    4. ``ResponseAgent`` synthesises a grounded answer, cross-validates all
       citations, and applies the output guard before returning.

    If any guardrail fires (injection, citation failure, output leak), a safe
    canned response is returned rather than the LLM output.
    """
    user = _to_user(current_user)

    log_info(
        logger,
        "chat.request_received",
        "Chat request received",
        conversation_id=chat_request.conversation_id,
        user_id=user.user_id,
        message_length=len(chat_request.message),
    )

    return await chat_service.chat(chat_request, user)
