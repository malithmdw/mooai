"""Chat endpoint.

No LLM processing is implemented yet (see CLAUDE.md "Foundation before
features"): this returns a fixed placeholder `ChatResponse`, wired through
real request validation, logging, and dependency injection, so the real
implementation has a working endpoint to slot into.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from src.api.dependencies import SettingsDep
from src.core.logging import get_logger, log_info
from src.models.chat import ChatRequest, ChatResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def post_chat(chat_request: ChatRequest, settings: SettingsDep) -> ChatResponse:
    """Accept a chat request and return a placeholder response.

    `conversation_id` is echoed back if the caller supplied one (continuing
    an existing conversation), otherwise a new one is generated — real
    conversation persistence belongs to `src.memory`, not implemented yet.
    """
    conversation_id = chat_request.conversation_id or str(uuid.uuid4())

    log_info(
        logger,
        "chat.request_received",
        "chat request received",
        conversation_id=conversation_id,
        user_id=chat_request.user_id,
        message_length=len(chat_request.message),
    )

    return ChatResponse(
        response_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        message=(
            f"This is a placeholder response from {settings.model_name}. "
            "LLM-backed chat is not implemented yet."
        ),
    )
