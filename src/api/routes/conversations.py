"""Conversation retrieval endpoint.

No conversation persistence exists yet (`src.memory` is not implemented —
see CLAUDE.md "Foundation before features"), so every request correctly
and explicitly reports `501 Not Implemented` rather than a `404` that
would misleadingly imply "we checked a real store and found nothing."
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.models.chat import Conversation
from src.models.common import EntityId

router = APIRouter(prefix="/api/v1", tags=["conversations"])


@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: EntityId) -> Conversation:
    """Retrieve a conversation by ID.

    `conversation_id` is validated by the same `EntityId` constraints used
    throughout `src.models` (see CLAUDE.md "Pydantic validation"): a
    malformed ID is rejected with `422` before this handler ever runs.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"Conversation retrieval is not implemented yet (requested: {conversation_id!r}).",
    )
