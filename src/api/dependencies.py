"""FastAPI dependency providers.

Centralizes the app's `Depends(...)` wiring so route modules stay thin and
never call `get_settings()` (or any future service provider) directly —
see CLAUDE.md "Composition over inheritance": prefer explicit dependency
injection over hidden global access. Route handlers request a typed
dependency parameter instead, which also makes them trivially testable
via FastAPI's `app.dependency_overrides`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from src.core.config import Settings, get_settings
from src.security.auth import AuthenticatedUser, get_current_user
from src.services.chat import ChatService

SettingsDep = Annotated[Settings, Depends(get_settings)]
"""Injects the cached, validated application `Settings`."""

CurrentUserDep = Annotated[AuthenticatedUser, Depends(get_current_user)]
"""Injects the authenticated caller, or raises `401` — see
`src.security.auth` for the (POC-only, hardcoded) authentication it performs.
"""


def get_chat_service(request: Request) -> ChatService:
    """Return the ``ChatService``, building it lazily on first request.

    The service is constructed once from ``Settings`` and cached on
    ``request.app.state.chat_service``.  Tests override the whole function
    via ``app.dependency_overrides[get_chat_service] = lambda: mock_service``
    so that ``ChatService.from_settings()`` (which creates real provider
    clients) is never called during testing.
    """
    if not hasattr(request.app.state, "chat_service"):
        request.app.state.chat_service = ChatService.from_settings()
    service: ChatService = request.app.state.chat_service
    return service


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
"""Injects the singleton ``ChatService`` built at application startup."""
