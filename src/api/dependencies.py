"""FastAPI dependency providers.

Centralizes the app's `Depends(...)` wiring so route modules stay thin and
never call `get_settings()` (or any future service provider) directly —
see CLAUDE.md "Composition over inheritance": prefer explicit dependency
injection over hidden global access. Route handlers request a
`SettingsDep` parameter instead, which also makes them trivially testable
via FastAPI's `app.dependency_overrides`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]
"""Injects the cached, validated application `Settings`."""
