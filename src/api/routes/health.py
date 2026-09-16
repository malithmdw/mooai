"""Health-check endpoint.

Liveness only — no downstream dependency checks (DB, Redis, Pinecone, etc.)
are performed yet. Deep readiness checks are a future addition once those
integrations exist.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import SettingsDep
from src.models.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(settings: SettingsDep) -> HealthResponse:
    """Report basic service liveness."""
    return HealthResponse(status="ok", service=settings.app_name)
