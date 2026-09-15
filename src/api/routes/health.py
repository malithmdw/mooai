"""Health-check endpoint.

Liveness only — no downstream dependency checks (DB, Redis, Pinecone, etc.)
are performed yet. Deep readiness checks are a future addition once those
integrations exist.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.core.config import get_settings
from src.models.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Report basic service liveness."""
    settings = get_settings()
    return HealthResponse(status="ok", service=settings.app_name)
