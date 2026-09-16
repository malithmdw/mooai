"""Readiness endpoint.

Distinct from `/health` (liveness): readiness is meant to reflect whether
the service is ready to accept traffic, typically by checking downstream
dependencies (Postgres, Redis, Pinecone). None of those integrations
exist yet, so this currently reports the same "always ready" status as
`/health` — it will start performing real checks once those integrations
land, without changing the response shape.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import SettingsDep
from src.models.health import ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/ready", response_model=ReadinessResponse)
async def get_ready(settings: SettingsDep) -> ReadinessResponse:
    """Report service readiness."""
    return ReadinessResponse(status="ready", service=settings.app_name)
