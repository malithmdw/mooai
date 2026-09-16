"""Schemas for the health/readiness endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body for `GET /health` — liveness only."""

    status: str = Field(default="ok", description="Service liveness status.")
    service: str = Field(description="Name of the service reporting health.")


class ReadinessResponse(BaseModel):
    """Response body for `GET /ready` — readiness to accept traffic.

    Kept as a distinct model from `HealthResponse` even though its shape
    is identical today: readiness will start reflecting downstream
    dependency checks (Postgres, Redis, Pinecone) once those integrations
    exist, while liveness never should.
    """

    status: str = Field(default="ready", description="Service readiness status.")
    service: str = Field(description="Name of the service reporting readiness.")
