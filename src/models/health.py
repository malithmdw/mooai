"""Schemas for the health-check endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    status: str = Field(default="ok", description="Service liveness status.")
    service: str = Field(description="Name of the service reporting health.")
