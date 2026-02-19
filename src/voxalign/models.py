"""Shared data models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response payload for the API health endpoint."""

    status: Literal["ok"]
    version: str
    env: str
