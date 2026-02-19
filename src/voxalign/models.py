"""Shared data models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response payload for the API health endpoint."""

    status: Literal["ok"]
    version: str
    env: str


class AlignRequest(BaseModel):
    """Alignment request payload used by both CLI and API."""

    audio_path: str = Field(min_length=1)
    transcript: str = Field(min_length=1)
    language: str = Field(default="auto", min_length=2)
    backend: Literal["uniform", "ctc_trellis"] = "uniform"
    include_phonemes: bool = True
    sample_rate_hz: int | None = Field(default=None, ge=8_000, le=192_000)


class WordAlignment(BaseModel):
    """Word-level alignment result."""

    word: str = Field(min_length=1)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class PhonemeAlignment(BaseModel):
    """Phoneme-level alignment result."""

    phoneme: str = Field(min_length=1)
    word_index: int = Field(ge=0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class AlignmentMetadata(BaseModel):
    """Metadata describing how an alignment was produced."""

    language: str
    alignment_backend: Literal["uniform", "ctc_trellis"]
    normalizer_id: str
    token_count: int = Field(ge=0)
    timing_source: Literal["audio", "heuristic"]
    model_id: str
    algorithm: str
    generated_at: datetime
    duration_sec: float = Field(ge=0.0)
    sample_rate_hz: int | None = Field(default=None, ge=8_000, le=192_000)


class AlignResponse(BaseModel):
    """Canonical alignment output schema."""

    metadata: AlignmentMetadata
    words: list[WordAlignment]
    phonemes: list[PhonemeAlignment]
