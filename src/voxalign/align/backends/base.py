"""Alignment backend interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from voxalign.models import WordAlignment

BackendName = Literal["uniform", "ctc_trellis"]


@dataclass(frozen=True)
class BackendResult:
    """Word alignment output with backend metadata."""

    words: list[WordAlignment]
    model_id: str
    algorithm: str


class AlignmentBackend(Protocol):
    """Protocol implemented by concrete alignment backends."""

    name: BackendName

    def align_words(
        self,
        tokens: list[str],
        duration_sec: float,
        *,
        audio_path: str | None = None,
        sample_rate_hz: int | None = None,
    ) -> BackendResult:
        """Align normalized tokens to time boundaries."""
