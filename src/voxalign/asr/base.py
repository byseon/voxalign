"""ASR backend interfaces and shared models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AsrBackendName = Literal[
    "disabled",
    "auto",
    "parakeet",
    "parakeet_tdt",
    "crisper_whisper",
    "whisper_large_v3",
]
AsrSource = Literal["real", "simulated"]


@dataclass(frozen=True)
class AsrResult:
    """ASR transcription output."""

    transcript: str
    language_code: str
    backend: AsrBackendName
    model_id: str
    source: AsrSource
