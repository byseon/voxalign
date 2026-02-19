"""Language pack base types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedTranscript:
    """Normalized transcript and tokenized result."""

    original: str
    normalized: str
    tokens: list[str]


class BaseLanguagePack(ABC):
    """Abstract interface for language-specific normalization behavior."""

    code: str
    name: str
    normalizer_id: str

    @abstractmethod
    def normalize(self, transcript: str) -> NormalizedTranscript:
        """Normalize and tokenize input transcript text."""
