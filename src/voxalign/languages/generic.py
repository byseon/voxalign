"""Generic/fallback language pack."""

from __future__ import annotations

import re

from voxalign.languages.base import BaseLanguagePack, NormalizedTranscript

_SPACES_RE = re.compile(r"\s+")
_INVALID_RE = re.compile(r"[^\w\s'\-]", re.UNICODE)
_TOKEN_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)?", re.UNICODE)
_CHARMAP = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
)


class GenericLanguagePack(BaseLanguagePack):
    """Generic normalization pack reusable across languages."""

    def __init__(self, *, code: str, name: str) -> None:
        self.code = code
        self.name = name
        self.normalizer_id = "generic-unicode-v1"

    def normalize(self, transcript: str) -> NormalizedTranscript:
        normalized = transcript.translate(_CHARMAP).casefold()
        normalized = _INVALID_RE.sub(" ", normalized).replace("_", " ")
        normalized = _SPACES_RE.sub(" ", normalized).strip()
        tokens = _TOKEN_RE.findall(normalized)
        return NormalizedTranscript(original=transcript, normalized=normalized, tokens=tokens)


GENERIC_PACK = GenericLanguagePack(code="und", name="Undetermined")
