"""English language pack."""

from __future__ import annotations

import re

from voxalign.languages.base import BaseLanguagePack, NormalizedTranscript

_SPACES_RE = re.compile(r"\s+")
_NON_ENGLISH_RE = re.compile(r"[^a-z0-9'\-\s]")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")
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


class EnglishLanguagePack(BaseLanguagePack):
    """English normalization rules."""

    code = "en"
    name = "English"
    normalizer_id = "english-basic-v1"

    def normalize(self, transcript: str) -> NormalizedTranscript:
        normalized = transcript.translate(_CHARMAP).casefold()
        normalized = _NON_ENGLISH_RE.sub(" ", normalized)
        normalized = _SPACES_RE.sub(" ", normalized).strip()
        tokens = _TOKEN_RE.findall(normalized)
        return NormalizedTranscript(original=transcript, normalized=normalized, tokens=tokens)


ENGLISH_PACK = EnglishLanguagePack()
