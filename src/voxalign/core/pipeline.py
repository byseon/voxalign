"""Deterministic baseline alignment pipeline.

This is a placeholder implementation to establish the output contract.
The CTC/trellis backend will replace timing generation in Phase 2/3.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from voxalign.models import (
    AlignmentMetadata,
    AlignRequest,
    AlignResponse,
    PhonemeAlignment,
    WordAlignment,
)

_WORD_PATTERN = re.compile(r"[0-9A-Za-z]+(?:['-][0-9A-Za-z]+)?")
_MODEL_ID = "baseline-rule-v1"
_ALGORITHM = "uniform-token-distribution"


def run_alignment(request: AlignRequest) -> AlignResponse:
    """Produce deterministic, schema-compliant alignments for a transcript."""
    words = _tokenize_words(request.transcript)
    duration_sec = _estimate_duration_sec(len(words))
    word_alignments = _build_word_alignments(words, duration_sec)
    phoneme_alignments = (
        _build_phoneme_alignments(word_alignments) if request.include_phonemes else []
    )

    resolved_language = "und" if request.language == "auto" else request.language
    metadata = AlignmentMetadata(
        language=resolved_language,
        model_id=_MODEL_ID,
        algorithm=_ALGORITHM,
        generated_at=datetime.now(UTC),
        duration_sec=duration_sec,
        sample_rate_hz=request.sample_rate_hz,
    )
    return AlignResponse(metadata=metadata, words=word_alignments, phonemes=phoneme_alignments)


def _tokenize_words(transcript: str) -> list[str]:
    return _WORD_PATTERN.findall(transcript)


def _estimate_duration_sec(word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    return round(max(1.0, word_count * 0.32), 3)


def _build_word_alignments(words: list[str], duration_sec: float) -> list[WordAlignment]:
    if not words:
        return []

    step = duration_sec / len(words)
    output: list[WordAlignment] = []
    for index, word in enumerate(words):
        start = round(step * index, 3)
        end = round(step * (index + 1), 3)
        if index == len(words) - 1:
            end = duration_sec
        confidence = round(max(0.75, 0.98 - index * 0.01), 3)
        output.append(
            WordAlignment(
                word=word,
                start_sec=start,
                end_sec=end,
                confidence=confidence,
            )
        )
    return output


def _build_phoneme_alignments(words: list[WordAlignment]) -> list[PhonemeAlignment]:
    output: list[PhonemeAlignment] = []

    for word_index, word in enumerate(words):
        phonemes = [char.upper() for char in word.word if char.isalpha()]
        if not phonemes:
            phonemes = [word.word]

        span = word.end_sec - word.start_sec
        step = span / len(phonemes) if span > 0 else 0.0
        for phoneme_index, phoneme in enumerate(phonemes):
            start = round(word.start_sec + step * phoneme_index, 3)
            end = round(word.start_sec + step * (phoneme_index + 1), 3)
            if phoneme_index == len(phonemes) - 1:
                end = word.end_sec
            confidence = round(max(0.65, word.confidence - 0.04), 3)
            output.append(
                PhonemeAlignment(
                    phoneme=phoneme,
                    word_index=word_index,
                    start_sec=start,
                    end_sec=end,
                    confidence=confidence,
                )
            )

    return output
