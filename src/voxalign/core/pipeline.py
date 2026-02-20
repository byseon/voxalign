"""Deterministic baseline alignment pipeline.

This is a placeholder implementation to establish the output contract.
The CTC/trellis backend will replace timing generation in Phase 2/3.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from voxalign.align import resolve_backend
from voxalign.asr.base import AsrResult
from voxalign.asr.registry import transcribe_audio
from voxalign.io import read_audio_metadata
from voxalign.languages import resolve_language_pack
from voxalign.languages.base import BaseLanguagePack
from voxalign.models import (
    AlignmentMetadata,
    AlignRequest,
    AlignResponse,
    PhonemeAlignment,
    WordAlignment,
)

_CRISPER_LICENSE_WARNING = (
    "CrisperWhisper uses CC BY-NC 4.0 (non-commercial). "
    "Do not use this ASR backend for commercial workloads."
)


def run_alignment(request: AlignRequest) -> AlignResponse:
    """Produce deterministic, schema-compliant alignments for a transcript."""
    backend = resolve_backend(request.backend)
    initial_language_pack = resolve_language_pack(request.language)
    transcript, transcript_source, asr_result = _resolve_transcript(
        request=request,
        language_code=initial_language_pack.code,
    )
    language_pack = _resolve_final_language_pack(
        request_language=request.language,
        initial_language_code=initial_language_pack.code,
        asr_result=asr_result,
    )
    normalized = language_pack.normalize(transcript)
    duration_sec, resolved_sample_rate_hz, timing_source = _resolve_timing(
        audio_path=request.audio_path,
        token_count=len(normalized.tokens),
        requested_sample_rate_hz=request.sample_rate_hz,
    )
    backend_result = backend.align_words(
        normalized.tokens,
        duration_sec,
        language_code=language_pack.code,
        audio_path=request.audio_path,
        sample_rate_hz=resolved_sample_rate_hz,
    )
    if request.include_phonemes:
        phoneme_alignments = backend_result.phonemes or _build_phoneme_alignments(
            backend_result.words
        )
    else:
        phoneme_alignments = []

    metadata = AlignmentMetadata(
        language=language_pack.code,
        alignment_backend=backend.name,
        normalizer_id=language_pack.normalizer_id,
        token_count=len(normalized.tokens),
        timing_source=timing_source,
        transcript_source=transcript_source,
        asr_backend=(asr_result.backend if asr_result is not None else None),
        asr_model_id=(asr_result.model_id if asr_result is not None else None),
        license_warning=_license_warning(asr_result),
        model_id=backend_result.model_id,
        algorithm=backend_result.algorithm,
        generated_at=datetime.now(UTC),
        duration_sec=duration_sec,
        sample_rate_hz=resolved_sample_rate_hz,
    )
    return AlignResponse(metadata=metadata, words=backend_result.words, phonemes=phoneme_alignments)


def _estimate_duration_sec(word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    return round(max(1.0, word_count * 0.32), 3)


def _resolve_timing(
    audio_path: str,
    token_count: int,
    requested_sample_rate_hz: int | None,
) -> tuple[float, int | None, Literal["audio", "heuristic"]]:
    metadata = read_audio_metadata(audio_path)
    if metadata is not None and metadata.duration_sec > 0:
        sample_rate_hz = requested_sample_rate_hz or metadata.sample_rate_hz
        return metadata.duration_sec, sample_rate_hz, "audio"

    duration_sec = _estimate_duration_sec(token_count)
    return duration_sec, requested_sample_rate_hz, "heuristic"


def _resolve_transcript(
    *,
    request: AlignRequest,
    language_code: str,
) -> tuple[str, Literal["provided", "asr"], AsrResult | None]:
    if request.transcript is not None:
        transcript = request.transcript.strip()
        if transcript:
            return transcript, "provided", None

    if request.asr == "disabled":
        raise ValueError("transcript is required when --asr is disabled")

    asr_result = transcribe_audio(
        audio_path=request.audio_path,
        language_code=(None if request.language.casefold() == "auto" else language_code),
        backend=request.asr,
        verbatim=request.verbatim,
        sample_rate_hz=request.sample_rate_hz,
    )
    transcript = asr_result.transcript.strip()
    if not transcript:
        raise ValueError("ASR did not return a transcript")
    return transcript, "asr", asr_result


def _resolve_final_language_pack(
    *,
    request_language: str,
    initial_language_code: str,
    asr_result: AsrResult | None,
) -> BaseLanguagePack:
    if request_language.casefold() != "auto":
        return resolve_language_pack(initial_language_code)
    if asr_result is None:
        return resolve_language_pack(initial_language_code)
    detected = asr_result.language_code
    if not detected or detected == "und":
        return resolve_language_pack(initial_language_code)
    return resolve_language_pack(detected)


def _license_warning(asr_result: AsrResult | None) -> str | None:
    if asr_result is None:
        return None
    if asr_result.backend == "crisper_whisper":
        return _CRISPER_LICENSE_WARNING
    return None


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
