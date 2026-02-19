"""Phoneme-first alignment backend.

Routing strategy:
- English: word boundaries from CTC backend (Parakeet default), then constrained
  phoneme timing within each word span.
- Other languages: phoneme-first timing over the full utterance, then derive
  word spans by grouping phonemes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from voxalign.align.backends.base import BackendName, BackendResult
from voxalign.align.backends.ctc_trellis import CtcTrellisBackend
from voxalign.models import PhonemeAlignment, WordAlignment

_PHONEME_MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
_ALGO_EN = "phoneme-first-en-word-ctc-then-ipa-constrained"
_ALGO_MULTI = "phoneme-first-multilingual-ipa-ctc"
_ALGO_MULTI_FALLBACK = "phoneme-first-multilingual-ipa-fallback-to-ctc-word"
_ENGLISH_CODE = "en"
_KOREAN_CODE = "ko"


@dataclass(frozen=True)
class _WordPhonemes:
    word: str
    phonemes: list[str]


class PhonemeFirstBackend:
    """Phoneme-first forced alignment orchestration backend."""

    name: BackendName = "phoneme_first"

    def __init__(self) -> None:
        self._word_backend = CtcTrellisBackend()

    def align_words(
        self,
        tokens: list[str],
        duration_sec: float,
        *,
        language_code: str | None = None,
        audio_path: str | None = None,
        sample_rate_hz: int | None = None,
    ) -> BackendResult:
        if not tokens:
            return BackendResult(
                words=[],
                model_id=_resolve_phoneme_model_id(),
                algorithm=_ALGO_MULTI,
                phonemes=[],
            )

        language = _normalize_language_code(language_code)
        words_with_phonemes = [_word_to_phonemes(token, language=language) for token in tokens]

        if language == _ENGLISH_CODE:
            word_result = self._word_backend.align_words(
                tokens,
                duration_sec,
                language_code=language_code,
                audio_path=audio_path,
                sample_rate_hz=sample_rate_hz,
            )
            phonemes = _align_phonemes_with_word_constraints(
                words=word_result.words,
                words_with_phonemes=words_with_phonemes,
            )
            model_id = f"{word_result.model_id}+{_resolve_phoneme_model_id()}"
            algorithm = f"{_ALGO_EN}+{word_result.algorithm}"
            return BackendResult(
                words=word_result.words,
                model_id=model_id,
                algorithm=algorithm,
                phonemes=phonemes,
            )

        phonemes = _align_phonemes_globally(
            words_with_phonemes=words_with_phonemes,
            duration_sec=duration_sec,
        )
        if not phonemes:
            fallback = self._word_backend.align_words(
                tokens,
                duration_sec,
                language_code=language_code,
                audio_path=audio_path,
                sample_rate_hz=sample_rate_hz,
            )
            return BackendResult(
                words=fallback.words,
                model_id=f"{_resolve_phoneme_model_id()}+{fallback.model_id}",
                algorithm=f"{_ALGO_MULTI_FALLBACK}+{fallback.algorithm}",
                phonemes=[],
            )

        words = _group_words_from_phonemes(
            tokens=tokens,
            phonemes=phonemes,
            duration_sec=duration_sec,
        )
        return BackendResult(
            words=words,
            model_id=_resolve_phoneme_model_id(),
            algorithm=_ALGO_MULTI,
            phonemes=phonemes,
        )


def _align_phonemes_with_word_constraints(
    *,
    words: list[WordAlignment],
    words_with_phonemes: list[_WordPhonemes],
) -> list[PhonemeAlignment]:
    output: list[PhonemeAlignment] = []
    for word_index, (word_alignment, word_phonemes) in enumerate(
        zip(words, words_with_phonemes, strict=True)
    ):
        phones = word_phonemes.phonemes or [word_phonemes.word]
        span = max(0.0, word_alignment.end_sec - word_alignment.start_sec)
        step = span / len(phones) if span > 0 else 0.0
        for phone_index, phone in enumerate(phones):
            start = round(word_alignment.start_sec + step * phone_index, 3)
            end = round(word_alignment.start_sec + step * (phone_index + 1), 3)
            if phone_index == len(phones) - 1:
                end = word_alignment.end_sec
            output.append(
                PhonemeAlignment(
                    phoneme=phone,
                    word_index=word_index,
                    start_sec=start,
                    end_sec=end,
                    confidence=round(max(0.6, word_alignment.confidence - 0.03), 3),
                )
            )
    return output


def _align_phonemes_globally(
    *,
    words_with_phonemes: list[_WordPhonemes],
    duration_sec: float,
) -> list[PhonemeAlignment]:
    output: list[PhonemeAlignment] = []
    total = sum(max(1, len(item.phonemes)) for item in words_with_phonemes)
    if total <= 0:
        return output

    step = duration_sec / total if duration_sec > 0 else 0.0
    cursor = 0
    for word_index, item in enumerate(words_with_phonemes):
        phones = item.phonemes or [item.word]
        for phone in phones:
            start = round(cursor * step, 3)
            cursor += 1
            end = round(cursor * step, 3)
            if cursor == total:
                end = duration_sec
            output.append(
                PhonemeAlignment(
                    phoneme=phone,
                    word_index=word_index,
                    start_sec=start,
                    end_sec=end,
                    confidence=0.7,
                )
            )
    return output


def _group_words_from_phonemes(
    *,
    tokens: list[str],
    phonemes: list[PhonemeAlignment],
    duration_sec: float,
) -> list[WordAlignment]:
    output: list[WordAlignment] = []
    by_word: dict[int, list[PhonemeAlignment]] = {}
    for phone in phonemes:
        by_word.setdefault(phone.word_index, []).append(phone)

    for word_index, token in enumerate(tokens):
        phones = by_word.get(word_index, [])
        if phones:
            start = phones[0].start_sec
            end = phones[-1].end_sec
            confidence = round(sum(phone.confidence for phone in phones) / len(phones), 3)
        else:
            start = 0.0
            end = 0.0
            confidence = 0.6
        if word_index == len(tokens) - 1:
            end = duration_sec
        output.append(
            WordAlignment(
                word=token,
                start_sec=start,
                end_sec=end,
                confidence=confidence,
            )
        )
    return output


def _word_to_phonemes(word: str, *, language: str | None) -> _WordPhonemes:
    letters = [char for char in word.casefold() if char.isalpha()]
    if language == _KOREAN_CODE:
        phones = _korean_word_to_ipa(word)
        return _WordPhonemes(word=word, phonemes=phones or letters)
    if language == _ENGLISH_CODE:
        phones = [_en_letter_to_ipa(letter) for letter in letters]
        return _WordPhonemes(word=word, phonemes=phones or [word])
    # Generic fallback: unicode grapheme-as-phone baseline.
    return _WordPhonemes(word=word, phonemes=letters or [word])


def _en_letter_to_ipa(letter: str) -> str:
    mapping = {
        "a": "ae",
        "b": "b",
        "c": "k",
        "d": "d",
        "e": "eh",
        "f": "f",
        "g": "g",
        "h": "h",
        "i": "ih",
        "j": "jh",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "o": "ow",
        "p": "p",
        "q": "k",
        "r": "r",
        "s": "s",
        "t": "t",
        "u": "uw",
        "v": "v",
        "w": "w",
        "x": "ks",
        "y": "y",
        "z": "z",
    }
    return mapping.get(letter, letter)


def _korean_word_to_ipa(word: str) -> list[str]:
    # Minimal deterministic fallback until g2pk2 mapping is added.
    output: list[str] = []
    for char in word:
        codepoint = ord(char)
        if 0xAC00 <= codepoint <= 0xD7A3:
            output.append("ko")
    return output


def _resolve_phoneme_model_id() -> str:
    return os.getenv("VOXALIGN_PHONEME_MODEL_ID", _PHONEME_MODEL_ID)


def _normalize_language_code(language_code: str | None) -> str | None:
    if language_code is None:
        return None
    cleaned = language_code.strip().casefold().replace("_", "-")
    if not cleaned:
        return None
    return cleaned.split("-")[0]
