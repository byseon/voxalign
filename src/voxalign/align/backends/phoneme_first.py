"""Phoneme-first alignment backend.

Routing strategy:
- English: word boundaries from CTC backend (Parakeet default), then constrained
  phoneme timing within each word span.
- Other languages: phoneme-first timing over the full utterance, then derive
  word spans by grouping phonemes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from voxalign.align.backends.base import BackendName, BackendResult
from voxalign.align.backends.ctc_trellis import CtcTrellisBackend
from voxalign.align.trellis import (
    TokenFrameSpan,
    build_state_symbols,
    token_spans_from_state_path,
    viterbi_state_path,
)
from voxalign.io import read_wav_audio, resample_linear
from voxalign.models import PhonemeAlignment, WordAlignment

_PHONEME_MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
_ALGO_EN = "phoneme-first-en-word-ctc-then-ipa-constrained"
_ALGO_MULTI = "phoneme-first-multilingual-ipa-ctc"
_ALGO_MULTI_REAL = "phoneme-first-multilingual-ipa-ctc-hf-emissions"
_ALGO_MULTI_FALLBACK = "phoneme-first-multilingual-ipa-fallback-to-ctc-word"
_ENGLISH_CODE = "en"
_KOREAN_CODE = "ko"
_HF_CACHE: dict[str, _HfBundle] = {}


@dataclass(frozen=True)
class _WordPhonemes:
    word: str
    phonemes: list[str]


@dataclass(frozen=True)
class _PhonemeUnit:
    phoneme: str
    word_index: int


@dataclass(frozen=True)
class _PhonemePack:
    phonemes: list[PhonemeAlignment]
    model_id: str
    algorithm: str


@dataclass(frozen=True)
class _HfBundle:
    processor: Any
    model: Any
    target_sample_rate_hz: int
    blank_id: int
    device: str
    model_id: str


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

        real_pack = _try_real_phoneme_pack(
            words_with_phonemes=words_with_phonemes,
            duration_sec=duration_sec,
            audio_path=audio_path,
            sample_rate_hz=sample_rate_hz,
        )
        if real_pack is None:
            phonemes = _align_phonemes_globally(
                words_with_phonemes=words_with_phonemes,
                duration_sec=duration_sec,
            )
            model_id = _resolve_phoneme_model_id()
            algorithm = _ALGO_MULTI
        else:
            phonemes = real_pack.phonemes
            model_id = real_pack.model_id
            algorithm = real_pack.algorithm

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
                model_id=f"{model_id}+{fallback.model_id}",
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
            model_id=model_id,
            algorithm=algorithm,
            phonemes=phonemes,
        )


def _try_real_phoneme_pack(
    *,
    words_with_phonemes: list[_WordPhonemes],
    duration_sec: float,
    audio_path: str | None,
    sample_rate_hz: int | None,
) -> _PhonemePack | None:
    if audio_path is None:
        return None
    if not _env_truthy("VOXALIGN_PHONEME_USE_HF", default=False):
        return None

    wav_payload = read_wav_audio(audio_path)
    if wav_payload is None:
        return None
    audio, detected_sample_rate_hz = wav_payload
    if sample_rate_hz is None:
        sample_rate_hz = detected_sample_rate_hz

    bundle = _load_hf_bundle()
    if bundle is None:
        return None

    if sample_rate_hz != bundle.target_sample_rate_hz:
        audio = resample_linear(
            audio,
            src_hz=sample_rate_hz,
            dst_hz=bundle.target_sample_rate_hz,
        )
        sample_rate_hz = bundle.target_sample_rate_hz

    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    try:
        inputs = bundle.processor(audio, sampling_rate=sample_rate_hz, return_tensors="pt")
        inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = bundle.model(**inputs).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1).cpu().tolist()
    except Exception:
        return None

    if not log_probs:
        return None

    phone_units = _flatten_phone_units(words_with_phonemes)
    if not phone_units:
        return None

    tokenization = _encode_phone_units_for_ctc(
        phone_units=phone_units,
        tokenizer=bundle.processor.tokenizer,
        blank_id=bundle.blank_id,
    )
    if tokenization is None:
        return None
    token_ids, phone_token_spans = tokenization
    if any(token_id >= len(log_probs[0]) for token_id in token_ids):
        return None

    state_symbols = build_state_symbols(token_ids, blank_id=bundle.blank_id)
    state_path = viterbi_state_path(emissions=log_probs, state_symbols=state_symbols)
    token_spans = token_spans_from_state_path(state_path=state_path, token_count=len(token_ids))
    phoneme_alignments = _phoneme_alignments_from_token_spans(
        phone_units=phone_units,
        phone_token_spans=phone_token_spans,
        token_ids=token_ids,
        token_spans=token_spans,
        emissions=log_probs,
        duration_sec=duration_sec,
    )
    if not phoneme_alignments:
        return None

    safe_model_id = bundle.model_id.replace("/", "-")
    return _PhonemePack(
        phonemes=phoneme_alignments,
        model_id=f"hf-{safe_model_id}",
        algorithm=_ALGO_MULTI_REAL,
    )


def _load_hf_bundle() -> _HfBundle | None:
    model_id = _resolve_phoneme_model_id()
    device_pref = os.getenv("VOXALIGN_PHONEME_DEVICE", "auto")
    cache_key = f"{model_id}@{device_pref}"
    cached = _HF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(model_id)
    except Exception:
        return None

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None

    device = _resolve_torch_device(torch=torch, preference=device_pref)
    model = model.to(device)
    model.eval()

    feature_extractor = getattr(processor, "feature_extractor", None)
    target_hz = int(getattr(feature_extractor, "sampling_rate", 16000))
    blank_id = getattr(tokenizer, "pad_token_id", 0)
    if blank_id is None:
        blank_id = 0

    bundle = _HfBundle(
        processor=processor,
        model=model,
        target_sample_rate_hz=target_hz,
        blank_id=int(blank_id),
        device=device,
        model_id=model_id,
    )
    _HF_CACHE[cache_key] = bundle
    return bundle


def _resolve_torch_device(torch: Any, preference: str) -> str:
    pref = preference.casefold()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _flatten_phone_units(words_with_phonemes: list[_WordPhonemes]) -> list[_PhonemeUnit]:
    units: list[_PhonemeUnit] = []
    for word_index, item in enumerate(words_with_phonemes):
        phones = item.phonemes or [item.word]
        for phone in phones:
            units.append(_PhonemeUnit(phoneme=phone, word_index=word_index))
    return units


def _encode_phone_units_for_ctc(
    *,
    phone_units: list[_PhonemeUnit],
    tokenizer: Any,
    blank_id: int,
) -> tuple[list[int], list[tuple[int, int]]] | None:
    token_ids: list[int] = []
    phone_token_spans: list[tuple[int, int]] = []
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for unit in phone_units:
        normalized_phone = _normalize_phone_text(unit.phoneme)
        encoded = tokenizer(normalized_phone, add_special_tokens=False).input_ids
        ids = [int(token_id) for token_id in encoded if int(token_id) != blank_id]
        if not ids:
            if unk_id is None or int(unk_id) == blank_id:
                return None
            ids = [int(unk_id)]

        start = len(token_ids)
        token_ids.extend(ids)
        end = len(token_ids)
        phone_token_spans.append((start, end))

    if not token_ids:
        return None
    return token_ids, phone_token_spans


def _normalize_phone_text(phone: str) -> str:
    return phone.strip()


def _phoneme_alignments_from_token_spans(
    *,
    phone_units: list[_PhonemeUnit],
    phone_token_spans: list[tuple[int, int]],
    token_ids: list[int],
    token_spans: list[TokenFrameSpan],
    emissions: list[list[float]],
    duration_sec: float,
) -> list[PhonemeAlignment]:
    if not emissions:
        return []
    frame_count = max(1, len(emissions))
    frame_sec = duration_sec / frame_count if duration_sec > 0 else 0.0
    output: list[PhonemeAlignment] = []

    for index, unit in enumerate(phone_units):
        token_start, token_end = phone_token_spans[index]
        span_slice = token_spans[token_start:token_end]
        id_slice = token_ids[token_start:token_end]
        valid_spans = [span for span in span_slice if span.end_frame > span.start_frame]
        if valid_spans:
            start_frame = valid_spans[0].start_frame
            end_frame = valid_spans[-1].end_frame
        else:
            start_frame = 0
            end_frame = 0

        start_sec = round(start_frame * frame_sec, 3)
        end_sec = round(end_frame * frame_sec, 3)
        if index == len(phone_units) - 1:
            end_sec = duration_sec
        confidence = round(_token_span_confidence(emissions, id_slice, span_slice), 3)

        output.append(
            PhonemeAlignment(
                phoneme=unit.phoneme,
                word_index=unit.word_index,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=confidence,
            )
        )
    return output


def _token_span_confidence(
    emissions: list[list[float]],
    token_ids: list[int],
    token_spans: list[TokenFrameSpan],
) -> float:
    probabilities: list[float] = []
    for token_id, span in zip(token_ids, token_spans, strict=True):
        if span.end_frame <= span.start_frame:
            continue
        for frame in range(span.start_frame, span.end_frame):
            probabilities.append(math.exp(emissions[frame][token_id]))
    if not probabilities:
        return 0.6
    mean_prob = sum(probabilities) / len(probabilities)
    return min(0.95, max(0.6, mean_prob))


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


def _env_truthy(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.casefold() not in {"0", "false", "no", "off"}
