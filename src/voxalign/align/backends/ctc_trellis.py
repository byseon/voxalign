"""CTC/trellis backend integration point.

This backend runs a trellis/Viterbi decoder over:
1) real emissions from a Hugging Face CTC model when available, or
2) deterministic simulated emissions as a fallback.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from voxalign.align.backends.base import BackendName, BackendResult
from voxalign.align.trellis import (
    TokenFrameSpan,
    build_state_symbols,
    token_spans_from_state_path,
    viterbi_state_path,
)
from voxalign.io import read_wav_audio, resample_linear
from voxalign.models import WordAlignment

_SIM_MODEL_ID = "ctc-trellis-v0"
_SIM_ALGORITHM = "ctc-viterbi-simulated-emissions"
_REAL_ALGORITHM = "ctc-viterbi-hf-emissions"
_DEFAULT_HF_MODEL = "facebook/wav2vec2-base-960h"
_FRAME_HZ = 100
_HF_CACHE: dict[str, _HfBundle] = {}


@dataclass(frozen=True)
class _EmissionPack:
    emissions: list[list[float]]
    token_ids: list[int]
    word_token_spans: list[tuple[int, int]]
    blank_id: int
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


class CtcTrellisBackend:
    """CTC-like backend with optional Hugging Face emission extraction."""

    name: BackendName = "ctc_trellis"

    def align_words(
        self,
        tokens: list[str],
        duration_sec: float,
        *,
        audio_path: str | None = None,
        sample_rate_hz: int | None = None,
    ) -> BackendResult:
        if not tokens:
            return BackendResult(words=[], model_id=_SIM_MODEL_ID, algorithm=_SIM_ALGORITHM)

        real_pack = _try_real_emissions(
            tokens=tokens,
            audio_path=audio_path,
            sample_rate_hz=sample_rate_hz,
        )
        if real_pack is None:
            emission_pack = _simulate_emission_pack(tokens=tokens, duration_sec=duration_sec)
        else:
            emission_pack = real_pack

        state_symbols = build_state_symbols(
            token_symbols=emission_pack.token_ids,
            blank_id=emission_pack.blank_id,
        )
        state_path = viterbi_state_path(
            emissions=emission_pack.emissions,
            state_symbols=state_symbols,
        )
        token_spans = token_spans_from_state_path(
            state_path=state_path,
            token_count=len(emission_pack.token_ids),
        )

        words = _word_alignments_from_token_spans(
            words=tokens,
            duration_sec=duration_sec,
            emissions=emission_pack.emissions,
            token_ids=emission_pack.token_ids,
            word_token_spans=emission_pack.word_token_spans,
            token_spans=token_spans,
        )

        return BackendResult(
            words=words,
            model_id=emission_pack.model_id,
            algorithm=emission_pack.algorithm,
        )


def _word_alignments_from_token_spans(
    words: list[str],
    duration_sec: float,
    emissions: list[list[float]],
    token_ids: list[int],
    word_token_spans: list[tuple[int, int]],
    token_spans: list[TokenFrameSpan],
) -> list[WordAlignment]:
    if not words:
        return []

    frame_count = max(1, len(emissions))
    frame_sec = duration_sec / frame_count if duration_sec > 0 else 0.0
    output: list[WordAlignment] = []

    for word_index, word in enumerate(words):
        token_start, token_end = word_token_spans[word_index]
        word_token_spans_slice = token_spans[token_start:token_end]
        word_token_ids = token_ids[token_start:token_end]

        valid_spans = [span for span in word_token_spans_slice if span.end_frame > span.start_frame]
        if valid_spans:
            start_frame = valid_spans[0].start_frame
            end_frame = valid_spans[-1].end_frame
        else:
            start_frame = 0
            end_frame = 0

        start_sec = round(start_frame * frame_sec, 3)
        end_sec = round(end_frame * frame_sec, 3)
        if word_index == len(words) - 1:
            end_sec = duration_sec

        confidence = round(
            _word_confidence(
                emissions=emissions,
                token_ids=word_token_ids,
                token_spans=word_token_spans_slice,
            ),
            3,
        )
        output.append(
            WordAlignment(
                word=word,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=confidence,
            )
        )
    return output


def _word_confidence(
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
        return 0.55
    mean_prob = sum(probabilities) / len(probabilities)
    return min(0.95, max(0.55, mean_prob))


def _simulate_emission_pack(tokens: list[str], duration_sec: float) -> _EmissionPack:
    token_ids = [index + 1 for index in range(len(tokens))]
    word_token_spans = [(index, index + 1) for index in range(len(tokens))]
    frame_count = max(1, int(round(duration_sec * _FRAME_HZ)))
    frame_count = max(frame_count, len(token_ids) * 3)

    state_symbols = build_state_symbols(token_ids, blank_id=0)
    emissions = _simulate_emissions(
        frame_count=frame_count,
        vocab_size=max(token_ids) + 1,
        state_symbols=state_symbols,
        blank_id=0,
    )
    return _EmissionPack(
        emissions=emissions,
        token_ids=token_ids,
        word_token_spans=word_token_spans,
        blank_id=0,
        model_id=_SIM_MODEL_ID,
        algorithm=_SIM_ALGORITHM,
    )


def _simulate_emissions(
    frame_count: int,
    vocab_size: int,
    state_symbols: list[int],
    blank_id: int,
) -> list[list[float]]:
    state_count = len(state_symbols)
    emissions: list[list[float]] = []

    for frame in range(frame_count):
        target_state = (
            state_count - 1
            if frame_count == 1
            else int(round((frame / (frame_count - 1)) * (state_count - 1)))
        )
        target_symbol = state_symbols[target_state]
        logits = [-2.0] * vocab_size
        logits[blank_id] = -0.2

        if target_symbol == blank_id:
            logits[blank_id] = 1.4
        else:
            logits[target_symbol] = 2.0
            logits[blank_id] = 0.3

        emissions.append(_log_softmax(logits))
    return emissions


def _log_softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_sum = sum(math.exp(v - max_logit) for v in logits)
    log_denom = max_logit + math.log(exp_sum)
    return [value - log_denom for value in logits]


def _try_real_emissions(
    tokens: list[str],
    audio_path: str | None,
    sample_rate_hz: int | None,
) -> _EmissionPack | None:
    if audio_path is None:
        return None
    if not _env_truthy("VOXALIGN_CTC_USE_HF", default=True):
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

    target_hz = bundle.target_sample_rate_hz
    if sample_rate_hz != target_hz:
        audio = resample_linear(audio, src_hz=sample_rate_hz, dst_hz=target_hz)
        sample_rate_hz = target_hz

    tokenization = _encode_words_for_ctc(
        tokens=tokens,
        tokenizer=bundle.processor.tokenizer,
        blank_id=bundle.blank_id,
    )
    if tokenization is None:
        return None
    token_ids, word_token_spans = tokenization

    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    inputs = bundle.processor(audio, sampling_rate=sample_rate_hz, return_tensors="pt")
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = bundle.model(**inputs).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1).cpu().tolist()

    if not log_probs:
        return None
    vocab_size = len(log_probs[0])
    if any(token_id >= vocab_size for token_id in token_ids):
        return None

    safe_model_id = bundle.model_id.replace("/", "-")
    return _EmissionPack(
        emissions=log_probs,
        token_ids=token_ids,
        word_token_spans=word_token_spans,
        blank_id=bundle.blank_id,
        model_id=f"hf-{safe_model_id}",
        algorithm=_REAL_ALGORITHM,
    )


def _load_hf_bundle() -> _HfBundle | None:
    model_id = os.getenv("VOXALIGN_CTC_MODEL_ID", _DEFAULT_HF_MODEL)
    device_pref = os.getenv("VOXALIGN_CTC_DEVICE", "auto")
    cache_key = f"{model_id}@{device_pref}"
    cached = _HF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
    except Exception:
        return None

    device = _resolve_torch_device(torch=torch, preference=device_pref)
    model = model.to(device)
    model.eval()

    target_hz = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    blank_id = getattr(processor.tokenizer, "pad_token_id", 0)
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


def _encode_words_for_ctc(
    tokens: list[str],
    tokenizer: Any,
    blank_id: int,
) -> tuple[list[int], list[tuple[int, int]]] | None:
    word_delimiter_id = getattr(tokenizer, "word_delimiter_token_id", None)
    if word_delimiter_id is not None:
        word_delimiter_id = int(word_delimiter_id)

    all_token_ids: list[int] = []
    word_spans: list[tuple[int, int]] = []

    for index, word in enumerate(tokens):
        encoded = tokenizer(word, add_special_tokens=False).input_ids
        token_ids = [int(token_id) for token_id in encoded if int(token_id) != blank_id]
        if not token_ids:
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if unk_id is None or int(unk_id) == blank_id:
                return None
            token_ids = [int(unk_id)]

        start = len(all_token_ids)
        all_token_ids.extend(token_ids)
        end = len(all_token_ids)
        word_spans.append((start, end))

        is_last = index == len(tokens) - 1
        if not is_last and word_delimiter_id is not None and word_delimiter_id != blank_id:
            all_token_ids.append(word_delimiter_id)

    if not all_token_ids:
        return None
    return all_token_ids, word_spans


def _env_truthy(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.casefold() not in {"0", "false", "no", "off"}
