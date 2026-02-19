"""CTC/trellis backend integration point.

This backend runs a real trellis/Viterbi decoder over simulated emissions.
It is structured so acoustic emissions can be swapped in later without
changing the pipeline contract.
"""

from __future__ import annotations

import math

from voxalign.align.backends.base import BackendName, BackendResult
from voxalign.align.trellis import (
    TokenFrameSpan,
    build_state_symbols,
    token_spans_from_state_path,
    viterbi_state_path,
)
from voxalign.models import WordAlignment

_MODEL_ID = "ctc-trellis-v0"
_ALGORITHM = "ctc-viterbi-simulated-emissions"
_FRAME_HZ = 100


class CtcTrellisBackend:
    """CTC-like backend with deterministic emission simulation."""

    name: BackendName = "ctc_trellis"

    def align_words(self, tokens: list[str], duration_sec: float) -> BackendResult:
        if not tokens:
            return BackendResult(words=[], model_id=_MODEL_ID, algorithm=_ALGORITHM)

        total_frames = max(1, int(round(duration_sec * _FRAME_HZ)))
        total_frames = max(total_frames, len(tokens) * 3)
        state_symbols = build_state_symbols(len(tokens))
        emissions = _simulate_emissions(
            frame_count=total_frames,
            token_count=len(tokens),
            state_symbols=state_symbols,
        )
        state_path = viterbi_state_path(emissions=emissions, state_symbols=state_symbols)
        spans = token_spans_from_state_path(
            state_path=state_path,
            state_symbols=state_symbols,
            token_count=len(tokens),
        )

        words: list[WordAlignment] = []
        for index, token in enumerate(tokens):
            span = spans[index]
            start = round(span.start_frame / _FRAME_HZ, 3)
            end = round(span.end_frame / _FRAME_HZ, 3)
            if index == len(tokens) - 1:
                end = duration_sec

            confidence = round(_span_confidence(emissions, token_id=index + 1, span=span), 3)
            words.append(
                WordAlignment(
                    word=token,
                    start_sec=start,
                    end_sec=end,
                    confidence=confidence,
                )
            )

        return BackendResult(words=words, model_id=_MODEL_ID, algorithm=_ALGORITHM)


def _simulate_emissions(
    frame_count: int,
    token_count: int,
    state_symbols: list[int],
) -> list[list[float]]:
    if token_count <= 0:
        return [[0.0] for _ in range(frame_count)]

    vocab_size = token_count + 1  # blank + ordered token symbols.
    state_count = len(state_symbols)
    emissions: list[list[float]] = []

    for frame in range(frame_count):
        if frame_count == 1:
            target_state = state_count - 1
        else:
            target_state = int(round((frame / (frame_count - 1)) * (state_count - 1)))
        target_symbol = state_symbols[target_state]

        logits = [-2.0] * vocab_size
        logits[0] = -0.2
        if target_symbol == 0:
            logits[0] = 1.4
        else:
            logits[target_symbol] = 2.0
            logits[0] = 0.3
            prev_symbol = max(1, target_symbol - 1)
            logits[prev_symbol] = max(logits[prev_symbol], 0.4)

        emissions.append(_log_softmax(logits))

    return emissions


def _log_softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_sum = sum(math.exp(v - max_logit) for v in logits)
    log_denom = max_logit + math.log(exp_sum)
    return [value - log_denom for value in logits]


def _span_confidence(
    emissions: list[list[float]],
    token_id: int,
    span: TokenFrameSpan,
) -> float:
    if span.end_frame <= span.start_frame:
        return 0.55

    values = [
        math.exp(emissions[frame][token_id]) for frame in range(span.start_frame, span.end_frame)
    ]
    mean_prob = sum(values) / len(values)
    return min(0.95, max(0.55, mean_prob))
