"""Uniform-distribution baseline backend."""

from __future__ import annotations

from voxalign.align.backends.base import BackendName, BackendResult
from voxalign.models import WordAlignment

_MODEL_ID = "baseline-rule-v1"
_ALGORITHM = "uniform-token-distribution"


class UniformBackend:
    """Evenly distribute token spans over total duration."""

    name: BackendName = "uniform"

    def align_words(self, tokens: list[str], duration_sec: float) -> BackendResult:
        if not tokens:
            return BackendResult(words=[], model_id=_MODEL_ID, algorithm=_ALGORITHM)

        step = duration_sec / len(tokens)
        words: list[WordAlignment] = []
        for index, token in enumerate(tokens):
            start = round(step * index, 3)
            end = round(step * (index + 1), 3)
            if index == len(tokens) - 1:
                end = duration_sec
            confidence = round(max(0.75, 0.98 - index * 0.01), 3)
            words.append(
                WordAlignment(
                    word=token,
                    start_sec=start,
                    end_sec=end,
                    confidence=confidence,
                )
            )

        return BackendResult(words=words, model_id=_MODEL_ID, algorithm=_ALGORITHM)
