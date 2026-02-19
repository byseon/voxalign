"""CTC/trellis backend integration point.

Current implementation is deterministic and model-free, but shaped to match
the interface needed for a real acoustic CTC trellis backend.
"""

from __future__ import annotations

from voxalign.align.backends.base import BackendName, BackendResult
from voxalign.models import WordAlignment

_MODEL_ID = "ctc-trellis-sim-v1"
_ALGORITHM = "ctc-trellis-simulated"


class CtcTrellisBackend:
    """Length-weighted frame allocation that mimics trellis-style behavior."""

    name: BackendName = "ctc_trellis"

    def align_words(self, tokens: list[str], duration_sec: float) -> BackendResult:
        if not tokens:
            return BackendResult(words=[], model_id=_MODEL_ID, algorithm=_ALGORITHM)

        total_frames = max(1, int(round(duration_sec * 100)))
        min_required_frames = len(tokens)
        if total_frames < min_required_frames:
            total_frames = min_required_frames

        weights = [max(1, len(token)) for token in tokens]
        frame_counts = _allocate_frames(total_frames=total_frames, weights=weights)

        words: list[WordAlignment] = []
        cursor_frames = 0
        for index, (token, frames) in enumerate(zip(tokens, frame_counts, strict=True)):
            start = round(cursor_frames / 100.0, 3)
            cursor_frames += frames
            end = round(cursor_frames / 100.0, 3)
            if index == len(tokens) - 1:
                end = duration_sec

            confidence = round(max(0.68, 0.93 - index * 0.012), 3)
            words.append(
                WordAlignment(
                    word=token,
                    start_sec=start,
                    end_sec=end,
                    confidence=confidence,
                )
            )

        return BackendResult(words=words, model_id=_MODEL_ID, algorithm=_ALGORITHM)


def _allocate_frames(total_frames: int, weights: list[int]) -> list[int]:
    counts = [1 for _ in weights]
    remaining = total_frames - len(weights)
    if remaining <= 0:
        return counts

    total_weight = sum(weights)
    extras = [int(remaining * (w / total_weight)) for w in weights]
    assigned = sum(extras)
    residual = remaining - assigned

    for i in range(len(extras)):
        counts[i] += extras[i]
    for i in range(residual):
        counts[i % len(counts)] += 1

    return counts
