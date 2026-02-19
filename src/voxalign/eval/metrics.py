"""Benchmark metric helpers for alignment quality and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from voxalign.models import WordAlignment


@dataclass(frozen=True)
class ReferenceWord:
    """Reference word boundary annotation."""

    word: str
    start_sec: float
    end_sec: float


def compute_boundary_errors_ms(
    predicted_words: list[WordAlignment],
    reference_words: list[ReferenceWord],
) -> list[float]:
    """Compute absolute boundary errors in milliseconds for matched words by index."""
    matched_count = min(len(predicted_words), len(reference_words))
    errors_ms: list[float] = []
    for idx in range(matched_count):
        pred = predicted_words[idx]
        ref = reference_words[idx]
        errors_ms.append(abs(pred.start_sec - ref.start_sec) * 1000.0)
        errors_ms.append(abs(pred.end_sec - ref.end_sec) * 1000.0)
    return errors_ms


def summarize_metrics(
    boundary_errors_ms: list[float],
    *,
    total_runtime_sec: float,
    total_audio_sec: float,
    matched_words: int,
    reference_words: int,
) -> dict[str, float]:
    """Summarize benchmark metrics for release-gate reporting."""
    p50_ms = _percentile(boundary_errors_ms, 50.0)
    p90_ms = _percentile(boundary_errors_ms, 90.0)
    p95_ms = _percentile(boundary_errors_ms, 95.0)
    mae_ms = mean(boundary_errors_ms) if boundary_errors_ms else 0.0
    median_ms = median(boundary_errors_ms) if boundary_errors_ms else 0.0

    tolerance_20 = _rate_leq(boundary_errors_ms, 20.0)
    tolerance_50 = _rate_leq(boundary_errors_ms, 50.0)
    tolerance_100 = _rate_leq(boundary_errors_ms, 100.0)

    rtf = total_runtime_sec / total_audio_sec if total_audio_sec > 0 else 0.0
    throughput = total_audio_sec / total_runtime_sec if total_runtime_sec > 0 else 0.0
    coverage = matched_words / reference_words if reference_words > 0 else 0.0

    return {
        "word_boundary_mae_ms": round(mae_ms, 3),
        "word_boundary_median_ms": round(median_ms, 3),
        "word_boundary_p50_ms": round(p50_ms, 3),
        "word_boundary_p90_ms": round(p90_ms, 3),
        "word_boundary_p95_ms": round(p95_ms, 3),
        "tolerance_le_20ms": round(tolerance_20, 4),
        "tolerance_le_50ms": round(tolerance_50, 4),
        "tolerance_le_100ms": round(tolerance_100, 4),
        "rtf": round(rtf, 4),
        "throughput_x": round(throughput, 4),
        "matched_word_coverage": round(coverage, 4),
        "matched_words": float(matched_words),
        "reference_words": float(reference_words),
        "boundary_sample_count": float(len(boundary_errors_ms)),
        "total_runtime_sec": round(total_runtime_sec, 4),
        "total_audio_sec": round(total_audio_sec, 4),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]

    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = rank - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


def _rate_leq(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    count = sum(1 for value in values if value <= threshold)
    return count / len(values)
