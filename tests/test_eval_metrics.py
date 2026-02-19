import pytest

from voxalign.eval.metrics import ReferenceWord, compute_boundary_errors_ms, summarize_metrics
from voxalign.models import WordAlignment


def test_compute_boundary_errors_ms() -> None:
    predicted = [
        WordAlignment(word="hello", start_sec=0.0, end_sec=0.4, confidence=0.9),
        WordAlignment(word="world", start_sec=0.4, end_sec=1.0, confidence=0.9),
    ]
    reference = [
        ReferenceWord(word="hello", start_sec=0.02, end_sec=0.38),
        ReferenceWord(word="world", start_sec=0.45, end_sec=1.01),
    ]

    errors = compute_boundary_errors_ms(predicted, reference)
    assert len(errors) == 4
    assert errors[0] == pytest.approx(20.0)
    assert errors[1] == pytest.approx(20.0)


def test_summarize_metrics() -> None:
    summary = summarize_metrics(
        [10.0, 20.0, 40.0, 120.0],
        total_runtime_sec=2.0,
        total_audio_sec=4.0,
        matched_words=2,
        reference_words=3,
    )

    assert summary["word_boundary_mae_ms"] == 47.5
    assert summary["word_boundary_p95_ms"] >= summary["word_boundary_p90_ms"]
    assert summary["tolerance_le_20ms"] == 0.5
    assert summary["tolerance_le_50ms"] == 0.75
    assert summary["tolerance_le_100ms"] == 0.75
    assert summary["rtf"] == 0.5
    assert summary["throughput_x"] == 2.0
    assert summary["matched_word_coverage"] == 0.6667
