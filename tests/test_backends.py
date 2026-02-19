from voxalign.align.backends import resolve_backend


def test_resolve_uniform_backend() -> None:
    backend = resolve_backend("uniform")
    result = backend.align_words(tokens=["hello", "world"], duration_sec=1.2)

    assert backend.name == "uniform"
    assert result.model_id == "baseline-rule-v1"
    assert result.algorithm == "uniform-token-distribution"
    assert len(result.words) == 2
    assert result.words[-1].end_sec == 1.2


def test_resolve_ctc_backend() -> None:
    backend = resolve_backend("ctc_trellis")
    result = backend.align_words(tokens=["hello", "world"], duration_sec=1.2)

    assert backend.name == "ctc_trellis"
    assert result.model_id == "ctc-trellis-v0"
    assert result.algorithm == "ctc-viterbi-simulated-emissions"
    assert len(result.words) == 2
    assert result.words[-1].end_sec == 1.2
