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


def test_resolve_phoneme_first_backend_english() -> None:
    backend = resolve_backend("phoneme_first")
    result = backend.align_words(tokens=["hello", "world"], duration_sec=1.2, language_code="en")

    assert backend.name == "phoneme_first"
    assert "facebook/wav2vec2-xlsr-53-espeak-cv-ft" in result.model_id
    assert "phoneme-first-en-word-ctc-then-ipa-constrained" in result.algorithm
    assert result.phonemes is not None
    assert len(result.phonemes) >= 2
    assert result.words[-1].end_sec == 1.2


def test_resolve_phoneme_first_backend_multilingual() -> None:
    backend = resolve_backend("phoneme_first")
    result = backend.align_words(
        tokens=["안녕하세요", "반갑습니다"], duration_sec=1.2, language_code="ko"
    )

    assert backend.name == "phoneme_first"
    assert result.model_id == "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    assert result.algorithm == "phoneme-first-multilingual-ipa-ctc"
    assert result.phonemes is not None
    assert len(result.phonemes) >= 2
    assert result.words[-1].end_sec == 1.2
