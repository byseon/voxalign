import wave
from pathlib import Path

from voxalign.core import run_alignment
from voxalign.models import AlignRequest


def test_run_alignment_with_words() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript="Hello, multilingual world!!!",
            language="en",
            include_phonemes=True,
        )
    )

    assert response.metadata.language == "en"
    assert response.metadata.alignment_backend == "uniform"
    assert response.metadata.normalizer_id == "english-basic-v1"
    assert response.metadata.token_count == 3
    assert response.metadata.timing_source == "heuristic"
    assert response.metadata.transcript_source == "provided"
    assert response.metadata.asr_backend is None
    assert response.metadata.asr_model_id is None
    assert response.metadata.license_warning is None
    assert response.metadata.duration_sec > 0.0
    assert len(response.words) == 3
    assert response.words[0].word == "hello"
    assert response.words[-1].end_sec == response.metadata.duration_sec
    assert response.phonemes


def test_run_alignment_without_phonemes() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript="hello world",
            include_phonemes=False,
        )
    )

    assert response.metadata.language == "und"
    assert response.metadata.alignment_backend == "uniform"
    assert response.metadata.normalizer_id == "generic-unicode-v1"
    assert response.metadata.token_count == 2
    assert response.metadata.timing_source == "heuristic"
    assert response.metadata.transcript_source == "provided"
    assert response.metadata.license_warning is None
    assert len(response.words) == 2
    assert response.phonemes == []


def test_run_alignment_empty_word_list() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript="!!! ??? ...",
            include_phonemes=True,
        )
    )

    assert response.metadata.duration_sec == 0.0
    assert response.metadata.alignment_backend == "uniform"
    assert response.metadata.token_count == 0
    assert response.metadata.timing_source == "heuristic"
    assert response.metadata.transcript_source == "provided"
    assert response.metadata.license_warning is None
    assert response.words == []
    assert response.phonemes == []


def test_run_alignment_uses_wav_duration(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    _write_wav(path=wav_path, sample_rate_hz=16000, duration_sec=1.0)

    response = run_alignment(
        AlignRequest(
            audio_path=str(wav_path),
            transcript="hello world",
            language="en",
            include_phonemes=True,
        )
    )

    assert response.metadata.timing_source == "audio"
    assert response.metadata.alignment_backend == "uniform"
    assert response.metadata.duration_sec == 1.0
    assert response.metadata.sample_rate_hz == 16000
    assert response.metadata.transcript_source == "provided"
    assert response.metadata.license_warning is None
    assert response.words[-1].end_sec == 1.0


def test_run_alignment_ctc_backend_selection() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript="hello world",
            language="en",
            backend="ctc_trellis",
        )
    )

    assert response.metadata.alignment_backend == "ctc_trellis"
    assert response.metadata.model_id == "ctc-trellis-v0"
    assert response.metadata.algorithm == "ctc-viterbi-simulated-emissions"
    assert response.metadata.transcript_source == "provided"
    assert response.metadata.license_warning is None


def test_run_alignment_phoneme_first_english() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript="hello world",
            language="en",
            backend="phoneme_first",
            include_phonemes=True,
        )
    )

    assert response.metadata.alignment_backend == "phoneme_first"
    assert "facebook/wav2vec2-xlsr-53-espeak-cv-ft" in response.metadata.model_id
    assert "phoneme-first-en-word-ctc-then-ipa-constrained" in response.metadata.algorithm
    assert response.metadata.transcript_source == "provided"
    assert response.metadata.license_warning is None
    assert response.phonemes
    assert response.phonemes[0].word_index == 0


def test_run_alignment_phoneme_first_korean() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript="안녕하세요 반갑습니다",
            language="ko",
            backend="phoneme_first",
            include_phonemes=True,
        )
    )

    assert response.metadata.alignment_backend == "phoneme_first"
    assert response.metadata.model_id == "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    assert response.metadata.algorithm == "phoneme-first-multilingual-ipa-ctc"
    assert response.metadata.transcript_source == "provided"
    assert response.metadata.license_warning is None
    assert len(response.words) == 2
    assert response.phonemes


def test_run_alignment_with_asr_auto_when_transcript_missing() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript=None,
            language="en",
            backend="phoneme_first",
            asr="auto",
        )
    )

    assert response.metadata.language == "en"
    assert response.metadata.transcript_source == "asr"
    assert response.metadata.asr_backend == "parakeet"
    assert response.metadata.asr_model_id == "simulated-asr-v1"
    assert response.metadata.license_warning is None
    assert response.words


def test_run_alignment_with_asr_auto_for_french_routes_to_parakeet_tdt() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript=None,
            language="fr",
            backend="phoneme_first",
            asr="auto",
        )
    )

    assert response.metadata.language == "fr"
    assert response.metadata.transcript_source == "asr"
    assert response.metadata.asr_backend == "parakeet_tdt"
    assert response.metadata.asr_model_id == "simulated-asr-v1"
    assert response.metadata.license_warning is None
    assert response.words


def test_run_alignment_with_asr_auto_verbatim_routes_to_crisper() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript=None,
            language="en",
            backend="phoneme_first",
            asr="auto",
            verbatim=True,
        )
    )

    assert response.metadata.transcript_source == "asr"
    assert response.metadata.asr_backend == "crisper_whisper"
    assert response.metadata.license_warning is not None
    assert "CC BY-NC 4.0" in response.metadata.license_warning
    assert response.words[0].word == "uh"


def test_run_alignment_without_transcript_and_disabled_asr_raises() -> None:
    try:
        run_alignment(
            AlignRequest(
                audio_path="audio.wav",
                transcript=None,
                language="en",
                asr="disabled",
            )
        )
    except ValueError as exc:
        assert "transcript is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def _write_wav(path: Path, sample_rate_hz: int, duration_sec: float) -> None:
    frame_count = int(sample_rate_hz * duration_sec)
    silence_pcm16 = b"\x00\x00" * frame_count
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(silence_pcm16)
