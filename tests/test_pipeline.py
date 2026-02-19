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
    assert response.metadata.normalizer_id == "english-basic-v1"
    assert response.metadata.token_count == 3
    assert response.metadata.timing_source == "heuristic"
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
    assert response.metadata.normalizer_id == "generic-unicode-v1"
    assert response.metadata.token_count == 2
    assert response.metadata.timing_source == "heuristic"
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
    assert response.metadata.token_count == 0
    assert response.metadata.timing_source == "heuristic"
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
    assert response.metadata.duration_sec == 1.0
    assert response.metadata.sample_rate_hz == 16000
    assert response.words[-1].end_sec == 1.0


def _write_wav(path: Path, sample_rate_hz: int, duration_sec: float) -> None:
    frame_count = int(sample_rate_hz * duration_sec)
    silence_pcm16 = b"\x00\x00" * frame_count
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(silence_pcm16)
