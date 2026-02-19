import wave
from pathlib import Path

from voxalign.io import read_audio_metadata, read_wav_audio, resample_linear


def test_read_audio_metadata_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    _write_wav(path=wav_path, sample_rate_hz=22050, duration_sec=0.5)

    metadata = read_audio_metadata(wav_path)

    assert metadata is not None
    assert metadata.audio_format == "wav"
    assert metadata.sample_rate_hz == 22050
    assert metadata.duration_sec == 0.5


def test_read_audio_metadata_missing_file() -> None:
    metadata = read_audio_metadata("does-not-exist.wav")
    assert metadata is None


def test_read_audio_metadata_unsupported_file(tmp_path: Path) -> None:
    txt_path = tmp_path / "note.txt"
    txt_path.write_text("hello", encoding="utf-8")
    metadata = read_audio_metadata(txt_path)
    assert metadata is None


def test_read_wav_audio_payload(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    _write_wav(path=wav_path, sample_rate_hz=16000, duration_sec=0.25)

    payload = read_wav_audio(wav_path)

    assert payload is not None
    audio, sample_rate_hz = payload
    assert sample_rate_hz == 16000
    assert int(audio.shape[0]) == 4000


def test_resample_linear() -> None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return

    src = np.linspace(-1.0, 1.0, num=8000, endpoint=False, dtype=np.float32)
    dst = resample_linear(src, src_hz=8000, dst_hz=16000)
    assert int(dst.shape[0]) == 16000


def _write_wav(path: Path, sample_rate_hz: int, duration_sec: float) -> None:
    frame_count = int(sample_rate_hz * duration_sec)
    silence_pcm16 = b"\x00\x00" * frame_count
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(silence_pcm16)
