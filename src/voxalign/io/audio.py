"""Audio metadata readers used by alignment timing logic."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioMetadata:
    """Minimal audio metadata required by the alignment pipeline."""

    duration_sec: float
    sample_rate_hz: int
    audio_format: str


def read_audio_metadata(audio_path: str | Path) -> AudioMetadata | None:
    """Read audio metadata for supported formats.

    Currently supports WAV files. Returns `None` for unsupported formats
    or when metadata cannot be parsed safely.
    """
    path = Path(audio_path)
    if not path.exists() or not path.is_file():
        return None

    suffix = path.suffix.casefold()
    if suffix in {".wav", ".wave"}:
        return _read_wav_metadata(path)

    return None


def _read_wav_metadata(path: Path) -> AudioMetadata | None:
    try:
        with wave.open(str(path), "rb") as handle:
            frame_count = handle.getnframes()
            sample_rate = handle.getframerate()
        if sample_rate <= 0:
            return None
    except (OSError, wave.Error):
        return None

    duration_sec = round(max(0.0, frame_count / sample_rate), 3)
    return AudioMetadata(
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate,
        audio_format="wav",
    )


def read_wav_audio(audio_path: str | Path) -> tuple[Any, int] | None:
    """Read WAV audio as mono float32 in range [-1, 1]."""
    path = Path(audio_path)
    if path.suffix.casefold() not in {".wav", ".wave"}:
        return None

    try:
        import numpy as np
    except ModuleNotFoundError:
        return None

    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except (OSError, wave.Error):
        return None

    if sample_rate <= 0 or channels <= 0:
        return None

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return data.astype(np.float32), sample_rate


def resample_linear(audio: Any, src_hz: int, dst_hz: int) -> Any:
    """Resample with linear interpolation."""
    if src_hz <= 0 or dst_hz <= 0:
        raise ValueError("Sample rates must be positive")
    if src_hz == dst_hz:
        return audio

    import numpy as np

    src_len = int(audio.shape[0])
    if src_len == 0:
        return audio

    duration_sec = src_len / src_hz
    dst_len = max(1, int(round(duration_sec * dst_hz)))
    src_x = np.linspace(0.0, duration_sec, num=src_len, endpoint=False)
    dst_x = np.linspace(0.0, duration_sec, num=dst_len, endpoint=False)
    resampled = np.interp(dst_x, src_x, audio).astype(np.float32)
    return resampled
