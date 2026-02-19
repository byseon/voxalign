"""Audio metadata readers used by alignment timing logic."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


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
