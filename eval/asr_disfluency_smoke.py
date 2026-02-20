#!/usr/bin/env python3
"""Run a small ASR disfluency smoke check across backends."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from voxalign.asr.registry import transcribe_audio

_FILLERS_EN = {"uh", "um", "erm", "hmm", "mm"}


@dataclass(frozen=True)
class SmokeRow:
    backend: str
    model_id: str
    source: str
    transcript: str
    token_count: int
    filler_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASR disfluency smoke test.")
    parser.add_argument("--audio-path", required=True, help="Input WAV path")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument(
        "--backends",
        default="parakeet,crisper_whisper,whisper_large_v3",
        help="Comma-separated ASR backends",
    )
    parser.add_argument("--verbatim", action="store_true", help="Enable verbatim routing hint")
    return parser.parse_args()


def run_smoke(audio_path: str, language: str, backends: list[str], verbatim: bool) -> list[SmokeRow]:
    rows: list[SmokeRow] = []
    for backend in backends:
        result = transcribe_audio(
            audio_path=audio_path,
            language_code=language,
            backend=backend,  # type: ignore[arg-type]
            verbatim=verbatim,
        )
        tokens = result.transcript.casefold().split()
        rows.append(
            SmokeRow(
                backend=result.backend,
                model_id=result.model_id,
                source=result.source,
                transcript=result.transcript,
                token_count=len(tokens),
                filler_count=sum(1 for token in tokens if token in _FILLERS_EN),
            )
        )
    return rows


def main() -> int:
    args = parse_args()
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    rows = run_smoke(
        audio_path=args.audio_path,
        language=args.language,
        backends=backends,
        verbatim=args.verbatim,
    )
    payload = {
        "audio_path": args.audio_path,
        "language": args.language,
        "rows": [asdict(row) for row in rows],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
