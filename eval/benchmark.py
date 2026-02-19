#!/usr/bin/env python3
"""Run alignment benchmark and write release-gate artifacts.

Manifest format (JSONL):
{
  "id": "utt-001",
  "audio_path": "path/to/audio.wav",
  "transcript": "hello world",
  "language": "en",
  "reference_words": [
    {"word": "hello", "start_sec": 0.02, "end_sec": 0.41},
    {"word": "world", "start_sec": 0.43, "end_sec": 0.98}
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voxalign.core import run_alignment
from voxalign.eval import ReferenceWord, compute_boundary_errors_ms, summarize_metrics
from voxalign.models import AlignRequest


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    audio_path: str
    transcript: str
    language: str
    reference_words: list[ReferenceWord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VoxAlign benchmark and save artifacts.")
    parser.add_argument("--manifest", required=True, help="Path to benchmark JSONL manifest")
    parser.add_argument("--output-root", default="eval/runs", help="Artifact root directory")
    parser.add_argument(
        "--backend",
        default="ctc_trellis",
        choices=["uniform", "ctc_trellis", "phoneme_first"],
        help="Alignment backend",
    )
    parser.add_argument(
        "--include-phonemes",
        action="store_true",
        help="Include phoneme output during benchmark runs",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line_num, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        case_id = str(payload.get("id") or f"line-{line_num}")
        references = [
            ReferenceWord(
                word=str(item["word"]),
                start_sec=float(item["start_sec"]),
                end_sec=float(item["end_sec"]),
            )
            for item in payload["reference_words"]
        ]
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                audio_path=str(payload["audio_path"]),
                transcript=str(payload["transcript"]),
                language=str(payload.get("language", "auto")),
                reference_words=references,
            )
        )
    return cases


def run_benchmark(cases: list[BenchmarkCase], *, backend: str, include_phonemes: bool) -> dict[str, Any]:
    boundary_errors_ms: list[float] = []
    rows: list[dict[str, Any]] = []
    total_runtime_sec = 0.0
    total_audio_sec = 0.0
    matched_words = 0
    reference_words = 0

    for case in cases:
        started = time.perf_counter()
        response = run_alignment(
            AlignRequest(
                audio_path=case.audio_path,
                transcript=case.transcript,
                language=case.language,
                backend=backend,
                include_phonemes=include_phonemes,
            )
        )
        elapsed = time.perf_counter() - started

        errors = compute_boundary_errors_ms(response.words, case.reference_words)
        boundary_errors_ms.extend(errors)
        total_runtime_sec += elapsed
        total_audio_sec += response.metadata.duration_sec
        matched_word_count = min(len(response.words), len(case.reference_words))
        matched_words += matched_word_count
        reference_words += len(case.reference_words)

        rows.append(
            {
                "case_id": case.case_id,
                "language": case.language,
                "backend": backend,
                "runtime_sec": round(elapsed, 6),
                "audio_sec": round(response.metadata.duration_sec, 3),
                "matched_words": matched_word_count,
                "reference_words": len(case.reference_words),
                "mean_boundary_error_ms": round(sum(errors) / len(errors), 3) if errors else 0.0,
                "p95_boundary_error_ms": round(_percentile(errors, 95.0), 3) if errors else 0.0,
                "model_id": response.metadata.model_id,
                "timing_source": response.metadata.timing_source,
            }
        )

    summary = summarize_metrics(
        boundary_errors_ms,
        total_runtime_sec=total_runtime_sec,
        total_audio_sec=total_audio_sec,
        matched_words=matched_words,
        reference_words=reference_words,
    )
    return {"summary": summary, "rows": rows}


def write_artifacts(
    output_root: Path,
    *,
    backend: str,
    manifest: Path,
    result: dict[str, Any],
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    git_sha = _git_sha()
    out_dir = output_root / f"{timestamp}_{git_sha[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "backend": backend,
        "manifest_path": str(manifest),
        "summary": result["summary"],
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (out_dir / "per_utterance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0].keys()) if result["rows"] else [])
        if result["rows"]:
            writer.writeheader()
            writer.writerows(result["rows"])

    command_payload = {
        "command": " ".join(_shell_argv()),
    }
    (out_dir / "run.json").write_text(
        json.dumps(command_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_dir


def _git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _shell_argv() -> list[str]:
    import sys

    return [sys.executable, *sys.argv]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest)
    cases = load_manifest(manifest)
    result = run_benchmark(cases, backend=args.backend, include_phonemes=args.include_phonemes)
    out_dir = write_artifacts(
        Path(args.output_root),
        backend=args.backend,
        manifest=manifest,
        result=result,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Artifacts written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
