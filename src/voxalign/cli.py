"""CLI entrypoint for voxalign."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from voxalign.config import load_config
from voxalign.core import run_alignment
from voxalign.io import to_json, write_json
from voxalign.models import AlignRequest


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="voxalign",
        description="Multilingual forced aligner.",
    )
    subparsers = parser.add_subparsers(dest="command")

    align = subparsers.add_parser("align", help="Align transcript to audio")
    align.add_argument("audio_path", help="Path to input audio file")
    align.add_argument("transcript", help="Transcript text or transcript file path")
    align.add_argument("--language", default="auto", help="Language code (default: auto)")
    align.add_argument(
        "--sample-rate-hz",
        type=int,
        default=None,
        help="Input audio sample rate for metadata (optional)",
    )
    align.add_argument(
        "--no-phonemes",
        action="store_true",
        help="Disable phoneme-level alignment output",
    )
    align.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path. If omitted, prints to stdout.",
    )

    serve = subparsers.add_parser("serve", help="Run the voxalign HTTP API")
    serve.add_argument("--host", default=None, help="Override API host")
    serve.add_argument("--port", type=int, default=None, help="Override API port")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "align":
        request = AlignRequest(
            audio_path=args.audio_path,
            transcript=args.transcript,
            language=args.language,
            include_phonemes=not args.no_phonemes,
            sample_rate_hz=args.sample_rate_hz,
        )
        response = run_alignment(request)
        if args.output:
            write_json(response, args.output)
            print(f"Wrote alignment JSON to {args.output}")
            return 0
        print(to_json(response))
        return 0

    if args.command == "serve":
        try:
            import uvicorn
        except ModuleNotFoundError:
            print(
                "`voxalign serve` requires uvicorn. Install project dependencies first.",
                file=sys.stderr,
            )
            return 1

        config = load_config()
        host = args.host or config.api_host
        port = args.port or config.api_port
        uvicorn.run(
            "voxalign.api:app",
            host=host,
            port=port,
            workers=config.workers,
            reload=False,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
