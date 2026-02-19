"""Alignment output serializers."""

from __future__ import annotations

from pathlib import Path

from voxalign.models import AlignResponse


def to_json(response: AlignResponse) -> str:
    """Serialize an alignment response to formatted JSON."""
    return response.model_dump_json(indent=2)


def write_json(response: AlignResponse, output_path: str | Path) -> None:
    """Write alignment response JSON to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(response) + "\n", encoding="utf-8")
