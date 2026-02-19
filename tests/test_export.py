import json
from pathlib import Path

from voxalign.core import run_alignment
from voxalign.io import to_json, write_json
from voxalign.models import AlignRequest


def test_to_json_and_write_json(tmp_path: Path) -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript="hello world",
            language="en",
        )
    )

    payload = json.loads(to_json(response))
    assert payload["metadata"]["language"] == "en"
    assert len(payload["words"]) == 2

    output_path = tmp_path / "out" / "alignment.json"
    write_json(response, output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["metadata"]["model_id"] == "baseline-rule-v1"
