import json
from pathlib import Path

from voxalign.cli import main


def test_cli_no_args_shows_help(capsys) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage: voxalign" in captured.out


def test_cli_align_returns_json(capsys) -> None:
    exit_code = main(["align", "sample.wav", "hello world", "--language", "en"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["metadata"]["language"] == "en"
    assert payload["metadata"]["alignment_backend"] == "uniform"
    assert payload["metadata"]["normalizer_id"] == "english-basic-v1"
    assert payload["metadata"]["timing_source"] == "heuristic"
    assert payload["words"][0]["word"] == "hello"
    assert payload["phonemes"]


def test_cli_align_writes_json_file(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "alignment.json"
    exit_code = main(
        [
            "align",
            "sample.wav",
            "hello world",
            "--language",
            "en",
            "--no-phonemes",
            "-o",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["phonemes"] == []
    assert "Wrote alignment JSON" in captured.out


def test_cli_align_ctc_backend(capsys) -> None:
    exit_code = main(
        ["align", "sample.wav", "hello world", "--language", "en", "--backend", "ctc_trellis"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["metadata"]["alignment_backend"] == "ctc_trellis"
    assert payload["metadata"]["model_id"] == "ctc-trellis-v0"
