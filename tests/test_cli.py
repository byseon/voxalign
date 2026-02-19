from voxalign.cli import main


def test_cli_no_args_shows_help(capsys) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage: voxalign" in captured.out


def test_cli_align_placeholder(capsys) -> None:
    exit_code = main(["align", "sample.wav", "hello world", "--language", "en"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "not implemented yet" in captured.out
    assert "audio_path=sample.wav" in captured.out
    assert "language=en" in captured.out
