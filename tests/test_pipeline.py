from voxalign.core import run_alignment
from voxalign.models import AlignRequest


def test_run_alignment_with_words() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript="hello multilingual world",
            language="en",
            include_phonemes=True,
        )
    )

    assert response.metadata.language == "en"
    assert response.metadata.duration_sec > 0.0
    assert len(response.words) == 3
    assert response.words[0].word == "hello"
    assert response.words[-1].end_sec == response.metadata.duration_sec
    assert response.phonemes


def test_run_alignment_without_phonemes() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript="hello world",
            include_phonemes=False,
        )
    )

    assert response.metadata.language == "und"
    assert len(response.words) == 2
    assert response.phonemes == []


def test_run_alignment_empty_word_list() -> None:
    response = run_alignment(
        AlignRequest(
            audio_path="audio.wav",
            transcript="!!! ??? ...",
            include_phonemes=True,
        )
    )

    assert response.metadata.duration_sec == 0.0
    assert response.words == []
    assert response.phonemes == []
