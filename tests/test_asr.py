from voxalign.asr.registry import transcribe_audio


def test_asr_auto_routes_to_parakeet_for_english() -> None:
    result = transcribe_audio(
        audio_path="sample.wav",
        language_code="en",
        backend="auto",
        verbatim=False,
    )
    assert result.backend == "parakeet"
    assert result.source == "simulated"
    assert result.model_id == "simulated-asr-v1"
    assert result.transcript == "hello world"


def test_asr_auto_routes_to_crisper_for_english_verbatim() -> None:
    result = transcribe_audio(
        audio_path="sample.wav",
        language_code="en",
        backend="auto",
        verbatim=True,
    )
    assert result.backend == "crisper_whisper"
    assert result.source == "simulated"
    assert result.transcript == "uh hello uh world"


def test_asr_auto_routes_to_whisper_for_non_english() -> None:
    result = transcribe_audio(
        audio_path="sample.wav",
        language_code="ko",
        backend="auto",
        verbatim=False,
    )
    assert result.backend == "whisper_large_v3"
    assert result.source == "simulated"
    assert result.language_code == "ko"
