from fastapi.testclient import TestClient

from voxalign.api import create_app


def test_health_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert payload["env"] == "dev"


def test_align_endpoint() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/align",
        json={
            "audio_path": "sample.wav",
            "transcript": "hello world",
            "language": "en",
            "include_phonemes": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["language"] == "en"
    assert payload["metadata"]["alignment_backend"] == "uniform"
    assert payload["metadata"]["normalizer_id"] == "english-basic-v1"
    assert payload["metadata"]["token_count"] == 2
    assert payload["metadata"]["timing_source"] == "heuristic"
    assert payload["metadata"]["transcript_source"] == "provided"
    assert payload["metadata"]["asr_backend"] is None
    assert payload["metadata"]["asr_model_id"] is None
    assert payload["metadata"]["model_id"] == "baseline-rule-v1"
    assert len(payload["words"]) == 2
    assert len(payload["phonemes"]) >= 2


def test_align_endpoint_ctc_backend() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/align",
        json={
            "audio_path": "sample.wav",
            "transcript": "hello world",
            "language": "en",
            "backend": "ctc_trellis",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["alignment_backend"] == "ctc_trellis"
    assert payload["metadata"]["model_id"] == "ctc-trellis-v0"


def test_align_endpoint_phoneme_first_backend() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/align",
        json={
            "audio_path": "sample.wav",
            "transcript": "hello world",
            "language": "en",
            "backend": "phoneme_first",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["alignment_backend"] == "phoneme_first"
    assert "facebook/wav2vec2-xlsr-53-espeak-cv-ft" in payload["metadata"]["model_id"]


def test_align_endpoint_with_asr_auto() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/align",
        json={
            "audio_path": "sample.wav",
            "language": "en",
            "backend": "phoneme_first",
            "asr": "auto",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["transcript_source"] == "asr"
    assert payload["metadata"]["asr_backend"] == "parakeet"
    assert payload["metadata"]["asr_model_id"] == "simulated-asr-v1"


def test_align_endpoint_without_transcript_and_disabled_asr() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/align",
        json={
            "audio_path": "sample.wav",
            "language": "en",
            "backend": "uniform",
            "asr": "disabled",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "transcript is required" in payload["detail"]
