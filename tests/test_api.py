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
    assert payload["metadata"]["normalizer_id"] == "english-basic-v1"
    assert payload["metadata"]["token_count"] == 2
    assert payload["metadata"]["timing_source"] == "heuristic"
    assert payload["metadata"]["model_id"] == "baseline-rule-v1"
    assert len(payload["words"]) == 2
    assert len(payload["phonemes"]) >= 2
