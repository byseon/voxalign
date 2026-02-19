from voxalign.align.backends import ctc_trellis


def test_language_bucket_resolution() -> None:
    assert ctc_trellis._language_bucket("en") == "en"
    assert ctc_trellis._language_bucket("fr") == "eu"
    assert ctc_trellis._language_bucket("ko") == "ko"
    assert ctc_trellis._language_bucket("ja") == "global"
    assert ctc_trellis._language_bucket(None) == "global"


def test_model_resolution_with_bucket_overrides(monkeypatch) -> None:
    monkeypatch.delenv("VOXALIGN_CTC_MODEL_ID", raising=False)
    monkeypatch.setenv("VOXALIGN_CTC_MODEL_EN", "org/model-en")
    monkeypatch.setenv("VOXALIGN_CTC_MODEL_EU", "org/model-eu")
    monkeypatch.setenv("VOXALIGN_CTC_MODEL_KO", "org/model-ko")
    monkeypatch.setenv("VOXALIGN_CTC_MODEL_DEFAULT", "org/model-global")

    assert ctc_trellis._resolve_model_id("en") == "org/model-en"
    assert ctc_trellis._resolve_model_id("de") == "org/model-eu"
    assert ctc_trellis._resolve_model_id("ko") == "org/model-ko"
    assert ctc_trellis._resolve_model_id("ja") == "org/model-global"


def test_model_resolution_with_global_override(monkeypatch) -> None:
    monkeypatch.setenv("VOXALIGN_CTC_MODEL_ID", "org/model-all")
    assert ctc_trellis._resolve_model_id("en") == "org/model-all"
    assert ctc_trellis._resolve_model_id("de") == "org/model-all"
