from voxalign.align.backends import ctc_trellis


def test_language_bucket_resolution() -> None:
    assert ctc_trellis._language_bucket("en") == "en"
    assert ctc_trellis._language_bucket("en-US") == "en"
    assert ctc_trellis._language_bucket("fr") == "eu"
    assert ctc_trellis._language_bucket("fr-FR") == "eu"
    assert ctc_trellis._language_bucket("ko") == "ko"
    assert ctc_trellis._language_bucket("ko-KR") == "ko"
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


def test_model_resolution_defaults(monkeypatch) -> None:
    monkeypatch.delenv("VOXALIGN_CTC_MODEL_ID", raising=False)
    monkeypatch.delenv("VOXALIGN_CTC_MODEL_EN", raising=False)
    monkeypatch.delenv("VOXALIGN_CTC_MODEL_EU", raising=False)
    monkeypatch.delenv("VOXALIGN_CTC_MODEL_KO", raising=False)
    monkeypatch.delenv("VOXALIGN_CTC_MODEL_DEFAULT", raising=False)

    assert ctc_trellis._resolve_model_id("en") == "nvidia/parakeet-ctc-1.1b"
    assert ctc_trellis._resolve_model_id("de") == "facebook/mms-1b-all"
    assert ctc_trellis._resolve_model_id("ko") == "facebook/mms-1b-all"
    assert ctc_trellis._resolve_model_id("ja") == "facebook/mms-1b-all"


def test_mms_adapter_resolution() -> None:
    assert ctc_trellis._resolve_adapter_language("facebook/mms-1b-all", "en") == "eng"
    assert ctc_trellis._resolve_adapter_language("facebook/mms-1b-all", "fr-FR") == "fra"
    assert ctc_trellis._resolve_adapter_language("facebook/mms-1b-all", "ko") == "kor"
    assert ctc_trellis._resolve_adapter_language("facebook/mms-1b-all", "ja") is None
    assert ctc_trellis._resolve_adapter_language("nvidia/parakeet-ctc-1.1b", "en") is None
