"""ASR backend routing and optional HF runtime integration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from voxalign.asr.base import AsrBackendName, AsrResult
from voxalign.io import read_wav_audio, resample_linear

_DEFAULT_PARAKEET_MODEL_ID = "nvidia/parakeet-ctc-1.1b"
_DEFAULT_PARAKEET_TDT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
_DEFAULT_CRISPER_MODEL_ID = "nyrahealth/CrisperWhisper"
_DEFAULT_WHISPER_MODEL_ID = "openai/whisper-large-v3"
_SIM_MODEL_ID = "simulated-asr-v1"
_SPACES_RE = re.compile(r"\s+")
_PARAKEET_TDT_EU_CODES = {
    "bg",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "es",
    "et",
    "eu",
    "fi",
    "fr",
    "ga",
    "gl",
    "hr",
    "hu",
    "is",
    "it",
    "lt",
    "lv",
    "mk",
    "mt",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "sq",
    "sr",
    "sk",
    "sl",
    "sv",
}

_HF_CTC_CACHE: dict[str, _CtcBundle] = {}
_HF_PIPELINE_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class _CtcBundle:
    processor: Any
    model: Any
    target_sample_rate_hz: int
    device: str


def transcribe_audio(
    *,
    audio_path: str,
    language_code: str | None,
    backend: AsrBackendName,
    verbatim: bool,
    sample_rate_hz: int | None = None,
) -> AsrResult:
    """Resolve and run ASR backend."""
    selected = _resolve_backend_choice(
        requested_backend=backend,
        language_code=language_code,
        verbatim=verbatim,
    )

    if selected == "parakeet":
        return _transcribe_parakeet(
            audio_path=audio_path,
            language_code=language_code,
            sample_rate_hz=sample_rate_hz,
        )
    if selected == "parakeet_tdt":
        return _transcribe_pipeline_asr(
            audio_path=audio_path,
            language_code=language_code,
            backend_name="parakeet_tdt",
            model_id=os.getenv(
                "VOXALIGN_ASR_PARAKEET_TDT_MODEL_ID", _DEFAULT_PARAKEET_TDT_MODEL_ID
            ),
        )
    if selected == "crisper_whisper":
        return _transcribe_pipeline_asr(
            audio_path=audio_path,
            language_code=language_code,
            backend_name="crisper_whisper",
            model_id=os.getenv("VOXALIGN_ASR_CRISPER_MODEL_ID", _DEFAULT_CRISPER_MODEL_ID),
        )
    if selected == "whisper_large_v3":
        return _transcribe_pipeline_asr(
            audio_path=audio_path,
            language_code=language_code,
            backend_name="whisper_large_v3",
            model_id=os.getenv("VOXALIGN_ASR_WHISPER_MODEL_ID", _DEFAULT_WHISPER_MODEL_ID),
        )

    raise ValueError("ASR backend is disabled; transcript must be provided.")


def _resolve_backend_choice(
    *,
    requested_backend: AsrBackendName,
    language_code: str | None,
    verbatim: bool,
) -> AsrBackendName:
    if requested_backend != "auto":
        return requested_backend

    code = _normalize_language_code(language_code)
    if code == "en":
        return "crisper_whisper" if verbatim else "parakeet"
    if code in _PARAKEET_TDT_EU_CODES:
        return "parakeet_tdt"
    return "whisper_large_v3"


def _transcribe_parakeet(
    *,
    audio_path: str,
    language_code: str | None,
    sample_rate_hz: int | None,
) -> AsrResult:
    model_id = os.getenv("VOXALIGN_ASR_PARAKEET_MODEL_ID", _DEFAULT_PARAKEET_MODEL_ID)
    simulated = _simulated_asr_result(
        backend="parakeet",
        language_code=language_code,
    )
    if not _env_truthy("VOXALIGN_ASR_USE_HF", default=False):
        return simulated

    wav_payload = read_wav_audio(audio_path)
    if wav_payload is None:
        return simulated
    audio, detected_sample_rate_hz = wav_payload
    if sample_rate_hz is None:
        sample_rate_hz = detected_sample_rate_hz

    bundle = _load_ctc_bundle(model_id=model_id)
    if bundle is None:
        return simulated

    if sample_rate_hz != bundle.target_sample_rate_hz:
        audio = resample_linear(
            audio,
            src_hz=sample_rate_hz,
            dst_hz=bundle.target_sample_rate_hz,
        )
        sample_rate_hz = bundle.target_sample_rate_hz

    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return simulated

    try:
        inputs = bundle.processor(audio, sampling_rate=sample_rate_hz, return_tensors="pt")
        inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = bundle.model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        transcript_raw = bundle.processor.batch_decode(predicted_ids)[0]
    except Exception:
        return simulated

    transcript = _normalize_transcript(transcript_raw)
    if not transcript:
        return simulated

    return AsrResult(
        transcript=transcript,
        language_code=_normalize_language_code(language_code) or "en",
        backend="parakeet",
        model_id=model_id,
        source="real",
    )


def _load_ctc_bundle(model_id: str) -> _CtcBundle | None:
    device_pref = os.getenv("VOXALIGN_ASR_DEVICE", "auto")
    cache_key = f"{model_id}@{device_pref}"
    cached = _HF_CTC_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(model_id)
    except Exception:
        return None

    device = _resolve_torch_device(torch=torch, preference=device_pref)
    model = model.to(device)
    model.eval()

    feature_extractor = getattr(processor, "feature_extractor", None)
    target_hz = int(getattr(feature_extractor, "sampling_rate", 16000))
    bundle = _CtcBundle(
        processor=processor,
        model=model,
        target_sample_rate_hz=target_hz,
        device=device,
    )
    _HF_CTC_CACHE[cache_key] = bundle
    return bundle


def _transcribe_pipeline_asr(
    *,
    audio_path: str,
    language_code: str | None,
    backend_name: AsrBackendName,
    model_id: str,
) -> AsrResult:
    simulated = _simulated_asr_result(
        backend=backend_name,
        language_code=language_code,
    )
    if not _env_truthy("VOXALIGN_ASR_USE_HF", default=False):
        return simulated

    try:
        import torch
        from transformers import pipeline
    except ModuleNotFoundError:
        return simulated

    device = _resolve_pipeline_device(
        torch=torch,
        preference=os.getenv("VOXALIGN_ASR_DEVICE", "auto"),
    )
    cache_key = f"{backend_name}@{model_id}@{device}"
    pipe = _HF_PIPELINE_CACHE.get(cache_key)
    if pipe is None:
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=device,
            )
        except Exception:
            return simulated
        _HF_PIPELINE_CACHE[cache_key] = pipe

    try:
        result = pipe(audio_path)
    except Exception:
        return simulated

    transcript_raw = str(result.get("text", "")).strip() if isinstance(result, dict) else ""
    transcript = _normalize_transcript(transcript_raw)
    if not transcript:
        return simulated

    return AsrResult(
        transcript=transcript,
        language_code=_normalize_language_code(language_code) or "und",
        backend=backend_name,
        model_id=model_id,
        source="real",
    )


def _simulated_asr_result(
    *,
    backend: AsrBackendName,
    language_code: str | None,
) -> AsrResult:
    code = _normalize_language_code(language_code) or "und"
    if backend == "crisper_whisper":
        transcript = os.getenv("VOXALIGN_ASR_SIM_CRISPER", "uh hello uh world")
    elif backend == "parakeet_tdt":
        if code == "fr":
            transcript = os.getenv("VOXALIGN_ASR_SIM_PARAKEET_TDT_FR", "bonjour le monde")
        elif code == "de":
            transcript = os.getenv("VOXALIGN_ASR_SIM_PARAKEET_TDT_DE", "hallo welt")
        elif code == "es":
            transcript = os.getenv("VOXALIGN_ASR_SIM_PARAKEET_TDT_ES", "hola mundo")
        else:
            transcript = os.getenv("VOXALIGN_ASR_SIM_PARAKEET_TDT", "hello world")
    elif backend == "whisper_large_v3":
        if code == "ko":
            transcript = os.getenv("VOXALIGN_ASR_SIM_WHISPER_KO", "안녕하세요 반갑습니다")
        else:
            transcript = os.getenv("VOXALIGN_ASR_SIM_WHISPER", "hello world")
    elif backend == "parakeet":
        transcript = os.getenv("VOXALIGN_ASR_SIM_PARAKEET", "hello world")
    else:
        transcript = "hello world"
    return AsrResult(
        transcript=_normalize_transcript(transcript) or "hello world",
        language_code=code,
        backend=backend,
        model_id=_SIM_MODEL_ID,
        source="simulated",
    )


def _normalize_transcript(text: str) -> str:
    normalized = _SPACES_RE.sub(" ", text.strip())
    return normalized


def _resolve_torch_device(torch: Any, preference: str) -> str:
    pref = preference.casefold()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_pipeline_device(torch: Any, preference: str) -> int:
    pref = preference.casefold()
    if pref == "cpu":
        return -1
    if pref == "cuda":
        return 0 if torch.cuda.is_available() else -1
    if torch.cuda.is_available():
        return 0
    return -1


def _normalize_language_code(language_code: str | None) -> str | None:
    if language_code is None:
        return None
    cleaned = language_code.strip().casefold().replace("_", "-")
    if not cleaned:
        return None
    return cleaned.split("-")[0]


def _env_truthy(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.casefold() not in {"0", "false", "no", "off"}
