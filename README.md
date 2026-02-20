# voxalign

Multilingual forced aligner for precise word- and phoneme-level timestamps.

## Overview

`voxalign` is a multilingual alignment toolkit designed to align transcripts to audio with high temporal precision across languages.

## Status

Phase 3 transition in progress: phoneme-first architecture integration on top of the existing
trellis baseline.

## Planned capabilities

- Multilingual alignment pipeline with language-specific normalization
- Word and phoneme timestamp output
- CLI for local/batch processing
- HTTP API for service integration
- Evaluation harness for alignment quality and runtime

## Target architecture (spec-synced)

- Pipeline ordering: phoneme-first (`ASR -> G2P -> IPA CTC forced alignment -> word grouping`)
- Primary phoneme aligner: `facebook/wav2vec2-xlsr-53-espeak-cv-ft`
- English high-precision word-boundary path: `nvidia/parakeet-ctc-1.1b`
- English verbatim ASR target: `nyrahealth/CrisperWhisper`
- Multilingual ASR target: Whisper large-v3 (`faster-whisper`)

## Documentation

- Operator guide: `AGENT.md`
- Project charter and milestone map: `PROJECT.md`
- Implementation plan: `docs/implementation-plan.md`
- Benchmark release-gate plan: `docs/benchmark-plan.md`

## Quick start

```bash
mise install
uv sync --dev
uv run voxalign --help
```

## Runtime management

Python runtime is pinned via `mise` in `.mise.toml` (`3.11.11`).

## Environment setup

### macOS / Linux

```bash
mise install
uv sync --dev --frozen
uv run voxalign --help
```

### VM (recommended: Ubuntu 22.04+)

Use the same Linux commands as above.

Minimum suggested VM size:

- 2 vCPU
- 4 GB RAM
- 10 GB free disk

### Cloud (Codespaces, EC2, GCP VM, Azure VM)

Use the same Linux commands as above.

For remote sessions, run the API server on `0.0.0.0` and control exposure with firewall/security-group rules:

```bash
uv run voxalign serve --host 0.0.0.0 --port 8000
```

### Windows

Recommended path: use Google Colab.

Example Colab cells:

```python
!git clone https://github.com/byseon/voxalign.git
%cd voxalign
!pip install -q uv
!uv sync --dev
!uv run voxalign --help
```

If you need local Windows execution, use WSL2 and follow the Linux setup.

## Local usage

CLI baseline alignment:

```bash
uv run voxalign align sample.wav "hello world" --language en
```

ASR-driven alignment (transcript omitted):

```bash
uv run voxalign align sample.wav --language en --asr auto --backend phoneme_first
```

English alias inputs are supported (`en-US`, `en-GB`, `en-CA`, `en-AU`).

Timing behavior:

- If `audio_path` points to a readable WAV file, timing uses real audio duration.
- Otherwise, timing falls back to a deterministic heuristic.
- Output metadata includes `timing_source` (`audio` or `heuristic`).

Backend behavior:

- `uniform` (default): even token distribution baseline
- `ctc_trellis`: trellis/Viterbi decoder (interim CTC backend)
  - uses Hugging Face CTC emissions when available
  - falls back to deterministic simulated emissions otherwise
- `phoneme_first`: spec-aligned routing backend
  - English: Parakeet-style CTC word boundaries -> constrained phoneme timing
  - Other languages: phoneme timing first -> word boundaries grouped from phonemes
  - Fallback: CTC word timing path when phoneme path cannot produce alignments

Use backend selection:

```bash
uv run voxalign align sample.wav "hello world" --backend ctc_trellis
uv run voxalign align sample.wav "hello world" --backend phoneme_first
```

ASR backend selection:

- `--asr disabled` (default; transcript required)
- `--asr auto` (English: `parakeet`, non-English: `whisper_large_v3`)
- `--asr parakeet`
- `--asr crisper_whisper`
- `--asr whisper_large_v3`

Use `--verbatim` to route English `--asr auto` to `crisper_whisper`.

License warning behavior for `crisper_whisper`:

- CLI prints a runtime warning to stderr.
- API response includes:
  - `metadata.license_warning`
  - `X-VoxAlign-License-Warning` response header

Enable Hugging Face emissions (optional):

```bash
uv sync --group asr
VOXALIGN_CTC_USE_HF=1 \
VOXALIGN_CTC_MODEL_ID=nvidia/parakeet-ctc-1.1b \
uv run voxalign align sample.wav "hello world" --backend ctc_trellis
```

Device selection (`cpu` / `cuda` / `mps` / `auto`):

```bash
VOXALIGN_CTC_DEVICE=auto uv run voxalign align sample.wav "hello world" --backend ctc_trellis
```

Language-model routing variables:

- `VOXALIGN_CTC_MODEL_EN` (default: `nvidia/parakeet-ctc-1.1b`)
- `VOXALIGN_CTC_MODEL_EU` (default: `facebook/mms-1b-all`)
- `VOXALIGN_CTC_MODEL_KO` (default: `facebook/mms-1b-all`)
- `VOXALIGN_CTC_MODEL_DEFAULT` (default: `facebook/mms-1b-all`)
- `VOXALIGN_CTC_MODEL_ID` (overrides all language routes)

Recommended baseline IDs:

- English: `nvidia/parakeet-ctc-1.1b`
- European languages: `facebook/mms-1b-all` (language adapter auto-selection by code)
- Korean: `facebook/mms-1b-all` (Korean adapter auto-selection)

Note: `nvidia/parakeet-tdt-0.6b-v3` is a Transducer/TDT model and needs a dedicated backend
(planned) for native timestamp extraction; current `ctc_trellis` is CTC-based.

Target phoneme-aligner ID for the phoneme-first path:

- `facebook/wav2vec2-xlsr-53-espeak-cv-ft`
- Override with `VOXALIGN_PHONEME_MODEL_ID=<hf_model_id>`

ASR model IDs (override via env):

- `VOXALIGN_ASR_PARAKEET_MODEL_ID` (default: `nvidia/parakeet-ctc-1.1b`)
- `VOXALIGN_ASR_CRISPER_MODEL_ID` (default: `nyrahealth/CrisperWhisper`)
- `VOXALIGN_ASR_WHISPER_MODEL_ID` (default: `openai/whisper-large-v3`)
- `VOXALIGN_ASR_USE_HF=1` to enable real HF model inference

Write result to file:

```bash
uv run voxalign align sample.wav "hello world" --language en -o outputs/alignment.json
```

Run API locally:

```bash
uv run uvicorn voxalign.api:app --reload
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/v1/align \
  -H "content-type: application/json" \
  -d '{"audio_path":"sample.wav","transcript":"hello world","language":"en"}'
```

## LLM integration

If you want to use `voxalign` from an LLM workflow, use one of these patterns.

### Pattern A: LLM calls API tool

1. Start the service:

```bash
uv run voxalign serve --host 127.0.0.1 --port 8000
```

2. LLM/tool call payload:

```json
{
  "audio_path": "sample.wav",
  "transcript": "hello world",
  "language": "en",
  "backend": "ctc_trellis",
  "include_phonemes": true
}
```

3. Endpoint:

- `POST /v1/align`

Use `metadata.token_count`, word timings, and phoneme timings in downstream reasoning.

### Pattern B: LLM executes CLI command

```bash
uv run voxalign align sample.wav "hello world" --language en -o outputs/alignment.json
```

Then have the LLM read and summarize `outputs/alignment.json`.

### Prompting tip for LLM tools

When asking an LLM agent to align speech, specify:

- audio path
- transcript text
- language code (`en`, `en-US`, `auto`, etc.)
- whether phoneme output is required

## Development checks

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest -q
```

## Benchmarking

Run benchmark harness:

```bash
uv run python eval/benchmark.py \
  --manifest eval/manifests/sample.jsonl \
  --backend ctc_trellis
```

Benchmark-first public datasets:

- English: TIMIT + Buckeye Corpus (word/phone timing references)
- Korean: Seoul Corpus (OpenSLR)

See benchmark details in:

- `eval/README.md`
- `docs/benchmark-plan.md`

ASR disfluency smoke check:

```bash
uv run python eval/asr_disfluency_smoke.py \
  --audio-path tests/fixtures/sample_en.wav \
  --language en
```

## Licensing

Project license:

- `voxalign` source code: MIT

Runtime dependencies in this repository:

- `fastapi`: MIT
- `uvicorn`: BSD-3-Clause
- `pydantic`: MIT
- `numpy`: BSD-3-Clause + additional component licenses (see NumPy distribution metadata)

Optional ASR/alignment runtime dependencies:

- `torch` (PyTorch): BSD-3-Clause
- `transformers`: Apache-2.0

Developer/tooling dependencies used in this repo:

- `uv`: Apache-2.0 OR MIT (dual license)
- `mise`: MIT
- `pre-commit`: MIT
- `pytest`: MIT
- `mypy`: MIT
- `ruff`: MIT

Model licenses (downloaded from Hugging Face at runtime when enabled):

- `nvidia/parakeet-ctc-1.1b`: CC-BY-4.0
- `nyrahealth/CrisperWhisper`: CC-BY-NC-4.0 (non-commercial)
- `openai/whisper-large-v3`: Apache-2.0
- `facebook/wav2vec2-xlsr-53-espeak-cv-ft`: Apache-2.0
- `facebook/mms-1b-all`: CC-BY-NC-4.0 (non-commercial)

Important:

- `CrisperWhisper` and `MMS` model licenses are non-commercial. Keep them opt-in when commercial use is possible.
- Always verify upstream license files/model cards at the commit/version you deploy.
- See `LICENSE` for repo license text and third-party notice summary.

## Pre-commit hooks

Install hooks:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

Run all hooks manually:

```bash
uv run pre-commit run --all-files
```

## Milestone policy

- Every completed milestone must be committed and pushed to remote.

## Repository

- GitHub: [github.com/byseon/voxalign](https://github.com/byseon/voxalign)
