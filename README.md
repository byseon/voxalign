# voxalign

Multilingual forced aligner for precise word- and phoneme-level timestamps.

## Overview

`voxalign` is a multilingual alignment toolkit designed to align transcripts to audio with high temporal precision across languages.

## Status

Phase 2 in progress: uv tooling, alignment schema, language-pack normalization, and deterministic pipeline scaffold.

## Planned capabilities

- Multilingual alignment pipeline with language-specific normalization
- Word and phoneme timestamp output
- CLI for local/batch processing
- HTTP API for service integration
- Evaluation harness for alignment quality and runtime

## Documentation

- Operator guide: `AGENT.md`
- Project charter and milestone map: `PROJECT.md`
- Implementation plan: `docs/implementation-plan.md`

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

English alias inputs are supported (`en-US`, `en-GB`, `en-CA`, `en-AU`).

Timing behavior:

- If `audio_path` points to a readable WAV file, timing uses real audio duration.
- Otherwise, timing falls back to a deterministic heuristic.
- Output metadata includes `timing_source` (`audio` or `heuristic`).

Backend behavior:

- `uniform` (default): even token distribution baseline
- `ctc_trellis`: trellis/Viterbi decoder running over deterministic simulated emissions

Use backend selection:

```bash
uv run voxalign align sample.wav "hello world" --backend ctc_trellis
```

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
