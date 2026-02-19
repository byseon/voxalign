# voxalign

Multilingual forced aligner for precise word- and phoneme-level timestamps.

## Overview

`voxalign` is a multilingual alignment toolkit designed to align transcripts to audio with high temporal precision across languages.

## Status

Phase 2 baseline in progress: uv tooling, alignment schema, and deterministic pipeline scaffold.

## Planned capabilities

- Multilingual alignment pipeline with language-specific normalization
- Word and phoneme timestamp output
- CLI for local/batch processing
- HTTP API for service integration
- Evaluation harness for alignment quality and runtime

## Documentation

- Implementation plan: `docs/implementation-plan.md`

## Quick start

```bash
uv sync --dev
uv run voxalign --help
```

## Local usage

CLI baseline alignment:

```bash
uv run voxalign align sample.wav "hello world" --language en
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

## Repository

- GitHub: [github.com/byseon/voxalign](https://github.com/byseon/voxalign)
