# voxalign

Multilingual forced aligner for precise word- and phoneme-level timestamps.

## Overview

`voxalign` is a multilingual alignment toolkit designed to align transcripts to audio with high temporal precision across languages.

## Status

Phase 1 scaffold in progress: package, CLI, API health endpoint, and CI baseline.

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
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
voxalign --help
```

## Local usage

CLI placeholder:

```bash
voxalign align sample.wav "hello world" --language en
```

Run API locally:

```bash
uvicorn voxalign.api:app --reload
curl http://127.0.0.1:8000/health
```

## Development checks

```bash
ruff check src tests
mypy src
pytest
```

## Repository

- GitHub: [github.com/byseon/voxalign](https://github.com/byseon/voxalign)
