# PROJECT

## Name

voxalign

## Goal

Build a multilingual forced aligner that maps transcript words and phonemes to precise audio timestamps through both CLI and API workflows.

## Current State

- Phase 1 completed: repository scaffold, CLI/API skeleton, CI baseline.
- Phase 2 in progress: canonical schema and deterministic alignment pipeline implemented.
- Tooling standardized on `mise` + `uv` + `pre-commit`.

## Milestones

1. Documentation and workflow baseline
   - Add project/operator docs
   - Define contributor workflow and milestone push rule
2. Text normalization and language plugin contract
   - Introduce pluggable language normalization interface
   - Wire pipeline to normalized/tokenized transcript source
   - Status: completed (baseline English + generic fallback packs)
3. First language module
   - Implement and test initial language pack behavior
4. Audio-driven alignment primitives
   - Replace placeholder duration logic with audio-derived timing
   - Status: completed (WAV metadata timing + heuristic fallback)
5. CTC/trellis model integration
   - Introduce real alignment backend and quality benchmarks

## Primary Paths

- CLI: `voxalign align <audio> <transcript>`
- API: `POST /v1/align`

## Quality Gates

- `uv run pre-commit run --all-files`
- `uv run pytest -q`
- GitHub CI green on `main`
