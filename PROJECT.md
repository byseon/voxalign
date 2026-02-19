# PROJECT

## Name

voxalign

## Goal

Build a multilingual forced aligner that maps transcript words and phonemes to precise audio timestamps through both CLI and API workflows.

## Current State

- Phase 1 completed: repository scaffold, CLI/API skeleton, CI baseline.
- Phase 2 completed: language routing, trellis decoder core, and optional HF emissions path implemented.
- Tooling standardized on `mise` + `uv` + `pre-commit`.
- Benchmark harness and release-gate plan added (`eval/`, `docs/benchmark-plan.md`).
- Benchmark gate datasets selected for initial run: Buckeye (EN) + Seoul Corpus (KO).
- Architecture synced to Notion spec: phoneme-first pipeline with IPA CTC alignment target model.

## Milestones

1. Documentation and workflow baseline
   - Add project/operator docs
   - Define contributor workflow and milestone push rule
   - Status: completed
2. Text normalization and language plugin contract
   - Introduce pluggable language normalization interface
   - Wire pipeline to normalized/tokenized transcript source
   - Status: completed (baseline English + generic fallback packs)
3. Audio-driven alignment primitives
   - Replace placeholder duration logic with audio-derived timing
   - Status: completed (WAV metadata timing + heuristic fallback)
4. CTC/trellis model integration
   - Introduce real alignment backend and quality benchmarks
   - Status: completed as interim backend (trellis/Viterbi + optional HF emissions + language model routing)
5. Phoneme-first pipeline foundation
   - Add VAD/ASR/G2P provider interfaces and IPA target sequence contracts
   - Status: in progress (phoneme-first backend routing added; provider interfaces pending)
6. IPA phoneme aligner integration
   - Integrate `facebook/wav2vec2-xlsr-53-espeak-cv-ft` with constrained CTC/Viterbi
   - Status: pending
7. English high-precision boundary path
   - Integrate Parakeet CTC 1.1B and evaluate Parakeet TDT backend
   - Status: pending

## Primary Paths

- CLI: `voxalign align <audio> <transcript>`
- API: `POST /v1/align`

## Quality Gates

- `uv run pre-commit run --all-files`
- `uv run pytest -q`
- GitHub CI green on `main`
