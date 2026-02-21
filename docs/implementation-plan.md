# VoxAlign Implementation Plan

Date: 2026-02-19
Owner: byseon
Repo: `github.com/byseon/voxalign`

## Spec Sync Status

This plan is now synced to the detailed Notion architecture content provided in-chat on 2026-02-19.

## Locked Decisions

1. Pipeline ordering
   - Primary design: phoneme-first alignment.
   - ASR transcript -> G2P IPA sequence (+ word->phoneme map) -> CTC phoneme alignment -> word boundaries from grouped phonemes.
2. Segment granularity
   - VAD first, then split by ASR segments/sentence units for alignment stability.
3. Core acoustic aligner (MVP)
   - `facebook/wav2vec2-xlsr-53-espeak-cv-ft` for IPA phoneme CTC alignment.
4. ASR strategy
   - English default: CrisperWhisper (verbatim behavior).
   - English high-precision word-boundary path: Parakeet CTC (`nvidia/parakeet-ctc-1.1b`) with CTC logits.
   - Multilingual fallback: Whisper large-v3 (`faster-whisper` runtime target).
5. Locale policy
   - `--language` is required for best G2P and alignment quality.
   - `auto` remains fallback-only.
6. Korean and tier-2 languages
   - Korean uses `g2pk2` + mapping to espeak IPA inventory.
   - Chinese/Japanese use language-specific G2P mapping to espeak IPA inventory.

## Current Implementation vs Target

- Completed baseline:
  - CLI/API scaffold, schema, normalization contract, timing primitives.
  - Trellis/Viterbi CTC backend with optional HF emissions and device selection.
  - MMS adapter loading for language-routed CTC usage.
  - ASR auto routing wired (EN: Parakeet/optional Crisper, EU non-EN: Parakeet TDT, other: Whisper).
  - Benchmark harness + artifact archiving.
- Gap to target:
  - Current backend is token CTC alignment, not full phoneme-first IPA pipeline.
  - Dedicated VAD/ASR/G2P adapters are not wired yet.
  - `xlsr-53-espeak-cv-ft` path is partially wired (multilingual optional HF), but constrained phoneme alignment and robust G2P mapping are still pending.
  - Export formats beyond JSON (TextGrid/SRT/ASS/CTM) pending.

## Milestones (Updated)

1. Scaffold and tooling baseline
   - Status: completed
2. Schema, normalization, and audio timing
   - Status: completed
3. CTC trellis baseline + HF routing
   - Status: completed (interim backend)
4. Phoneme-first pipeline foundation
   - Add VAD, ASR, and G2P provider interfaces.
   - Add word->phoneme mapping contract to pipeline objects.
   - Status: in progress (backend routing + ASR provider routing implemented; VAD/G2P pending)
5. IPA aligner integration
   - Integrate `facebook/wav2vec2-xlsr-53-espeak-cv-ft` as phoneme CTC aligner.
   - Add constrained Viterbi decoding over IPA target sequence.
   - Status: in progress (optional multilingual HF-emissions path added)
6. English high-precision word-boundary path
   - Integrate Parakeet CTC 1.1B path for English word boundary constraints.
   - Evaluate Parakeet TDT path in dedicated backend.
   - Status: pending
7. Multilingual expansion
   - Add Korean (`g2pk2`) and mapping tables for `ko`, then `zh`, `ja`.
   - Harden EU language paths with espeak-ng fallback.
   - Status: pending
8. Benchmark and release gate
   - Run Buckeye (EN) + Seoul Corpus (KO) release-gate benchmarks.
   - Run TIMIT as calibration set for phone boundary quality.
   - Compare against MFA/WhisperX baselines where feasible.
   - Status: pending

## Benchmark Direction

- Release-gate first pass:
  - Buckeye subset for English
  - Seoul Corpus subset for Korean
- Calibration track:
  - TIMIT phone-boundary evaluation for English
- Artifacts:
  - `eval/runs/<timestamp>_<git_sha>/metrics.json`
  - `eval/runs/<timestamp>_<git_sha>/per_utterance.csv`
  - `eval/runs/<timestamp>_<git_sha>/run.json`

## Open Items

1. Default English ASR at launch: CrisperWhisper vs Parakeet CTC transcript path priority.
2. Parakeet TDT backend scope for v0.1.0 vs post-v0.1.0.
3. License posture for optional non-commercial ASR choices in default docs.
4. `--language auto` UX and warning behavior.
5. Locale variants (`en-US`, `pt-BR`) in G2P backend selection.
