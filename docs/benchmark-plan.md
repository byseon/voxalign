# Benchmark Plan (Release Gate)

## Goal

Define reproducible timestamp-quality and runtime benchmarks required before `v0.1.0`.

## Required datasets

1. Public known dataset split (license-compatible for benchmarking):
   - English: Buckeye Corpus
   - Korean: Seoul Corpus (OpenSLR)
2. Internal in-domain holdout split

Both datasets should provide word-level timestamp references.

## Phase-1 benchmark gate datasets

- Buckeye subset for English (curated manifest under `eval/manifests/buckeye.sample.jsonl`)
- Seoul Corpus subset for Korean (curated manifest under `eval/manifests/seoul_corpus.sample.jsonl`)

## Required metrics

- `word_boundary_mae_ms`
- `word_boundary_median_ms`
- `word_boundary_p90_ms`
- `word_boundary_p95_ms`
- tolerance rates:
  - `tolerance_le_20ms`
  - `tolerance_le_50ms`
  - `tolerance_le_100ms`
- runtime:
  - `rtf` (real-time factor)
  - `throughput_x`
- coverage:
  - `matched_word_coverage`

## Release-gate threshold template

Set project thresholds per language bucket:

- English
- European languages
- Korean

Thresholds must be locked before release candidate tagging.

## Archiving policy

Every benchmark run must archive artifacts under:

`eval/runs/<timestamp>_<git_sha>/`

Minimum artifacts:

- `metrics.json`
- `per_utterance.csv`
- `run.json` (command context)

## Publishing policy

- Publish aggregate metrics, methods, and scripts.
- Publish raw per-utterance artifacts only when dataset licensing allows.
