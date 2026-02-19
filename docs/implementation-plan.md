# VoxAlign Implementation Plan

Date: 2026-02-19
Owner: byseon
Repo: `github.com/byseon/voxalign`

## Progress update (2026-02-19)

- Phase 1 scaffold started and committed:
  - Python package layout and build metadata (`pyproject.toml`)
  - Config profiles (`configs/dev.toml`, `configs/prod.toml`)
  - CLI entrypoint (`voxalign`)
  - FastAPI app with `/health` endpoint
  - Test suite skeleton and CI workflow
- Phase 2 baseline started:
  - Canonical alignment schema added (request/response, word, phoneme, metadata)
  - Deterministic baseline alignment pipeline implemented for contract testing
  - `POST /v1/align` endpoint and CLI JSON output wired to shared pipeline
  - Tooling switched to `uv` workflows and `pre-commit` hooks added
  - Text normalization + language pack contract added (English + generic fallback)
  - Audio-driven timing primitives added (WAV metadata + heuristic fallback)
  - CTC/trellis decoder core added (pluggable backends + selectable `ctc_trellis` + Viterbi pathing)
  - Optional HF CTC emissions path added (CPU/GPU auto-select with simulated fallback)
  - Language-routed CTC model selection added (English, European, Korean buckets)
  - Benchmark harness + release-gate benchmark plan added (`eval/benchmark.py`, `docs/benchmark-plan.md`)

## 1) Scope and assumptions

This plan targets a production-ready multilingual forced aligner with:

- Word- and phoneme-level timestamps
- Batch and API usage modes
- Extensible language packs
- Evaluation harness for alignment quality and runtime

Current limitation: the Notion page content could not be fetched from this environment yet, so this is a concrete baseline plan with explicit assumption tracking. Update this document after the spec content is available.

## 2) Target architecture

### 2.1 Core flow

1. Ingest audio + transcript (+ optional language hint)
2. Audio preprocessing (resample, VAD/chunking, normalization)
3. Language handling:
   - explicit language selection, or
   - LID fallback for mixed/unknown inputs
4. Text normalization and tokenization (language-specific rules)
5. Grapheme-to-phoneme conversion (or lexicon lookup)
6. Acoustic alignment:
   - per-language acoustic model adapter
   - CTC/trellis alignment stage
7. Post-processing:
   - merge/split segments
   - confidence scoring
   - boundary smoothing
8. Output serialization:
   - JSON (word/phone timings)
   - subtitles (SRT/VTT)
   - optional Praat TextGrid

### 2.2 Components

- `core/`
  - pipeline orchestration
  - shared data models
- `languages/`
  - language metadata
  - tokenization/G2P rules
  - lexicon packs
- `models/`
  - acoustic model registry and adapters
- `align/`
  - trellis/CTC aligner
  - boundary refinement
- `io/`
  - audio readers/writers
  - result exporters
- `api/`
  - HTTP service (async jobs + sync endpoint)
- `cli/`
  - local batch runner
- `eval/`
  - benchmark scripts and metrics

### 2.3 Non-functional targets (initial)

- P50 latency: <= 1.5x audio duration (single file, CPU baseline)
- Alignment error median: <= 40ms on evaluation set (word boundary)
- Deterministic outputs given fixed model + seed
- Structured logs + trace IDs for every job

## 3) Phased execution plan

## Phase 0: Requirement lock and technical baseline (2-3 days)

- Parse and reconcile the Notion architecture spec into:
  - final MVP feature set
  - explicit acceptance criteria
  - open decisions and owners
- Decide runtime stack:
  - Python version
  - inference backend (PyTorch/ONNX)
  - API framework (FastAPI suggested)
- Define supported languages for MVP (recommend 3-5)

Exit criteria:

- Approved requirement checklist
- MVP language list finalized
- Model selection documented

## Phase 1: Repo and platform scaffold (1-2 days)

- Initialize project structure and packaging
- Add lint/type/test baseline
- Add config system and environment profiles
- Add CI (lint + unit tests + smoke run)

Exit criteria:

- CI green on default branch
- `voxalign --help` and basic API health endpoint working

## Phase 2: Single-language end-to-end aligner (4-6 days)

- Implement preprocessing + normalization
- Integrate first language pack and G2P/lexicon path
- Implement base CTC alignment and JSON output
- Add golden tests from small labeled samples

Exit criteria:

- End-to-end alignment works for one language
- Accuracy and runtime baseline captured

## Phase 3: Multilingual expansion (5-8 days)

- Add language pack abstraction
- Add 2-4 additional language packs
- Implement LID fallback and mixed-language handling policy
- Add language-specific normalization edge cases

Exit criteria:

- Stable alignment across MVP language list
- Regression suite per language in CI

## Phase 4: API, batch jobs, and observability (3-5 days)

- Implement synchronous API endpoint
- Implement asynchronous job mode for long files
- Add metrics (latency, error counters, queue depth)
- Add structured job artifacts and failure diagnostics

Exit criteria:

- API and CLI produce consistent outputs
- Basic operational dashboard metrics available

## Phase 5: Quality hardening and release prep (3-4 days)

- Build evaluation harness and benchmark report
- Add load/perf tests
- Add docs and sample workflows
- Tag `v0.1.0` release candidate

Exit criteria:

- Release checklist complete
- Known-risk list accepted

## 4) Work breakdown (initial task list)

1. Define canonical alignment output schema (word/phoneme/confidence)
2. Implement text normalization interface + language plugin contract
3. Build first language module (lexicon + G2P adapter)
4. Implement core trellis alignment engine
5. Implement post-processor for boundary refinement
6. Build CLI command: `voxalign align <audio> <transcript>`
7. Build API endpoint: `POST /v1/align`
8. Add benchmark runner and result summary report
9. Add dataset loader and golden test fixtures
10. Add packaging and release workflow

## 5) Risks and mitigations

- Model drift across languages:
  - Mitigation: per-language validation suite and score thresholds
- Poor transcript quality:
  - Mitigation: normalization + mismatch detection + confidence flags
- Long-audio runtime:
  - Mitigation: chunking strategy + async job mode
- Language-specific tokenization errors:
  - Mitigation: pluggable tokenizer with targeted fixtures

## 6) Open decisions to confirm from Notion spec

1. Exact MVP language list
2. Required output formats beyond JSON/SRT/VTT/TextGrid
3. Hard latency/throughput SLOs
4. Deployment target (local tool only vs managed API service)
5. Licensing constraints for selected acoustic/G2P models

## 7) GitHub repo creation + bootstrap

When `gh` authentication is valid for `byseon`, run:

```bash
cd /Users/seongjinpark/dev/voxalign
gh auth login -h github.com
gh repo create byseon/voxalign --public --source=. --remote=origin --push
```

If creating manually in the UI first:

```bash
cd /Users/seongjinpark/dev/voxalign
git remote add origin git@github.com:byseon/voxalign.git
git add .
git commit -m "chore: initialize voxalign planning docs"
git push -u origin main
```
