# Benchmarking

`eval/benchmark.py` runs timestamp-quality and runtime benchmarks and archives artifacts.

## Manifest format

Use JSONL with one utterance per line:

```json
{
  "id": "utt-001",
  "audio_path": "path/to/audio.wav",
  "transcript": "hello world",
  "language": "en",
  "reference_words": [
    {"word": "hello", "start_sec": 0.02, "end_sec": 0.41},
    {"word": "world", "start_sec": 0.43, "end_sec": 0.98}
  ]
}
```

## Run

```bash
uv run python eval/benchmark.py \
  --manifest eval/manifests/sample.jsonl \
  --backend ctc_trellis
```

Artifacts are written under `eval/runs/<timestamp>_<sha>/`:

- `metrics.json`
- `per_utterance.csv`
- `run.json`
