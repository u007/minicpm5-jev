# TODO

## Next: sub-project 2 — calibration + Hugging Face release
- Fit calibration (temperature / L0-style position debiasing) on data **disjoint from JevBench public items**; publish to `mercstudio/MiniCPM5-2B-8bit-jev` with a model card. Needs its own spec. Current ECE 0.266 shows it's needed.

## Deferred minors (from reviews)
- `make_results.py:95` hardcodes "(32 GB)"; derive from `env.json` `ram_bytes`.
- `preset.py:36` no-arg call raises IndexError traceback instead of a usage message.
- `fetch_model.sh:20` REVISION write is not atomic (self-heals on rerun).
- `fetch_model.sh:7-9` a failing `preset.py` gives a raw traceback with no named step.
- `check.py:91` tokenizer load runs outside `run()`, so a corrupt tokenizer gives a traceback, not a FAIL line.
- `bench.sh` creates `results/<run>/` before fetch/preflight; a failure leaves a partial dir that must be deleted by hand.
- `smoke_tasks.py` loose type hints (`list` vs `list[dict]`).
- `tests/test_make_results.py` duplicates `make_results._load_records`.
- Consider passing jevbench `--manifest` (dataset hash, timestamps) and recording the `mlx` core version in `env.json`.
