# minicpm5-jev — serve + JevBench evaluation (sub-project 1)

Date: 2026-09-25
Status: design approved in chat, awaiting written-spec review

## Goal

Anyone with an Apple Silicon Mac can clone `u007/minicpm5-jev` and, with one
command each:

1. Serve `mlx-community/MiniCPM5-2B-8bit` as a Jev-compatible
   `/v1/systemone` decision service.
2. Reproduce our JevBench public-item numbers for that model.

## Background

- **llm2jev** (`tic-top/anyjev`, PyPI `llm2jev`, MIT) already turns any chat
  model into a Jev-compatible probability service: one prefill, read the
  next-token logprobs over single-token option letters. Its MLX backend loads
  `mlx-community` quantized repos, and its chat prompt already passes
  `enable_thinking=False`. It serves `POST /v1/systemone` and `GET /health`.
- **MiniCPM5-2B** (OpenBMB, Apache-2.0): dense LlamaForCausalLM, 2.52B
  params, hybrid thinking chat template. The 8-bit MLX build is
  `mlx-community/MiniCPM5-2B-8bit` (group size 64).
- **JevBench** (`fstandhartinger/jevbench`, MIT): 231 public items (original
  72, easy 48, hard 111; by type: choice 139, noul 74, score 18; at most 6
  options per item). The sealed items are run only by the maintainers.
  Its `typesafe` adapter posts `{state, model, questions: {decision}}` to
  `{endpoint}/v1/systemone` and parses `answers.decision`, which matches the
  llm2jev response shape.
- Reference numbers (llm2jev README, A100, public 231): Qwen3.5-4B 0.740,
  Qwen3.5-9B 0.810, Qwen3.5-27B 0.879.

## Scope

In scope: a preset, a preflight check, a bench runner, committed results and
a results table.

Out of scope:
- A new server, or forking or patching llm2jev.
- Calibration and the Hugging Face release. That is sub-project 2, which gets
  its own spec after these results exist.
- CUDA, vLLM and SGLang paths.
- The sealed JevBench items, and any official leaderboard submission.

## Repo layout

| Path | Purpose |
|---|---|
| `pyproject.toml` | uv project. Pins `llm2jev[mlx]==0.6.1` exactly, and jevbench as a git dependency pinned to commit `1bcc55e`. |
| `preset.toml` | The single config: model id, backend `mlx`, prompt `chat`, host, port. |
| `serve.sh` | Runs the llm2jev CLI with values from the preset. No wrapper code. |
| `check.py` | Preflight checks (see below). Exits non-zero on any failure. |
| `bench.sh` | End-to-end bench run (see Eval flow). |
| `results/<YYYY-MM-DD>-minicpm5-2b-8bit/` | Raw responses, per-file `results.jsonl`, ledger, `summary.json`, `env.json`. Committed. |
| `make_results.py` | Builds `RESULTS.md` from the `summary.json` files. |
| `RESULTS.md` | Generated table: our row beside the llm2jev Qwen reference rows. |
| `README.md`, `LICENSE` (MIT), `tests/` | Standard. |

All model behaviour lives in llm2jev. This repo only adds configuration,
checks and evaluation.

## Preflight (`check.py`)

It loads the model's tokenizer and chat template through llm2jev's own prompt
path (the same render function the server uses). It fails if:

1. llm2jev's own `find_labels` raises when run on the MiniCPM5 tokenizer.
   It looks for 255 labels (A..Z, then AA..) that are each a single token
   right after the real prompt ending. The check calls that function
   rather than reimplementing it, and reports how many labels it found.
2. The rendered prompt contains `<think>`, or does not end at the assistant
   generation prompt.
3. When a server is running, `/health` reports a model id different from
   the preset.

## Eval flow (`bench.sh`)

The steps run in order. The first failure aborts with the failing step's name
and its log path. There are no retries.

1. Run `check.py`.
2. Start `serve.sh` in the background and poll `/health` with a 120 s
   timeout.
3. Smoke test: run the jevbench `typesafe` adapter on one item each of type
   `noul`, `choice` and `score`. Any `ok=false` aborts.
4. Run the full bench on `original`, `easy` and `hard` with: typesafe
   adapter, the local endpoint, empty key env, reserve 0 USD, cost basis
   `self_hosted_local`. Output goes to `results/<run>/`.
5. Run `jevbench.cli summarize` per file and write `summary.json`.
6. Write `env.json`: chip, RAM, macOS version, Python version, llm2jev and
   mlx-lm versions, the model's HF revision hash, the jevbench commit.
7. A `trap` stops the server on every exit path.

## Error handling

- Item-level failures (`ok=false`) count as incorrect answers, never as
  skipped items, and their count is reported in `RESULTS.md`.
- Process-level failures abort loudly (see Eval flow). Nothing is swallowed
  and nothing is retried.

## Results table (`RESULTS.md`)

Per model row:
- Public accuracy (231) with its 95% CI.
- Hard-111 accuracy.
- ECE and Brier score.
- p50 and p95 latency.
- Item failure count.

The llm2jev Qwen rows are copied with attribution and a note that their
latency came from A100s, while ours came from an M1 Max (32 GB).
`make_results.py` builds the table.

## Testing

`pytest`. The tests run offline once the tokenizer is cached.

- Preflight logic, tested against the real MiniCPM5 tokenizer: single-token
  letters, and no `<think>` in the rendered prompt.
- `make_results.py`, tested against a small fixture of
  `summary.json` data.
- The server and the full bench are not unit-tested. Step 3 (smoke test) of
  every bench run covers them.

## Success criteria

- `serve.sh` on an M1 Max returns valid `/v1/systemone` answers for all
  three question types.
- `bench.sh` completes all 231 public items and commits results plus
  `env.json`.
- `RESULTS.md` shows the MiniCPM5-2B-8bit row beside the reference rows.
- A fresh clone plus `uv sync` plus `bash bench.sh` reproduces the accuracy
  within the reported CI.

## Hosting

- GitHub: `u007/minicpm5-jev`, public, MIT. Created and pushed only after
  implementation, with the user's go-ahead.
- Hugging Face `mercstudio/MiniCPM5-2B-8bit-jev` belongs to sub-project 2.

## Known risks

- llm2jev's MLX backend has only been validated upstream on Qwen3-0.6B.
  Preflight and the smoke test are there to catch problems specific to
  MiniCPM5.
- The MLX backend has no prefix cache. That only affects latency, since
  JevBench sends one question per request.
- There is bf16/8-bit numerical noise (llm2jev reports 0.01–0.03 of
  probability drift across engines). That's acceptable at the resolution of
  the CI.
