# minicpm5-jev

Serves `mlx-community/MiniCPM5-2B-8bit` as a Jev-compatible `/v1/systemone`
decision service via [llm2jev](https://github.com/tic-top/anyjev), and
reproduces JevBench public-item numbers for that model using the official
[jevbench](https://github.com/fstandhartinger/jevbench) harness.

**Headline:** 0.580 accuracy (95% CI 0.516–0.642) on the 231 public JevBench
items, running on an Apple M1 Max (see [Results](#results)).

Credits: [llm2jev](https://github.com/tic-top/anyjev) (MIT), [jevbench](https://github.com/fstandhartinger/jevbench)
(MIT), [OpenBMB MiniCPM5](https://huggingface.co/openbmb/MiniCPM5-2B) (Apache-2.0),
and the [mlx-community](https://huggingface.co/mlx-community) 8-bit MLX build.
This repo is MIT licensed (see `LICENSE`).

## Requirements

- Apple Silicon Mac (MLX backend).
- [`uv`](https://docs.astral.sh/uv/).
- The `hf` CLI, installed by `uv sync` via `huggingface_hub` — nothing extra
  to install.
- About 2.5 GB of disk for the model weights, plus the `uv`-managed Python
  environment (~300 MB).

## Setup

```
git clone --recurse-submodules https://github.com/u007/minicpm5-jev
cd minicpm5-jev
uv sync
```

## Serving the model

```
bash fetch_model.sh   # downloads the pinned model revision (no-op if already local)
bash serve.sh         # serves it on http://127.0.0.1:8080
```

## Running the benchmark

```
bash bench.sh
```

This fetches the model if needed, starts its own server, smoke-tests it, runs
all 231 public JevBench items (original/easy/hard), writes
`results/<run>/`, and regenerates `RESULTS.md`. It refuses to start if port
8080 is already in use, so stop any `serve.sh` you started manually first.

## Example request

With `serve.sh` running:

```
curl -s -X POST http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "A customer writes: \"My package arrived broken and I want a replacement.\"",
    "model": "mlx-community/MiniCPM5-2B-8bit",
    "questions": {
      "q": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
          "track_order": "Wants to know where an order is",
          "report_damage": "Received an item that is broken or damaged",
          "cancel_order": "Wants to cancel an order"
        }
      }
    }
  }'
```

Response:

```json
{
    "id": "jev-4087b8d015be4c0d",
    "model": "mlx-community/MiniCPM5-2B-8bit",
    "answers": {
        "q": {
            "type": "choice",
            "choice": "report_damage",
            "probabilities": {
                "track_order": 0.00010876956782490854,
                "report_damage": 0.9987218473742457,
                "cancel_order": 0.0011693830579294182
            },
            "confidence": 0.9907475747251409
        }
    },
    "usage": {
        "input_tokens": 125,
        "output_tokens": 1
    }
}
```

## Results

JevBench public items, MiniCPM5-2B-8bit on an Apple M1 Max (32 GB), official
`jevbench` harness, 0 failed items:

| Tier | Items | Accuracy | 95% CI (Wilson) | ECE | Brier | p50 (s) | p95 (s) |
|---|---|---|---|---|---|---|---|
| **Public (all)** | 231 | **0.580** | 0.516–0.642 | 0.266 | 0.634 | 0.186 | 2.868 |
| original | 72 | 0.528 | 0.414–0.639 | 0.346 | 0.764 | 0.156 | 0.185 |
| easy | 48 | 0.938 | 0.832–0.979 | 0.025 | 0.085 | 0.156 | 0.174 |
| hard | 111 | 0.459 | 0.370–0.552 | 0.346 | 0.786 | 0.801 | 3.250 |

Chance level (mean of 1/number of options per item) is 0.311 for original,
0.284 for easy and 0.336 for hard. See [RESULTS.md](./RESULTS.md) for the
generated table with the Qwen3.5 reference rows.

The committed run (`results/2026-09-26-minicpm5-2b-8bit/`) took 3m16s wall
time (`time bash bench.sh` measured 3:15.69) on an Apple M1 Max (32 GB).

**This is not an official JevBench leaderboard score.** JevBench also has a
sealed item set, run only by the JevBench maintainers, which is not included
here — these numbers are the public-item subset only.

**The Qwen3.5 reference rows in `RESULTS.md` are not like-for-like.** They
were produced by `baby-jev`, a private harness, on A100 GPUs, per the
[llm2jev README](https://github.com/tic-top/anyjev) and its
[`scripts/bench.sh`](https://github.com/tic-top/anyjev/blob/main/scripts/bench.sh).
The MiniCPM5-2B-8bit row here comes from the official `jevbench` harness on
an Apple M1 Max (MLX). Treat the reference rows as context, not a comparison.

**In the original tier, the model answered "no" on all 24 yes/no (`noul`)
items**, scoring 12/24 — exactly chance, since the tier splits those items
12 yes / 12 no. The hard tier's `noul` items get a mix of "yes" and "no"
predictions, so this is model bias on the original tier, not a broken
label mapping.
