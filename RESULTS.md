# JevBench Results

## MiniCPM5-2B-8bit (this repo)

| Run | Chip | Public Acc (95% CI) | Hard-111 Acc | ECE | Brier | p50 (s) | p95 (s) | Failures |
|---|---|---|---|---|---|---|---|---|
| 2026-09-26-minicpm5-2b-8bit | Apple M1 Max | 0.580 (0.516, 0.642) | 0.459 | 0.266 | 0.634 | 0.186 | 2.868 | 0 |

## Reference (baby-jev harness, A100, not like-for-like)

Source: https://github.com/tic-top/anyjev

Note: baby-jev harness, A100, not like-for-like

| Model | Public Acc | Hard-111 Acc |
|---|---|---|
| Qwen3.5-4B | 0.740 | 0.595 |
| Qwen3.5-9B | 0.810 | 0.667 |
| Qwen3.5-27B | 0.879 | 0.775 |
