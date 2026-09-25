# minicpm5-jev serve + JevBench eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> Per the user's global rules, this plan is high-level: it names files,
> functions and behaviours, and contains no code. Implementers write the
> code themselves, test-first where a step says so.

**Goal:** Serve `mlx-community/MiniCPM5-2B-8bit` as a Jev-compatible
`/v1/systemone` service on Apple Silicon via the unmodified llm2jev, and
publish reproducible JevBench public-item results.

**Architecture:** This repo contains no server code. The llm2jev CLI serves
the model. Small Python scripts handle config (`preset.py`), preflight
(`check.py`), smoke-task selection (`smoke_tasks.py`), environment capture
(`env_info.py`) and results (`make_results.py`). Shell scripts orchestrate
fetch, serve and bench. The official jevbench harness is vendored as a
pinned submodule and runs through its `typesafe` adapter.

**Tech Stack:** Python 3.12, uv, pytest, llm2jev 0.6.1 (MLX backend),
mlx-lm 0.31.3, transformers 5.17.0, jevbench @ `1bcc55e`, bash, `hf` CLI.

**Spec:** `docs/specs/2026-09-25-minicpm5-jev-eval-design.md`

## Global Constraints

- Exact pins: `llm2jev[mlx]==0.6.1`, `mlx-lm==0.31.3`, `transformers==5.17.0`.
  The `hf` CLI comes from `huggingface_hub`, pinned to whatever version
  `uv lock` resolves, and is recorded in `env.json`.
- jevbench: git submodule `vendor/jevbench` at commit `1bcc55e`, installed as
  a uv path dependency. Datasets are read from
  `vendor/jevbench/datasets/public/{original,easy,hard}.jsonl`.
- Model: `mlx-community/MiniCPM5-2B-8bit`, revision
  `2d20e8e672ce892d50f7265bfd3fc9b59b718f2a`, always passed to llm2jev as a
  local path, never as a hub id.
- Backend `mlx`, prompt `chat`, host `127.0.0.1`, port `8080`. All of these
  live only in `preset.toml`.
- Bench adapter flags: `--adapter typesafe --key-env '' --reserve-usd 0
  --cost-basis self_hosted_local`.
- Item failures (`ok=false`) count as incorrect. They are never dropped,
  and jevbench already scores them that way.
- No retries, no silent fallbacks. Every script exits non-zero with the
  failing step's name and log path.
- MIT licence. Local commits only. Creating and pushing to GitHub
  `u007/minicpm5-jev` needs the user's explicit go-ahead.
- Out of scope: calibration, Hugging Face publishing, CUDA/vLLM/SGLang,
  sealed items.

## Review Focus

1. **Port 8080 already in use by another server.** `/health` answers, but
   from the wrong process. Expected: preflight check 3 fails because the
   model path doesn't match (tested in Task 3). `bench.sh` also refuses to
   start if the port is taken before it launches the server (Task 7).
2. **Server dies mid-run.** Every remaining item would silently become
   `ok=false`. Expected: `bench.sh` checks that the server process is alive
   after each tier and aborts with the server log path (Task 7).
3. **Rerun into an existing results directory.** Expected: `bench.sh`
   refuses to overwrite `results/<run>/` and names it (Task 7). Old
   numbers are never mixed with new ones.
4. **Model not fetched, or the HF download fails, or the revision is
   missing.** Expected: `fetch_model.sh` fails loudly, and `check.py` fails
   if the local model directory's recorded revision doesn't match the
   preset (Tasks 2 and 3).
5. **Empty or mismatched results passed to `make_results.py`** (a result
   file's task ids aren't in the tasks, or no item is scorable). Expected:
   it raises instead of writing a table with nonsense or divide-by-zero
   numbers (Task 6).

---

### Task 1: Scaffold, pins, vendored harness, preset loader

**Files:**
- Create: `pyproject.toml`, `uv.lock`, `.gitignore` (ignores `models/` and
  `.venv/`), `LICENSE` (MIT, 2026, u007), `.gitmodules` + `vendor/jevbench`
  (submodule at `1bcc55e`), `preset.toml`, `preset.py`
- Test: `tests/test_preset.py`

**Interfaces:**
- Produces: `preset.load_preset() -> dict` with keys `model_id`, `revision`,
  `backend`, `prompt`, `host`, `port`, `model_dir`. `model_dir` is derived
  as `models/MiniCPM5-2B-8bit@<first 12 chars of revision>` and made
  absolute from the repo root. Running `python preset.py <key>` prints one
  value, for use by shell scripts. An unknown key exits non-zero.

- [ ] Add the submodule and check out `1bcc55e`. Verify with
  `git -C vendor/jevbench rev-parse HEAD`.
- [ ] Write `pyproject.toml` with the exact pins, `pytest` as a dev
  dependency, `huggingface_hub` as a dependency, and jevbench as a path
  source. Run `uv sync`. Verify that `uv run python -c` can import
  `llm2jev`, `mlx_lm` and `jevbench.summarize`.
- [ ] Write the failing tests: `load_preset` returns all keys with the
  values from the Global Constraints; `model_dir` is absolute and contains
  the short revision; `python preset.py port` prints `8080`; an unknown key
  exits non-zero.
- [ ] Run the tests. Expect failures (no module).
- [ ] Implement `preset.toml` and `preset.py` (stdlib `tomllib`, no
  defaults: a missing key raises `KeyError` naming the key).
- [ ] Run the tests. Expect them to pass. Commit with the message
  "chore: scaffold, pins, vendored jevbench, preset".

### Task 2: Pinned model fetch

**Files:**
- Create: `fetch_model.sh`

**Interfaces:**
- Consumes: `preset.py model_id|revision|model_dir`.
- Produces: the model files in `model_dir`, plus a `REVISION` file in it
  containing the full revision sha. Prints `model_dir` on success.

- [ ] Implement: if `model_dir/REVISION` exists and matches the preset,
  print the path and exit 0. Otherwise run
  `hf download <model_id> --revision <revision> --local-dir <model_dir>`,
  then write `REVISION`. Any `hf` failure exits non-zero with its output.
  Use `set -euo pipefail`.
- [ ] Verify: run it once (it downloads about 2.7 GB), then again (it
  exits immediately). `ls model_dir` shows `chat_template.jinja`,
  `model.safetensors`, `tokenizer.json` and `REVISION`.
- [ ] Commit with the message "feat: pinned model fetch".

### Task 3: Preflight `check.py`

**Files:**
- Create: `check.py`
- Test: `tests/test_check.py` (needs the Task 2 model fetched; tests skip
  with an explicit reason if `model_dir` is absent, and are never silently
  green)

**Interfaces:**
- Consumes: `preset.load_preset()`; `llm2jev.prompt.render`, `find_labels`
  and `ANSWER`; `transformers.AutoTokenizer`.
- Produces: `check_revision(model_dir, revision)`,
  `check_labels(tokenizer, probe_prompt) -> int` (the number of labels),
  `check_prompt(prompt: str)` and `check_health(url, expected_model)`. Each
  raises `PreflightError` with a message on failure. `main()` runs the
  revision, labels and prompt checks, adds the health check when passed
  `--health`, prints one line per check and exits non-zero on the first
  failure.

- [ ] Write the failing tests:
  - With the real tokenizer, `check_labels` returns 255 on a probe built
    the way `LLM2Jev.__init__` builds it: `render` a one-question `noul`
    with labels `["A","B"]` and take that question's prompt.
  - `check_prompt` accepts the real rendered prompt, which ends with
    `</think>\n\n` followed by `Answer:`.
  - `check_prompt` rejects a synthetic prompt with an open `<think>`, and
    one that doesn't end with `ANSWER`.
  - `check_revision` rejects a directory whose `REVISION` differs.
  - `check_health` rejects a `/health` body whose `model` differs, using a
    stub HTTP server from stdlib `http.server` on an ephemeral port.
- [ ] Run the tests. Expect failures.
- [ ] Implement the four checks and `main()`. `check_prompt` applies
  the spec rule: from the last `<|im_start|>assistant` onward, every
  `<think>` must be closed, and the prompt must end with
  `</think>\n\n` + `ANSWER`.
- [ ] Run the tests. Expect them to pass. Run `uv run python check.py`
  and expect three passing lines, with labels=255. Commit with the message
  "feat: preflight checks".

### Task 4: `serve.sh`

**Files:**
- Create: `serve.sh`

**Interfaces:**
- Consumes: `preset.py backend|prompt|host|port|model_dir`.
- Produces: an llm2jev server in the foreground, run as
  `llm2jev --model <model_dir> --backend <backend> --prompt <prompt>
  --host <host> --port <port>`, with `exec` so signals reach it.

- [ ] Implement with `set -euo pipefail`. Fail if `model_dir/REVISION` is
  missing, telling the user to run `fetch_model.sh`.
- [ ] Verify manually: start it, `curl /health` shows `model` equal to
  `model_dir`, and three hand-written `POST /v1/systemone` requests (one
  each of noul, choice and score) return `answers` with probabilities that
  sum to about 1. Run `uv run python check.py --health` and expect four
  passing lines.
- [ ] Commit with the message "feat: serve script".

### Task 5: Smoke-task selection

**Files:**
- Create: `smoke_tasks.py`
- Test: `tests/test_smoke_tasks.py`

**Interfaces:**
- Consumes: `jevbench.tasks.load_jsonl`, the vendored public dataset
  paths.
- Produces: `pick_smoke(paths) -> list[dict]`, the first raw record of each
  type (`noul`, `choice`, `score`) in file order. It raises if any type is
  missing. `main(out_path)` writes those three records as JSONL.

- [ ] Write the failing tests: on the real public files it returns exactly
  3 records with distinct types; it raises on a fixture that has no
  `score` item; the output round-trips through `load_jsonl`.
- [ ] Run the tests (fail), implement, run the tests (pass), commit with
  the message "feat: smoke task picker".

### Task 6: `make_results.py`

**Files:**
- Create: `make_results.py`, `reference_rows.json` (the llm2jev Qwen
  rows: model, public acc, hard-111 acc, source URL, note "baby-jev
  harness, A100, not like-for-like")
- Test: `tests/test_make_results.py`, `tests/fixtures/` (tiny task and
  result JSONL for two tiers)

**Interfaces:**
- Consumes: `jevbench.tasks.load_jsonl`, `jevbench.summarize.metric`, the
  run layout `results/<run>/{original,easy,hard}/results.jsonl`, and
  `env.json`.
- Produces:
  - `wilson(k, n, z=1.96) -> (lo, hi)`, which raises when `n == 0`.
  - `summarize_run(run_dir, tier_paths) -> dict` with keys `tiers`
    (tier → metric dict) and `public` (metric dict plus `ci95`), where
    the combined result comes from `metric()` on the concatenated tasks
    and records.
  - `render_table(summaries, reference_rows) -> str`.
  - `main()`, which writes `results/<run>/summary.json` and regenerates
    `RESULTS.md` from every `results/*/summary.json` (sorted by run name)
    plus the reference block.

- [ ] Write the failing tests:
  - `wilson(7, 10)` matches the known value (0.397, 0.892) to 3 dp;
    `wilson(0, 0)` raises.
  - `summarize_run` on the fixtures: the per-tier dicts equal a direct
    `metric()` call, and the combined `n_scorable` equals the sum of the
    tiers'.
  - Records whose `task_id` isn't in the tasks make `summarize_run` raise
    with the offending ids.
  - A fixture with a failed item counts it in `n_scorable` and reduces
    accuracy.
  - The `render_table` output contains our row with accuracy and CI, hard
    accuracy, ECE, Brier, p50/p95 latency and failure count, plus a
    separate reference block with the "not like-for-like" note.
- [ ] Run the tests (fail), implement, run the tests (pass), commit with
  the message "feat: results summary and table".

### Task 7: `env_info.py` + `bench.sh`

**Files:**
- Create: `env_info.py`, `bench.sh`
- Test: `tests/test_env_info.py`

**Interfaces:**
- Consumes: everything above; `python -m jevbench.cli run`.
- Produces:
  - `env_info.collect() -> dict` with keys `chip`, `ram_bytes`, `macos`,
    `python`, `llm2jev`, `mlx_lm`, `transformers`, `huggingface_hub`,
    `model_id`, `model_revision`, `jevbench_commit`. It reads `sysctl`,
    `sw_vers`, `importlib.metadata` and `git -C vendor/jevbench rev-parse
    HEAD`, and raises if any of them is unavailable.
  - `bench.sh [run-name]`. The run name defaults to
    `<YYYY-MM-DD>-minicpm5-2b-8bit`.

- [ ] Write the failing test: `collect()` returns every key, each with a
  non-empty value, on this Mac. Run it (fail), implement, run it (pass).
- [ ] Implement `bench.sh` with `set -euo pipefail` and a `trap` that
  kills the server on EXIT. In order:
  1. Refuse to run if `results/<run>/` exists.
  2. Refuse to run if the port is already listening (`lsof -i`).
  3. Run `fetch_model.sh` and `check.py`.
  4. Start `serve.sh` in the background with its log at
     `results/<run>/server.log`. Poll `/health` for up to 120 s, and abort
     early if the process has died.
  5. Run `check.py --health`.
  6. Generate the smoke tasks, run jevbench on them into
     `results/<run>/smoke/`, and abort if any result has `ok=false`,
     printing its error.
  7. For each tier, run jevbench into `results/<run>/<tier>/` (results,
     raw, ledger), then check that the server is still alive and abort
     with the log path if not.
  8. Write `env.json` via `env_info.py`.
  9. Run `make_results.py`.
- [ ] Verify the failure paths by hand:
  - Rerunning with an existing run name aborts.
  - Occupying the port with `python -m http.server 8080` makes it abort
    before the server starts.
  - Killing the server during a tier makes it abort with the log path.

  Clean up the test run directories.
- [ ] Commit with the message "feat: end-to-end bench runner".

### Task 8: Real run, results, README

**Files:**
- Create: `results/<run>/…` (committed), `RESULTS.md`, `README.md`

- [ ] Run `bash bench.sh`. The expected outcome is all 231 items
  attempted, the smoke test passing and `RESULTS.md` regenerated. Record
  the wall time.
- [ ] Sanity-check the results:
  - `n_attempted` is 231, and the per-tier counts are 72, 48 and 111.
  - Failed items are listed if there are any.
  - Accuracy is above chance for each tier. If it isn't, stop and report
    rather than committing.
- [ ] Write `README.md`: what it is (credits llm2jev, jevbench, OpenBMB,
  mlx-community), requirements (Apple Silicon, uv, hf CLI, about 3 GB of
  disk), the four commands (clone with `--recurse-submodules`,
  `uv sync`, `bash serve.sh`, `bash bench.sh`), an example request, a link
  to `RESULTS.md`, the sealed-set caveat and the reference-rows caveat.
- [ ] Run the whole test suite: `uv run pytest -v`. Expect all tests to
  pass, with none skipped now that the model is present.
- [ ] Commit with the message "results: MiniCPM5-2B-8bit JevBench public
  run". Ask the user before creating or pushing the GitHub repo.
