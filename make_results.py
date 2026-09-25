"""Builds results/<run>/summary.json and regenerates RESULTS.md.

Usage: uv run python make_results.py <run_dir>
"""

import json
import math
import sys
from pathlib import Path

from jevbench.summarize import metric
from jevbench.tasks import load_jsonl

REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "vendor" / "jevbench" / "datasets" / "public"
RESULTS_DIR = REPO_ROOT / "results"
REFERENCE_ROWS_PATH = REPO_ROOT / "reference_rows.json"
RESULTS_MD_PATH = REPO_ROOT / "RESULTS.md"
TIERS = ("original", "easy", "hard")


def wilson(k, n, z=1.96):
    if n == 0:
        raise ValueError("wilson: n must be > 0")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = p + z2 / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return ((centre - adj) / denom, (centre + adj) / denom)


def _load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_run(run_dir, tier_paths):
    run_dir = Path(run_dir)
    tiers = {}
    failures = {}
    all_tasks = []
    all_records = []
    for tier, task_path in tier_paths.items():
        tasks = load_jsonl(str(task_path))
        records = _load_records(run_dir / tier / "results.jsonl")
        task_ids = {t.id for t in tasks}
        unknown = sorted({r["task_id"] for r in records if r["task_id"] not in task_ids})
        if unknown:
            raise ValueError(f"{tier}: results reference unknown task_id(s): {unknown}")
        tiers[tier] = metric(tasks, records)
        failures[tier] = sum(1 for r in records if not r["ok"])
        all_tasks.extend(tasks)
        all_records.extend(records)
    public = metric(all_tasks, all_records)
    public["ci95"] = wilson(public["n_correct"], public["n_scorable"])
    failures["public"] = sum(failures.values())
    return {"tiers": tiers, "public": public, "failures": failures}


def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_table(summaries, reference_rows):
    lines = [
        "# JevBench Results",
        "",
        "## MiniCPM5-2B-8bit (this repo)",
        "",
        "| Run | Chip | Public Acc (95% CI) | Hard-111 Acc | ECE | Brier | p50 (s) | p95 (s) | Failures |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        public = s["public"]
        hard = s["tiers"]["hard"]
        ci_lo, ci_hi = public["ci95"]
        ece = public["ece"]["ece"] if public["ece"] else None
        lines.append(
            f"| {s['run']} | {s['chip']} | {_fmt(public['accuracy'])} "
            f"({_fmt(ci_lo)}, {_fmt(ci_hi)}) | {_fmt(hard['accuracy'])} | "
            f"{_fmt(ece)} | {_fmt(public['brier_mean'])} | "
            f"{_fmt(public['latency']['p50_s'])} | {_fmt(public['latency']['p95_s'])} | "
            f"{s['failures']['public']} |"
        )
    lines += [
        "",
        "## Reference (baby-jev harness, A100, not like-for-like)",
        "",
        f"Source: {reference_rows['source']}",
        "",
        f"Note: {reference_rows['note']}",
        "",
        "| Model | Public Acc | Hard-111 Acc |",
        "|---|---|---|",
    ]
    for row in reference_rows["rows"]:
        lines.append(f"| {row['model']} | {_fmt(row['public_accuracy'])} | {_fmt(row['hard_accuracy'])} |")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) != 2:
        print("usage: make_results.py <run_dir>", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    tier_paths = {t: DATASET_DIR / f"{t}.jsonl" for t in TIERS}
    result = summarize_run(run_dir, tier_paths)
    env = json.loads((run_dir / "env.json").read_text())
    summary = {"run": run_dir.name, "chip": env["chip"], **result}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    reference_rows = json.loads(REFERENCE_ROWS_PATH.read_text())
    summaries = sorted(
        (json.loads(p.read_text()) for p in RESULTS_DIR.glob("*/summary.json")),
        key=lambda s: s["run"],
    )
    RESULTS_MD_PATH.write_text(render_table(summaries, reference_rows))


if __name__ == "__main__":
    main()
