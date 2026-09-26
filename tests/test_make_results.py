import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from jevbench.summarize import metric
from jevbench.tasks import load_jsonl

import make_results
from make_results import render_table, summarize_run, wilson

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

TIER_PATHS = {
    "easy": FIXTURES / "tasks" / "easy.jsonl",
    "hard": FIXTURES / "tasks" / "hard.jsonl",
}


def _load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_wilson_matches_known_value():
    lo, hi = wilson(7, 10)
    assert round(lo, 3) == 0.397
    assert round(hi, 3) == 0.892


def test_wilson_raises_on_zero_n():
    with pytest.raises(ValueError):
        wilson(0, 0)


def test_summarize_run_per_tier_matches_direct_metric_call():
    result = summarize_run(FIXTURES / "run", TIER_PATHS)

    for tier, task_path in TIER_PATHS.items():
        tasks = load_jsonl(str(task_path))
        records = _load_records(FIXTURES / "run" / tier / "results.jsonl")
        assert result["tiers"][tier] == metric(tasks, records)


def test_summarize_run_combined_n_scorable_is_sum_of_tiers():
    result = summarize_run(FIXTURES / "run", TIER_PATHS)

    assert result["public"]["n_scorable"] == (
        result["tiers"]["easy"]["n_scorable"] + result["tiers"]["hard"]["n_scorable"]
    )


def test_summarize_run_raises_on_unknown_task_id():
    with pytest.raises(ValueError, match="zzz"):
        summarize_run(
            FIXTURES / "run_bad_task",
            {"easy": TIER_PATHS["easy"]},
        )


def test_summarize_run_counts_failed_item_in_scorable_and_reduces_accuracy():
    result = summarize_run(FIXTURES / "run", TIER_PATHS)

    hard = result["tiers"]["hard"]
    # h1 correct, h2 ok=false -> counted incorrect, both scorable
    assert hard["n_scorable"] == 2
    assert hard["n_correct"] == 1
    assert hard["accuracy"] == 0.5
    assert result["failures"]["hard"] == 1
    assert result["failures"]["public"] == 1


def test_render_table_contains_our_row_and_reference_block():
    result = summarize_run(FIXTURES / "run", TIER_PATHS)
    summary = {"run": "2026-09-26-test-run", "chip": "Apple M1 Max", **result}
    reference_rows = json.loads((REPO_ROOT / "reference_rows.json").read_text())

    table = render_table([summary], reference_rows)

    public = result["public"]
    hard = result["tiers"]["hard"]
    lo, hi = public["ci95"]

    assert "2026-09-26-test-run" in table
    assert "Apple M1 Max" in table
    assert f"{public['accuracy']:.3f}" in table
    assert f"{lo:.3f}" in table
    assert f"{hi:.3f}" in table
    assert f"{hard['accuracy']:.3f}" in table
    assert "ECE" in table
    assert f"{public['ece']['ece']:.3f}" in table
    assert "Brier" in table
    assert f"{public['brier_mean']:.3f}" in table
    assert f"{public['latency']['p50_s']:.3f}" in table
    assert f"{public['latency']['p95_s']:.3f}" in table
    assert "| 1 |" in table  # failure count
    assert "baby-jev harness, A100, not like-for-like" in table
    assert "Qwen3.5-4B" in table
    assert "0.740" in table
    assert "0.595" in table


def test_render_table_raises_on_none_metric_value():
    result = summarize_run(FIXTURES / "run", TIER_PATHS)
    summary = {"run": "2026-09-26-test-run", "chip": "Apple M1 Max", **result}
    summary["public"]["brier_mean"] = None
    reference_rows = json.loads((REPO_ROOT / "reference_rows.json").read_text())

    with pytest.raises(ValueError):
        render_table([summary], reference_rows)


def test_main_writes_summary_and_rebuilds_results_md_from_summaries_alone(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    results_md = tmp_path / "RESULTS.md"
    monkeypatch.setattr(make_results, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(make_results, "RESULTS_MD_PATH", results_md)
    monkeypatch.setattr(make_results, "DATASET_DIR", FIXTURES / "tasks")
    monkeypatch.setattr(make_results, "TIERS", ("easy", "hard"))

    run_dir = results_dir / "2026-09-26-fixture"
    shutil.copytree(FIXTURES / "run", run_dir)
    (run_dir / "env.json").write_text(json.dumps({"chip": "Apple M1 Max"}))

    monkeypatch.setattr(sys, "argv", ["make_results.py", str(run_dir)])
    make_results.main()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["run"] == "2026-09-26-fixture"
    assert summary["chip"] == "Apple M1 Max"

    reference_rows = json.loads(make_results.REFERENCE_ROWS_PATH.read_text())
    expected = render_table([summary], reference_rows)
    assert results_md.read_text() == expected


def test_main_regenerates_results_md_sorted_by_run_name(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    results_md = tmp_path / "RESULTS.md"
    monkeypatch.setattr(make_results, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(make_results, "RESULTS_MD_PATH", results_md)
    monkeypatch.setattr(make_results, "DATASET_DIR", FIXTURES / "tasks")
    monkeypatch.setattr(make_results, "TIERS", ("easy", "hard"))

    for run_name in ("2026-09-26-later-run", "2026-01-01-earlier-run"):
        run_dir = results_dir / run_name
        shutil.copytree(FIXTURES / "run", run_dir)
        (run_dir / "env.json").write_text(json.dumps({"chip": "Apple M1 Max"}))
        monkeypatch.setattr(sys, "argv", ["make_results.py", str(run_dir)])
        make_results.main()

    text = results_md.read_text()
    assert text.index("2026-01-01-earlier-run") < text.index("2026-09-26-later-run")


def test_main_wrong_arg_count_exits_nonzero_with_usage():
    result = subprocess.run(
        [sys.executable, "make_results.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "usage" in result.stderr.lower()
