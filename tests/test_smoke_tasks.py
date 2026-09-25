import json
import subprocess
import sys
from pathlib import Path

import pytest
from jevbench.tasks import load_jsonl

from smoke_tasks import DATASET_DIR, main, pick_smoke

PUBLIC_PATHS = [str(DATASET_DIR / f"{name}.jsonl") for name in ("original", "easy", "hard")]


def test_pick_smoke_returns_three_records_with_distinct_types():
    records = pick_smoke(PUBLIC_PATHS)
    assert len(records) == 3
    assert {r["question"]["type"] for r in records} == {"noul", "choice", "score"}


def test_pick_smoke_raises_when_a_type_is_missing(tmp_path):
    fixture = tmp_path / "no_score.jsonl"
    fixture.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"question": {"type": "noul"}},
                {"question": {"type": "choice"}},
            )
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="score"):
        pick_smoke([str(fixture)])


def test_main_output_round_trips_through_load_jsonl(tmp_path):
    out_path = tmp_path / "smoke.jsonl"
    main(str(out_path))

    tasks = load_jsonl(str(out_path))
    assert len(tasks) == 3
    assert {t.question["type"] for t in tasks} == {"noul", "choice", "score"}


def test_pick_smoke_returns_first_match_in_file_order(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"question": {"type": "noul"}, "marker": "first"},
                {"question": {"type": "choice"}, "marker": "first"},
                {"question": {"type": "score"}, "marker": "first"},
            )
        )
        + "\n"
    )
    second.write_text(json.dumps({"question": {"type": "noul"}, "marker": "second"}) + "\n")

    records = pick_smoke([str(first), str(second)])

    noul = next(r for r in records if r["question"]["type"] == "noul")
    assert noul["marker"] == "first"


def test_pick_smoke_ignores_unrecognized_question_types(tmp_path):
    fixture = tmp_path / "with_unknown.jsonl"
    fixture.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"question": {"type": "unknown"}},
                {"question": {"type": "noul"}},
                {"question": {"type": "choice"}},
                {"question": {"type": "score"}},
            )
        )
        + "\n"
    )

    records = pick_smoke([str(fixture)])

    assert len(records) == 3
    assert {r["question"]["type"] for r in records} == {"noul", "choice", "score"}


def test_main_wrong_arg_count_exits_non_zero(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "smoke_tasks.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "usage" in result.stderr.lower()
