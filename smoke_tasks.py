"""Picks one raw record of each question type (noul, choice, score) for smoke testing.

Usage: uv run python smoke_tasks.py <out_path>
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "vendor" / "jevbench" / "datasets" / "public"
QUESTION_TYPES = ("noul", "choice", "score")


def pick_smoke(paths: list) -> list:
    picked = []
    seen_types = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                qtype = record["question"]["type"]
                if qtype not in QUESTION_TYPES:
                    continue
                if qtype not in seen_types:
                    seen_types.add(qtype)
                    picked.append(record)
        if len(seen_types) == len(QUESTION_TYPES):
            break
    missing = [t for t in QUESTION_TYPES if t not in seen_types]
    if missing:
        raise ValueError(f"missing question type(s): {', '.join(missing)}")
    return picked


def main(out_path: str) -> None:
    paths = [str(DATASET_DIR / f"{name}.jsonl") for name in ("original", "easy", "hard")]
    records = pick_smoke(paths)
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: smoke_tasks.py <out_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
