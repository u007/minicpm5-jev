"""Preflight checks: model revision, tokenizer labels, prompt shape, and (optionally) server health.

Usage: uv run python check.py [--health]
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from llm2jev.prompt import ANSWER, find_labels, render
from transformers import AutoTokenizer

from preset import load_preset

ASSISTANT_TURN = "<|im_start|>assistant"
THINK_TAG_RE = re.compile(r"<think>|</think>")


class PreflightError(Exception):
    pass


def check_revision(model_dir, revision) -> None:
    path = Path(model_dir) / "REVISION"
    if not path.exists():
        raise PreflightError(f"{path} does not exist")
    actual = path.read_text().strip()
    if actual != revision:
        raise PreflightError(f"{path} holds revision {actual!r}, expected {revision!r}")


def check_labels(tokenizer, probe_prompt) -> int:
    try:
        labels, _ = find_labels(tokenizer, probe_prompt)
    except ValueError as e:
        raise PreflightError(str(e)) from e
    return len(labels)


def check_prompt(prompt: str) -> None:
    idx = prompt.rfind(ASSISTANT_TURN)
    if idx == -1:
        raise PreflightError(f"prompt has no {ASSISTANT_TURN!r} turn")
    tail = prompt[idx:]
    depth = 0
    for m in THINK_TAG_RE.finditer(tail):
        depth += 1 if m.group() == "<think>" else -1
        if depth < 0:
            raise PreflightError("prompt has a '</think>' with no matching '<think>' in the assistant turn")
    if depth != 0:
        raise PreflightError("prompt has an unclosed '<think>' block in the assistant turn")
    ending = "</think>\n\n" + ANSWER
    if not prompt.endswith(ending):
        raise PreflightError(f"prompt does not end with {ending!r}")


def check_health(url, expected_model) -> None:
    try:
        with urllib.request.urlopen(url) as resp:
            body = json.load(resp)
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        raise PreflightError(f"GET {url} failed: {e}") from e
    if body.get("model") != expected_model:
        raise PreflightError(f"health reports model {body.get('model')!r}, expected {expected_model!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--health", action="store_true", help="also check the running server's /health")
    args = parser.parse_args()

    preset = load_preset()
    model_dir = preset["model_dir"]

    def run(name, fn):
        try:
            result = fn()
        except PreflightError as e:
            print(f"{name}: FAIL {e}")
            sys.exit(1)
        suffix = f" (labels={result})" if name == "labels" else ""
        print(f"{name}: ok{suffix}")
        return result

    run("revision", lambda: check_revision(model_dir, preset["revision"]))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _, probe, _ = render(tokenizer, "x", {"q": {"type": "noul"}}, ["A", "B"], preset["prompt"])
    probe_prompt = probe["q"][0]

    run("labels", lambda: check_labels(tokenizer, probe_prompt))
    run("prompt", lambda: check_prompt(probe_prompt))
    if args.health:
        url = f"http://{preset['host']}:{preset['port']}/health"
        run("health", lambda: check_health(url, model_dir))


if __name__ == "__main__":
    main()
