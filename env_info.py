"""Collects environment info for a bench run's env.json.

Usage: uv run python env_info.py <out_path>
"""

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from preset import load_preset

REPO_ROOT = Path(__file__).resolve().parent


def _run(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def collect() -> dict:
    preset = load_preset()
    return {
        "chip": _run("sysctl", "-n", "machdep.cpu.brand_string"),
        "ram_bytes": int(_run("sysctl", "-n", "hw.memsize")),
        "macos": _run("sw_vers", "-productVersion"),
        "python": sys.version.split()[0],
        "llm2jev": importlib.metadata.version("llm2jev"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "transformers": importlib.metadata.version("transformers"),
        "huggingface_hub": importlib.metadata.version("huggingface_hub"),
        "model_id": preset["model_id"],
        "model_revision": preset["revision"],
        "jevbench_commit": _run("git", "-C", str(REPO_ROOT / "vendor" / "jevbench"), "rev-parse", "HEAD"),
    }


def main(out_path: str) -> None:
    Path(out_path).write_text(json.dumps(collect(), indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: env_info.py <out_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
