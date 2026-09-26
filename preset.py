"""Loads preset.toml and exposes its values, including the derived model_dir.

Usage: python preset.py <key>   -- prints one value, for shell scripts.
"""

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PRESET_PATH = REPO_ROOT / "preset.toml"


def load_preset() -> dict:
    with open(PRESET_PATH, "rb") as f:
        data = tomllib.load(f)

    model_id = data["model_id"]
    revision = data["revision"]
    model_name = model_id.split("/")[-1]
    short_revision = revision[:12]
    model_dir = REPO_ROOT / "models" / f"{model_name}@{short_revision}"

    return {
        "model_id": model_id,
        "revision": revision,
        "backend": data["backend"],
        "prompt": data["prompt"],
        "host": data["host"],
        "port": data["port"],
        "model_dir": str(model_dir),
    }


def main() -> None:
    key = sys.argv[1]
    preset = load_preset()
    if key not in preset:
        print(f"unknown preset key: {key}", file=sys.stderr)
        sys.exit(1)
    print(preset[key])


if __name__ == "__main__":
    main()
