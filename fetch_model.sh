#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL_ID="$(uv run python preset.py model_id)"
REVISION="$(uv run python preset.py revision)"
MODEL_DIR="$(uv run python preset.py model_dir)"

if [[ -f "$MODEL_DIR/REVISION" && "$(cat "$MODEL_DIR/REVISION")" == "$REVISION" ]]; then
    echo "$MODEL_DIR"
    exit 0
fi

if ! uv run hf download "$MODEL_ID" --revision "$REVISION" --local-dir "$MODEL_DIR" >&2; then
    echo "fetch_model.sh: step 'hf download' failed" >&2
    exit 1
fi
echo "$REVISION" > "$MODEL_DIR/REVISION"
echo "$MODEL_DIR"
