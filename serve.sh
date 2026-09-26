#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

BACKEND="$(uv run python preset.py backend)"
PROMPT="$(uv run python preset.py prompt)"
HOST="$(uv run python preset.py host)"
PORT="$(uv run python preset.py port)"
MODEL_DIR="$(uv run python preset.py model_dir)"

if [[ ! -f "$MODEL_DIR/REVISION" ]]; then
    echo "serve.sh: $MODEL_DIR/REVISION missing, run fetch_model.sh first" >&2
    exit 1
fi

exec uv run llm2jev --model "$MODEL_DIR" --backend "$BACKEND" --prompt "$PROMPT" --host "$HOST" --port "$PORT"
