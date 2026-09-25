#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${1:-$(date +%Y-%m-%d)-minicpm5-2b-8bit}"
RUN_DIR="$REPO_ROOT/results/$RUN_NAME"

if [[ -e "$RUN_DIR" ]]; then
    echo "bench.sh: step 'run dir check' failed: $RUN_DIR already exists; delete it to rerun" >&2
    exit 1
fi

HOST="$(uv run python preset.py host)"
PORT="$(uv run python preset.py port)"
MODEL_ID="$(uv run python preset.py model_id)"

if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "bench.sh: step 'port check' failed: port $PORT is already listening" >&2
    exit 1
fi

SERVER_PID=""

kill_tree() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        kill_tree "$child"
    done
    kill "$pid" 2>/dev/null || true
}

cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill_tree "$SERVER_PID"
    fi
}
trap cleanup EXIT

mkdir -p "$RUN_DIR"

"$REPO_ROOT/fetch_model.sh"
uv run python check.py

"$REPO_ROOT/serve.sh" >"$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!

HEALTH_URL="http://$HOST:$PORT/health"
healthy=""
for _ in $(seq 1 120); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "bench.sh: step 'start server' failed: server process died; log: $RUN_DIR/server.log" >&2
        exit 1
    fi
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
        healthy=1
        break
    fi
    sleep 1
done
if [[ -z "$healthy" ]]; then
    echo "bench.sh: step 'start server' failed: server did not become healthy within 120s; log: $RUN_DIR/server.log" >&2
    exit 1
fi

uv run python check.py --health

run_jevbench() {
    local tasks="$1" out_dir="$2"
    mkdir -p "$out_dir"
    if ! uv run python -m jevbench.cli run \
        --tasks "$tasks" \
        --adapter typesafe \
        --endpoint "http://$HOST:$PORT" \
        --key-env '' \
        --model "$MODEL_ID" \
        --reserve-usd 0 \
        --cost-basis self_hosted_local \
        --results "$out_dir/results.jsonl" \
        --raw-dir "$out_dir/raw" \
        --ledger "$out_dir/ledger.jsonl"; then
        echo "bench.sh: step 'jevbench run ($out_dir)' failed: jevbench exited non-zero; log: $RUN_DIR/server.log" >&2
        exit 1
    fi
}

SMOKE_DIR="$RUN_DIR/smoke"
mkdir -p "$SMOKE_DIR"
uv run python smoke_tasks.py "$SMOKE_DIR/tasks.jsonl"
run_jevbench "$SMOKE_DIR/tasks.jsonl" "$SMOKE_DIR"

if ! smoke_failures="$(uv run python -c "
import json
for line in open('$SMOKE_DIR/results.jsonl'):
    r = json.loads(line)
    if not r['ok']:
        print(json.dumps(r))
")"; then
    echo "bench.sh: step 'smoke' failed: could not inspect $SMOKE_DIR/results.jsonl; log: $RUN_DIR/server.log" >&2
    exit 1
fi
if [[ -n "$smoke_failures" ]]; then
    echo "bench.sh: step 'smoke' failed: item(s) with ok=false; log: $RUN_DIR/server.log" >&2
    echo "$smoke_failures" >&2
    exit 1
fi

server_alive() {
    kill -0 "$SERVER_PID" 2>/dev/null && curl -sf --max-time 5 "$HEALTH_URL" >/dev/null 2>&1
}

for TIER in original easy hard; do
    run_jevbench "vendor/jevbench/datasets/public/$TIER.jsonl" "$RUN_DIR/$TIER"
    if ! server_alive; then
        echo "bench.sh: step 'tier $TIER' failed: server died; log: $RUN_DIR/server.log" >&2
        exit 1
    fi
done

uv run python env_info.py "$RUN_DIR/env.json"
uv run python make_results.py "$RUN_DIR"

echo "bench.sh: done -> $RUN_DIR"
