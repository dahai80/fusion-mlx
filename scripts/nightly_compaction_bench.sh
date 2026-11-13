#!/usr/bin/env bash
# Nightly harness-compaction retention benchmark driver (#1559 / plan P2-4).
#
# Runs the MLX-path compaction benchmark on a schedule, never in PR CI.
# Each policy run starts from a fresh server so cache state cannot leak
# between policies.
#
# Install (launchd, daily 03:00):
#   cp scripts/com.fusionmlx.nightly-compaction.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/com.fusionmlx.nightly-compaction.plist
#
# Manual:  scripts/nightly_compaction_bench.sh [model_id]
set -euo pipefail

MODEL_ID="${1:-${FUSION_MLX_NIGHTLY_MODEL:-}}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/benchmarks/reports"
START_SH="$REPO_DIR/start.sh"
PORT="${FUSION_MLX_PORT:-11434}"
mkdir -p "$LOG_DIR"

if [[ -z "$MODEL_ID" ]]; then
    echo "usage: $0 <model_id>  (or set FUSION_MLX_NIGHTLY_MODEL)" >&2
    exit 2
fi

wait_healthy() {
    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "server failed to become healthy on :${PORT}" >&2
    return 1
}

run_one() {
    local policy_flag="$1" out="$2"
    echo "[nightly-compaction] restarting server for $out"
    "$START_SH" stop >/dev/null 2>&1 || true
    "$START_SH" start >/dev/null 2>&1 || true
    wait_healthy
    echo "[nightly-compaction] running benchmark -> $out"
    python "$REPO_DIR/benchmarks/compaction_retention_bench.py" \
        --model "$MODEL_ID" \
        --policy "$policy_flag" \
        --output "$out"
}

STAMP="$(date +%Y%m%d)"
run_one "fixed"   "$LOG_DIR/compaction_retention_fixed_$STAMP.json"
run_one "advisor" "$LOG_DIR/compaction_retention_advisor_$STAMP.json"

echo "[nightly-compaction] done; reports in $LOG_DIR"
