#!/bin/bash
# hang-capture.sh (issue #1188): capture BEFORE-restart evidence for a wedged
# fusion-mlx process and classify it against the H1-H4 hypothesis table
# (architecture/fusion-mlx-deep-bug-1014.md). Zero deps beyond macOS/Linux
# built-ins; py-spy used when available.
#
# Usage: scripts/hang-capture.sh <PID> [PORT] [OUTDIR]
#   PID    the wedged fusion-mlx server process
#   PORT   metrics/admin port (default 11434; use 11433 for debug instances)
#   OUTDIR output dir (default /tmp/hang-capture-<pid>-<ts>)
#
# Exit codes: 0 captured, 1 usage error, 2 pid not found.
set -u

PID="${1:-}"
PORT="${2:-11434}"
if [ -z "$PID" ]; then
    echo "usage: $0 <PID> [PORT] [OUTDIR]" >&2
    exit 1
fi
if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: pid $PID not alive" >&2
    exit 2
fi
TS=$(date +%Y%m%d-%H%M%S)
OUT="${3:-/tmp/hang-capture-${PID}-${TS}}"
mkdir -p "$OUT"
echo "== hang capture for pid $PID -> $OUT =="
note() { echo "$1" | tee -a "$OUT/summary.txt"; }

# 1. Process state (H1 fingerprint: CPU 0% + Sleeping, memory static;
#    H3 fingerprint: single Python thread 100%).
ps -o pid,pcpu,pmem,stat,nlwp,etime,command -p "$PID" > "$OUT/ps.txt" 2>&1
note "process state: $(head -2 "$OUT/ps.txt" | tail -1)"

# Per-thread CPU (top 20) — a single hot thread suggests H3.
if top -l 1 -pid "$PID" -stats pid,command,cpu,th 2>/dev/null > "$OUT/top-macos.txt"; then
    :
elif top -b -H -n 20 -p "$PID" > "$OUT/top-linux.txt" 2>&1; then
    :
fi

# 2. Thread dump via faulthandler (H2 fingerprint: all workers lock-waiting).
#    fusion-mlx registers SIGUSR1 -> all-thread python stacks on stderr/log.
kill -USR1 "$PID" 2>/dev/null && note "SIGUSR1 faulthandler dump requested (check server log for all-thread stacks)" || true

# 3. Python stacks (H2/H3) — py-spy if installed.
if command -v py-spy >/dev/null 2>&1; then
    py-spy dump --pid "$PID" > "$OUT/pyspy-dump.txt" 2>&1 || note "py-spy dump failed (needs sudo?)"
else
    note "py-spy not installed — skipping python stack dump (pip install py-spy)"
fi

# 4. Native C++ stacks (H1: Metal queue wait shows here).
if command -v sample >/dev/null 2>&1; then
    sample "$PID" 5 -file "$OUT/sample.txt" >/dev/null 2>&1 && note "sample: 5s native profile captured"
elif command -v eu-stack >/dev/null 2>&1; then
    eu-stack "$PID" > "$OUT/eu-stack.txt" 2>&1 || true
fi

# 5. FD exhaustion check (H4 fingerprint: lsof count near ulimit -n).
FD_COUNT=$(lsof -p "$PID" 2>/dev/null | wc -l | tr -d ' ')
FD_LIMIT=$(ulimit -n)
FD_PCT=$(( FD_COUNT * 100 / (FD_LIMIT > 0 ? FD_LIMIT : 1) ))
echo "open_fds=$FD_COUNT limit=$FD_LIMIT pct=$FD_PCT" > "$OUT/fd.txt"
note "fd usage: $FD_COUNT/$FD_LIMIT (${FD_PCT}%) — H4 if >=90%"

# 6. Kernel Metal log (H1 fingerprint: GPU command-buffer stalls).
if command -v log >/dev/null 2>&1; then
    log show --predicate 'process == "kernel" AND message CONTAINS[c] "Metal"' --last 10m > "$OUT/kernel-metal.log" 2>&1 || true
fi

# 7. Live metrics: watchdog poison/heartbeat, fd gauges, admission, eviction.
#    Auth: settings.json auth.api_key if present (server may require it).
AUTH_HDR=()
SETTINGS="${FUSION_MLX_SETTINGS:-$HOME/.fusion-mlx/settings.json}"
if command -v python3 >/dev/null 2>&1 && [ -f "$SETTINGS" ]; then
    KEY=$(python3 -c "
import json,sys
try:
    print(json.load(open(sys.argv[1])).get('auth',{}).get('api_key','') or '')
except Exception:
    print('')" "$SETTINGS" 2>/dev/null)
    [ -n "$KEY" ] && AUTH_HDR=(-H "Authorization: Bearer $KEY")
fi
curl -s --max-time 5 "${AUTH_HDR[@]}" "http://127.0.0.1:${PORT}/metrics" > "$OUT/metrics.txt" 2>&1 || note "metrics fetch failed on :$PORT"
if [ -s "$OUT/metrics.txt" ] && grep -q "authentication_error" "$OUT/metrics.txt"; then
    note "metrics: auth required and no usable key in $SETTINGS — re-run with FUSION_MLX_SETTINGS pointing at the server settings"
fi
if [ -s "$OUT/metrics.txt" ]; then
    grep -E "watchdog|fd_used|fd_limit|fd_by_type|admission_|eviction|lock" "$OUT/metrics.txt" > "$OUT/metrics-relevant.txt" 2>/dev/null || true
fi

# 8. Workload state at hang time (#1184 overlap check: was training running?).
#    Endpoints may 404 on server versions predating #1187/#1186 — tolerated.
fetch_ep() {
    curl -s --max-time 5 "${AUTH_HDR[@]}" "http://127.0.0.1:${PORT}/$1" > "$OUT/$2" 2>&1 || true
    if grep -q "not_found_error" "$OUT/$2" 2>/dev/null; then
        note "$2: endpoint unavailable on this server version"
    fi
}
fetch_ep "v1/admission/status" "admission-status.json"
fetch_ep "admin/api/fine-tune/jobs" "fine-tune-jobs.json"

# 9. Rough classification (evidence hints, not a verdict).
note "--- classification hints ---"
if [ "$FD_PCT" -ge 90 ]; then
    note "H4 likely: fd usage ${FD_PCT}%"
fi
CPU=$(ps -o pcpu= -p "$PID" 2>/dev/null | tr -d ' ')
if [ -n "${CPU:-}" ]; then
    note "cpu_now=${CPU}% (H1: ~0% + Sleeping; H3: ~100% single thread — confirm in top/sample)"
fi
if [ -s "$OUT/metrics-relevant.txt" ]; then
    grep "watchdog_poison" "$OUT/metrics-relevant.txt" >> "$OUT/summary.txt" 2>/dev/null || true
fi
note "H1-H4 verdicts require human review of sample.txt / pyspy-dump.txt / kernel-metal.log against docs/hang-diagnosis.md"
note "capture complete: $OUT"
