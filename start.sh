#!/usr/bin/env bash
# star.sh — fusion-mlx lifecycle manager (start/stop/restart/status/log/tune/doctor)
# Keeps fusion-mlx at peak performance on Apple Silicon

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${PROJ_DIR}/.venv"
ACTIVATE="${VENV}/bin/activate"
LOG_DIR="${HOME}/.fusion-mlx/logs"
SETTINGS="${HOME}/.fusion-mlx/settings.json"
PORT=11434
HF_MIRROR="https://hf-mirror.com"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log_info()  { printf "${GREEN}[INFO]${NC}  %s\n" "$*"; }
log_warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
log_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
log_step()  { printf "${CYAN}[STEP]${NC}  %s\n" "$*"; }

# ── Activate venv ───────────────────────────────────────────────────
ensure_venv() {
    if [[ ! -f "${ACTIVATE}" ]]; then
        log_error "Virtualenv not found at ${VENV}"
        exit 1
    fi
    source "${ACTIVATE}"
}

# ── Check if server is running ──────────────────────────────────────
is_running() {
    fusion-mlx ps 2>/dev/null | /usr/bin/grep -q "${PORT}"
}

get_pid() {
    fusion-mlx ps 2>/dev/null | /usr/bin/grep "${PORT}" | awk '{print $1}' | head -1
}

# ── Wait for healthy ────────────────────────────────────────────────
wait_healthy() {
    local timeout="${1:-60}"
    local elapsed=0
    while (( elapsed < timeout )); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            log_info "Server is healthy (took ${elapsed}s)"
            return 0
        fi
        sleep 2
        (( elapsed += 2 ))
    done
    log_error "Server did not become healthy within ${timeout}s"
    return 1
}

# ── Preflight checks ────────────────────────────────────────────────
preflight() {
    log_step "Preflight checks"
    ensure_venv

    # Check port conflict
    if lsof -iTCP:"${PORT}" -sTCP:LISTEN -P -n 2>/dev/null | /usr/bin/grep -qv "fusion-mlx\|python"; then
        log_warn "Port ${PORT} occupied by another process:"
        lsof -iTCP:"${PORT}" -sTCP:LISTEN -P -n 2>/dev/null | head -3
        log_error "Free port ${PORT} first, or change PORT in this script"
        exit 1
    fi

    # Ensure log directory
    mkdir -p "${LOG_DIR}"

    # Set HF mirror for model downloads
    export HF_ENDPOINT="${HF_MIRROR}"
    export HUGGINGFACE_HUB_CACHE="${HOME}/.fusion-mlx/models"

    log_info "Preflight OK (port=${PORT}, HF mirror=${HF_MIRROR})"
}

# ── start ───────────────────────────────────────────────────────────
do_start() {
    preflight

    if is_running; then
        log_warn "Server already running on port ${PORT} (PID $(get_pid))"
        wait_healthy 10
        return 0
    fi

    log_step "Starting fusion-mlx on port ${PORT}"

    # Read model_dir from settings if available
    local model_dir="${HOME}/.fusion-mlx/models"
    if [[ -f "${SETTINGS}" ]]; then
        local md
        md=$(python3 -c "import json; d=json.load(open('${SETTINGS}')); print(d.get('model',{}).get('model_dir','${model_dir}'))" 2>/dev/null || echo "${model_dir}")
        model_dir="${md}"
    fi

    fusion-mlx serve \
        --model-dir "${model_dir}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --log-level INFO \
        --enable-prefix-cache \
        --continuous-batching \
        --chunked-prefill-tokens 4096 \
        &

    local serve_pid=$!
    log_info "Server PID: ${serve_pid}"

    if wait_healthy 120; then
        log_info "Fusion-MLX v$(fusion-mlx version 2>/dev/null | /usr/bin/grep -oP '[\d.]+' | head -1) started successfully"
        show_status
    else
        log_error "Start failed. Check logs: ${LOG_DIR}/server.log"
        tail -20 "${LOG_DIR}/server.log" 2>/dev/null || true
        exit 1
    fi
}

# ── stop ────────────────────────────────────────────────────────────
do_stop() {
    ensure_venv
    if ! is_running; then
        log_warn "Server not running on port ${PORT}"
        return 0
    fi

    local pid
    pid=$(get_pid)
    log_step "Stopping fusion-mlx (PID ${pid})"

    # Graceful: SIGTERM
    kill -TERM "${pid}" 2>/dev/null || true
    local waited=0
    while (( waited < 15 )); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            log_info "Server stopped gracefully"
            return 0
        fi
        sleep 1
        (( waited += 1 ))
    done

    # Force: SIGKILL
    log_warn "Graceful shutdown timed out, force killing..."
    kill -KILL "${pid}" 2>/dev/null || true
    sleep 1
    log_info "Server force-stopped"
}

# ── restart ─────────────────────────────────────────────────────────
do_restart() {
    log_step "Restarting fusion-mlx"
    do_stop
    sleep 2
    do_start
}

# ── status ──────────────────────────────────────────────────────────
show_status() {
    ensure_venv

    echo ""
    printf "${BLUE}━━━ Fusion-MLX Status ━━━${NC}\n"
    echo ""

    if is_running; then
        local pid
        pid=$(get_pid)
        printf "${GREEN}● Running${NC}  PID=%s  PORT=%s\n" "${pid}" "${PORT}"

        # Quick health check
        local health
        health=$(curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo "unreachable")
        printf "  Health: %s\n" "${health}"

        # Memory usage
        local rss
        rss=$(ps -o rss= -p "${pid}" 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024}')
        printf "  Memory: %s\n" "${rss:-unknown}"

        # Uptime
        local uptime
        uptime=$(ps -o etime= -p "${pid}" 2>/dev/null | xargs || echo "unknown")
        printf "  Uptime: %s\n" "${uptime}"

        # Models loaded
        local models
        models=$(curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'    - {m[\"id\"]}') for m in d.get('data',[])]" 2>/dev/null || echo "    (unable to list)")
        printf "  Models:\n%s\n" "${models}"
    else
        printf "${RED}● Stopped${NC}\n"
    fi

    # Disk usage
    local cache_size
    cache_size=$(du -sh "${HOME}/.fusion-mlx/models" 2>/dev/null | awk '{print $1}' || echo "N/A")
    local log_size
    log_size=$(du -sh "${LOG_DIR}" 2>/dev/null | awk '{print $1}' || echo "N/A")
    printf "\n  Cache: %s  Logs: %s\n" "${cache_size}" "${log_size}"
    echo ""
}

# ── log ─────────────────────────────────────────────────────────────
show_log() {
    local logfile="${LOG_DIR}/server.log"
    if [[ ! -f "${logfile}" ]]; then
        log_error "No log file at ${logfile}"
        return 1
    fi
    local lines="${1:-50}"
    if [[ "${lines}" == "-f" ]]; then
        tail -f "${logfile}"
    else
        tail -n "${lines}" "${logfile}"
    fi
}

# ── errors ──────────────────────────────────────────────────────────
show_errors() {
    local logfile="${LOG_DIR}/server.log"
    if [[ ! -f "${logfile}" ]]; then
        log_error "No log file at ${logfile}"
        return 1
    fi
    /usr/bin/grep -h "ERROR\|CRITICAL" "${logfile}" 2>/dev/null \
        | /usr/bin/grep -v "MagicMock\|simulated\|test_\|kaboom\|fatal test" \
        | tail -20
}

# ── tune — optimize settings for current hardware ───────────────────
do_tune() {
    log_step "Tuning for current hardware"

    local total_mem_gb
    total_mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    # Memory guard: 87.5% of total RAM (leave room for OS + other apps)
    local ceiling_gb=$(( total_mem_gb * 7 / 8 ))

    log_info "Total RAM: ${total_mem_gb} GB → memory ceiling: ${ceiling_gb} GB"

    if [[ ! -f "${SETTINGS}" ]]; then
        log_warn "No settings.json found, creating defaults"
        mkdir -p "$(dirname "${SETTINGS}")"
        python3 -c "
import json
json.dump({
    'version': '1.0',
    'server': {'port': ${PORT}, 'host': '127.0.0.1', 'log_level': 'INFO', 'auto_start_on_launch': True},
    'model': {'model_dir': '${HOME}/.fusion-mlx/models', 'model_dirs': ['${HOME}/.fusion-mlx/models']},
    'huggingface': {'endpoint': '${HF_MIRROR}'},
    'sampling': {'temperature': 0.25, 'repetition_penalty': 1.05, 'max_context_window': 131072, 'max_tokens': 8192},
    'cache': {'enabled': True, 'hot_cache_only': True, 'hot_cache_max_size': '20GB', 'initial_cache_blocks': 384},
    'idle_timeout': {'idle_timeout_seconds': 180},
    'scheduler': {'max_concurrent_requests': 4, 'chunked_prefill': True},
    'memory': {'memory_guard_tier': 'custom', 'memory_guard_custom_ceiling_gb': ${ceiling_gb}, 'prefill_memory_guard': True}
}, open('${SETTINGS}', 'w'), indent=4)
"
        log_info "Created ${SETTINGS} with tuned defaults"
    else
        # Patch existing settings
        python3 -c "
import json
with open('${SETTINGS}') as f:
    s = json.load(f)
# Tune memory
mem = s.setdefault('memory', {})
mem['memory_guard_tier'] = 'custom'
mem['memory_guard_custom_ceiling_gb'] = ${ceiling_gb}
mem['prefill_memory_guard'] = True
# Tune cache
cache = s.setdefault('cache', {})
cache.setdefault('enabled', True)
cache.setdefault('initial_cache_blocks', 384)
# Tune scheduler
sched = s.setdefault('scheduler', {})
sched.setdefault('max_concurrent_requests', 4)
sched.setdefault('chunked_prefill', True)
# HF mirror
hf = s.setdefault('huggingface', {})
hf['endpoint'] = '${HF_MIRROR}'
with open('${SETTINGS}', 'w') as f:
    json.dump(s, f, indent=4)
print(f'Tuned: memory ceiling=${ceiling_gb}GB, cache enabled, chunked prefill, HF mirror')
"
        log_info "Settings tuned in ${SETTINGS}"
    fi
}

# ── clean — rotate old logs, clear stale caches ─────────────────────
do_clean() {
    log_step "Cleaning up"

    # Rotate logs older than 7 days
    if [[ -d "${LOG_DIR}" ]]; then
        local count
        count=$(find "${LOG_DIR}" -name "*.log.*" -mtime +7 -delete -print 2>/dev/null | wc -l | tr -d ' ')
        log_info "Deleted ${count} old log files (7+ days)"
    fi

    # Clear __pycache__
    find "${PROJ_DIR}/fusion_mlx" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    log_info "Cleared __pycache__"

    # Trim launchd error log if > 5MB
    local err_log="${LOG_DIR}/launchd.err.log"
    if [[ -f "${err_log}" ]]; then
        local size
        size=$(stat -f%z "${err_log}" 2>/dev/null || echo 0)
        if (( size > 5242880 )); then
            tail -1000 "${err_log}" > "${err_log}.tmp" && mv "${err_log}.tmp" "${err_log}"
            log_info "Trimmed launchd.err.log (was $(( size / 1048576 )) MB)"
        fi
    fi

    log_info "Clean done"
}

# ── doctor ──────────────────────────────────────────────────────────
do_doctor() {
    ensure_venv
    fusion-mlx doctor 2>&1 || true
}

# ── Usage ───────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
star.sh — fusion-mlx lifecycle manager

Usage: star.sh <command> [args]

Commands:
  start       Start fusion-mlx with optimal settings
  stop        Graceful stop (SIGTERM → SIGKILL fallback)
  restart     Stop + start
  status      Show PID, port, memory, models, health
  log [N]     Tail server log (default 50 lines, -f to follow)
  errors      Show recent ERROR/CRITICAL from logs
  tune        Auto-tune settings.json for current hardware
  clean       Rotate old logs, clear __pycache__, trim error logs
  doctor      Run fusion-mlx doctor
  help        Show this help

Environment:
  PORT        Server port (default: 11434)
  HF_MIRROR   HuggingFace mirror (default: https://hf-mirror.com)
EOF
}

# ── Main ────────────────────────────────────────────────────────────
cmd="${1:-help}"
shift || true

case "${cmd}" in
    start)   do_start   ;;
    stop)    do_stop    ;;
    restart) do_restart ;;
    status)  show_status ;;
    log)     show_log "${1:-}" ;;
    errors)  show_errors ;;
    tune)    do_tune    ;;
    clean)   do_clean   ;;
    doctor)  do_doctor  ;;
    help|-h|--help) usage ;;
    *)
        log_error "Unknown command: ${cmd}"
        usage
        exit 1
        ;;
esac
