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
HOST="${FUSION_HOST:-127.0.0.1}"
HF_MIRROR_DEFAULT="https://hf-mirror.com"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log_info()  { printf "${GREEN}[INFO]${NC}  %s\n" "$*"; }
log_warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
log_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
log_step()  { printf "${CYAN}[STEP]${NC}  %s\n" "$*"; }

# ── Read mirror config from settings.json ────────────────────────────
# Priority: HF_MIRROR env var > settings.json huggingface.endpoint > default
resolve_hf_mirror() {
    if [[ -n "${HF_MIRROR:-}" ]]; then
        log_info "HF mirror from env: ${HF_MIRROR}"
        return 0
    fi
    if [[ -f "${SETTINGS}" ]]; then
        local endpoint
        endpoint=$(python3 -c "import json; d=json.load(open('${SETTINGS}')); print(d.get('huggingface',{}).get('endpoint',''))" 2>/dev/null || echo "")
        if [[ -n "${endpoint}" ]]; then
            HF_MIRROR="${endpoint}"
            log_info "HF mirror from config: ${HF_MIRROR}"
            return 0
        fi
    fi
    HF_MIRROR="${HF_MIRROR_DEFAULT}"
    log_info "HF mirror from default: ${HF_MIRROR}"
}

# ── Read API key from settings.json ─────────────────────────────────
resolve_api_key() {
    if [[ -n "${FUSION_MLX_API_KEY:-}" ]]; then
        API_KEY="${FUSION_MLX_API_KEY}"
        return 0
    fi
    if [[ -f "${SETTINGS}" ]]; then
        local key
        key=$(python3 -c "import json; d=json.load(open('${SETTINGS}')); print(d.get('auth',{}).get('api_key',''))" 2>/dev/null || echo "")
        if [[ -n "${key}" ]]; then
            API_KEY="${key}"
            return 0
        fi
    fi
    API_KEY=""
}

# ── curl with optional auth ─────────────────────────────────────────
auth_curl() {
    if is_uds; then
        local sock
        sock="$(uds_socket)"
        if [[ -n "${API_KEY:-}" ]]; then
            curl -sf --unix-socket "${sock}" -H "Authorization: Bearer ${API_KEY}" "$@"
        else
            curl -sf --unix-socket "${sock}" "$@"
        fi
    else
        if [[ -n "${API_KEY:-}" ]]; then
            curl -sf -H "Authorization: Bearer ${API_KEY}" "$@"
        else
            curl -sf "$@"
        fi
    fi
}

# ── UDS listen mode (#351) ───────────────────────────────────────────
# FUSION_HOST=unix:/path/to.sock -> listen on a Unix Domain Socket so
# only a process with filesystem access to the socket can reach MLX
# (physical isolation on top of the #349/#350 auth chain). Default TCP
# loopback is unchanged.
is_uds() { [[ "${HOST}" == unix:* ]]; }
uds_socket() { echo "${HOST#unix:}"; }

base_url() {
    if is_uds; then
        echo "http://localhost"
    else
        echo "http://${HOST}:${PORT}"
    fi
}

health_curl() {
    if is_uds; then
        curl -sf --unix-socket "$(uds_socket)" http://localhost/health
    else
        curl -sf "http://${HOST}:${PORT}/health"
    fi
}

host_port_args() {
    if is_uds; then
        echo "--host ${HOST}"
    else
        echo "--host ${HOST} --port ${PORT}"
    fi
}

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
    if is_uds; then
        fusion-mlx ps 2>/dev/null | /usr/bin/grep -Fq "$(uds_socket)"
    else
        fusion-mlx ps 2>/dev/null | /usr/bin/grep -q "${PORT}"
    fi
}

get_pid() {
    if is_uds; then
        fusion-mlx ps 2>/dev/null | /usr/bin/grep -F "$(uds_socket)" | awk '{print $1}' | head -1
    else
        fusion-mlx ps 2>/dev/null | /usr/bin/grep "${PORT}" | awk '{print $1}' | head -1
    fi
}

# ── Wait for healthy ────────────────────────────────────────────────
wait_healthy() {
    local timeout="${1:-60}"
    local elapsed=0
    while (( elapsed < timeout )); do
        if health_curl >/dev/null 2>&1; then
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

    # Resolve HF mirror from config, then set env
    resolve_hf_mirror
    resolve_api_key
    export HF_ENDPOINT="${HF_MIRROR}"
    export HUGGINGFACE_HUB_CACHE="${HOME}/.fusion-mlx/models"

    log_info "Preflight OK (port=${PORT}, HF mirror=${HF_MIRROR}, api_key=$([ -n "${API_KEY}" ] && echo "set" || echo "none"))"
}

# ── Parse start args ─────────────────────────────────────────────────
_parse_start_args() {
    local watchdog=""
    local preload=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --watchdog)
                watchdog="--watchdog"
                shift
                ;;
            --preload)
                if [[ -z "${2:-}" || "${2}" == --* ]]; then
                    log_error "--preload requires a comma-separated model list"
                    exit 1
                fi
                preload="$2"
                shift 2
                ;;
            *)
                log_error "Unknown start option: $1"
                exit 1
                ;;
        esac
    done
    START_WATCHDOG="${watchdog}"
    START_PRELOAD="${preload}"
}

# ── start ───────────────────────────────────────────────────────────
do_start() {
    _parse_start_args "$@"
    preflight

    if is_running; then
        log_warn "Server already running on ${HOST} (PID $(get_pid))"
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

    # Resolve preload models: CLI --preload > settings.json models.preload
    local preload_models="${START_PRELOAD}"
    if [[ -z "${preload_models}" && -f "${SETTINGS}" ]]; then
        local sp
        sp=$(python3 -c "import json; d=json.load(open('${SETTINGS}')); p=d.get('model',{}).get('preload',''); print(p if isinstance(p,str) else ','.join(p) if isinstance(p,list) else '')" 2>/dev/null || echo "")
        preload_models="${sp}"
    fi

    # Export PRELOAD_MODELS env var for the Python server to read
    if [[ -n "${preload_models}" ]]; then
        export PRELOAD_MODELS="${preload_models}"
        log_info "Preload models: ${preload_models}"
    fi

    if [[ -n "${START_WATCHDOG}" ]]; then
        _run_with_watchdog "${model_dir}"
    else
        local api_key_arg=""
        if [[ -n "${API_KEY}" ]]; then
            api_key_arg="--api-key ${API_KEY}"
            export FUSION_MLX_API_KEY="${API_KEY}"
        fi
        fusion-mlx serve \
            --model-dir "${model_dir}" \
            --log-level INFO \
            --enable-prefix-cache \
            --continuous-batching \
            --chunked-prefill-tokens 4096 \
            $(host_port_args) \
            ${api_key_arg} \
            &
    fi

    local serve_pid=$!
    log_info "Server PID: ${serve_pid}"

    # With preload, increase health timeout (models can take 10-30s each)
    local health_timeout=120
    if [[ -n "${preload_models}" ]]; then
        local model_count
        model_count=$(echo "${preload_models}" | tr ',' '\n' | wc -l | tr -d ' ')
        health_timeout=$(( 120 + model_count * 60 ))
        log_info "Extended health timeout to ${health_timeout}s for ${model_count} preload models"
    fi

    if wait_healthy "${health_timeout}"; then
        log_info "Fusion-MLX v$(fusion-mlx version 2>/dev/null | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1) started successfully"
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
    do_start "$@"
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
        printf "${GREEN}● Running${NC}  PID=%s  ADDR=%s\n" "${pid}" "${HOST}"

        # Quick health check
        local health
        health=$(health_curl 2>/dev/null || echo "unreachable")
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
        resolve_api_key
        local models
        models=$(auth_curl "$(base_url)/v1/models" \
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
    'huggingface': {'endpoint': '${HF_MIRROR_DEFAULT}'},
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
hf['endpoint'] = '${HF_MIRROR_DEFAULT}'
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

# ── watchdog supervisor ─────────────────────────────────────────────
_run_with_watchdog() {
    local model_dir="$1"
    # PRELOAD_MODELS is already exported in the environment
    local backoff=1
    local max_backoff=30
    local crash_count=0
    local max_crashes=5
    local window=300
    local crash_timestamps=()

    log_info "Starting in watchdog mode (auto-restart on crash)"

    while true; do
        local now
        now=$(date +%s)

        # Prune old crash timestamps outside the window
        local fresh=()
        for ts in "${crash_timestamps[@]}"; do
            if (( now - ts < window )); then
                fresh+=("$ts")
            fi
        done
        crash_timestamps=("${fresh[@]}")
        crash_count=${#crash_timestamps[@]}

        if (( crash_count >= max_crashes )); then
            log_error "Too many crashes (${crash_count} in ${window}s), stopping watchdog"
            log_error "Check logs: ${LOG_DIR}/server.log"
            exit 1
        fi

        log_info "Launching fusion-mlx (attempt $((crash_count + 1)))..."
        local api_key_arg=""
        if [[ -n "${API_KEY:-}" ]]; then
            api_key_arg="--api-key ${API_KEY}"
        fi
        fusion-mlx serve \
            --model-dir "${model_dir}" \
            --log-level INFO \
            --enable-prefix-cache \
            --continuous-batching \
            --chunked-prefill-tokens 4096 \
            $(host_port_args) \
            ${api_key_arg}

        local exit_code=$?

        if wait_healthy 10 2>/dev/null; then
            # Clean exit (SIGTERM) — don't record as crash
            if (( exit_code == 0 || exit_code == 143 )); then
                log_info "Server exited cleanly (code=${exit_code})"
                break
            fi
        fi

        # Unexpected exit — record crash and backoff
        crash_timestamps+=("$(date +%s)")
        log_warn "Server exited unexpectedly (code=${exit_code}), restarting in ${backoff}s..."
        sleep "${backoff}"
        backoff=$(( backoff * 2 ))
        if (( backoff > max_backoff )); then
            backoff=${max_backoff}
        fi
    done
}

# ── launchd install/uninstall ──────────────────────────────────────
_LAUNCHD_PLIST="${HOME}/Library/LaunchAgents/com.fusion-mlx.server.plist"
_LAUNCHD_LABEL="com.fusion-mlx.server"

do_install_launchd() {
    if [[ -f "${_LAUNCHD_PLIST}" ]]; then
        log_warn "LaunchAgent already installed at ${_LAUNCHD_PLIST}"
        log_info "Use 'start.sh uninstall-launchd' to remove first"
        return 0
    fi

    resolve_hf_mirror

    local model_dir="${HOME}/.fusion-mlx/models"
    if [[ -f "${SETTINGS}" ]]; then
        local md
        md=$(python3 -c "import json; d=json.load(open('${SETTINGS}')); print(d.get('model',{}).get('model_dir','${model_dir}'))" 2>/dev/null || echo "${model_dir}")
        model_dir="${md}"
    fi

    mkdir -p "$(dirname "${_LAUNCHD_PLIST}")"
    mkdir -p "${LOG_DIR}"

    cat > "${_LAUNCHD_PLIST}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${_LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PROJ_DIR}/start.sh</string>
        <string>start</string>
        <string>--watchdog</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJ_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd.out.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd.err.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HF_ENDPOINT</key>
        <string>${HF_MIRROR}</string>
        <key>HF_MIRROR</key>
        <string>${HF_MIRROR}</string>
        <key>HUGGINGFACE_HUB_CACHE</key>
        <string>${HOME}/.fusion-mlx/models</string>
        <key>PRELOAD_MODELS</key>
        <string>${PRELOAD_MODELS:-}</string>
    </dict>
</dict>
</plist>
PLIST

    launchctl load "${_LAUNCHD_PLIST}" 2>/dev/null || true
    log_info "LaunchAgent installed and loaded: ${_LAUNCHD_PLIST}"
    log_info "Server will auto-start on login and restart on crash"
}

do_uninstall_launchd() {
    if [[ ! -f "${_LAUNCHD_PLIST}" ]]; then
        log_warn "No LaunchAgent found at ${_LAUNCHD_PLIST}"
        return 0
    fi

    launchctl unload "${_LAUNCHD_PLIST}" 2>/dev/null || true
    rm -f "${_LAUNCHD_PLIST}"
    log_info "LaunchAgent uninstalled"
}

# ── doctor ──────────────────────────────────────────────────────────
do_doctor() {
    ensure_venv
    fusion-mlx doctor 2>&1 || true
}

# ── Usage ───────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
start.sh — fusion-mlx lifecycle manager

Usage: start.sh <command> [args]

Commands:
  start [--watchdog] [--preload MODEL,MODEL,...]
                      Start fusion-mlx
                      --watchdog  Auto-restart on crash
                      --preload   Comma-separated models to preload at startup
  stop                Graceful stop (SIGTERM → SIGKILL fallback)
  restart             Stop + start (passes --preload to start)
  status              Show PID, port, memory, models, health
  log [N]             Tail server log (default 50 lines, -f to follow)
  errors              Show recent ERROR/CRITICAL from logs
  tune                Auto-tune settings.json for current hardware
  clean               Rotate old logs, clear __pycache__, trim error logs
  doctor              Run fusion-mlx doctor
  install-launchd     Install launchd LaunchAgent (auto-start + crash restart)
  uninstall-launchd   Remove launchd LaunchAgent
  help                Show this help

Environment:
  PORT            Server port (default: 11434; ignored in UDS mode)
  FUSION_HOST     Bind address (default: 127.0.0.1). Set to unix:/path/to.sock
                  for UDS listen mode (#351) - only a process with filesystem
                  access to the socket can reach MLX. TCP loopback is the default.
  HF_MIRROR       HuggingFace mirror override (default: read from config)
  PRELOAD_MODELS  Comma-separated models to preload (overrides --preload)

Preload Config (~/.fusion-mlx/settings.json):
  model.preload   Array or comma-string of models to preload at startup
  Priority: --preload flag > PRELOAD_MODELS env > settings.json

Mirror Config (~/.fusion-mlx/settings.json):
  huggingface.endpoint  HF mirror URL (default: https://hf-mirror.com)
  Priority: HF_MIRROR env var > settings.json > built-in default
EOF
}

# ── Main ────────────────────────────────────────────────────────────
cmd="${1:-help}"
shift || true

case "${cmd}" in
    start)             do_start "$@" ;;
    stop)              do_stop    ;;
    restart)           do_restart ;;
    status)            show_status ;;
    log)               show_log "${1:-}" ;;
    errors)            show_errors ;;
    tune)              do_tune    ;;
    clean)             do_clean   ;;
    doctor)            do_doctor  ;;
    install-launchd)   do_install_launchd ;;
    uninstall-launchd) do_uninstall_launchd ;;
    help|-h|--help)    usage ;;
    *)
        log_error "Unknown command: ${cmd}"
        usage
        exit 1
        ;;
esac
