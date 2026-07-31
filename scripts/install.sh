#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# fusion-mlx installer — curl one-liner:
#   curl -fsSL https://raw.githubusercontent.com/dahai80/fusion-mlx/main/scripts/install.sh | bash
#
# Supports: macOS (Apple Silicon), Python 3.11+
# Install methods: pip (default), uv (if available)
# Auto-detects RAM and recommends a default model.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { printf "${GREEN}[fusion-mlx]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[fusion-mlx]${NC} %s\n" "$*"; }
err()  { printf "${RED}[fusion-mlx]${NC} %s\n" "$*" >&2; }

# ── Platform check ──────────────────────────────────────────────
if [[ "$(uname)" != "Darwin" ]]; then
    err "fusion-mlx requires macOS (Apple Silicon). Detected: $(uname)"
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    err "fusion-mlx requires Apple Silicon (arm64). Detected: $(uname -m)"
    exit 1
fi

# ── Python detection ────────────────────────────────────────────
detect_python() {
    local candidates=("python3" "python3.13" "python3.12" "python3.11")
    for cmd in "${candidates[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            local major="${ver%%.*}"
            local minor="${ver#*.}"
            if [[ "$major" -eq 3 && "$minor" -ge 11 ]]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON=$(detect_python)
if [[ -z "$PYTHON" ]]; then
    err "Python 3.11+ not found. Install via:"
    err "  brew install python@3.13"
    err "  or download from https://www.python.org/downloads/"
    exit 1
fi

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log "Using Python $PY_VER at $(command -v "$PYTHON")"

# ── RAM detection ───────────────────────────────────────────────
detect_ram_gb() {
    local bytes
    bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    echo $((bytes / 1073741824))
}

recommend_model() {
    local ram_gb="$1"
    if [[ "$ram_gb" -ge 64 ]]; then
        echo "qwen3.5-72b-4bit"
    elif [[ "$ram_gb" -ge 32 ]]; then
        echo "qwen3.5-27b-8bit"
    elif [[ "$ram_gb" -ge 16 ]]; then
        echo "qwen3.5-9b-4bit"
    elif [[ "$ram_gb" -ge 8 ]]; then
        echo "qwen3.5-4b-4bit"
    else
        echo "qwen3.5-4b-4bit"
        warn "Less than 8 GB RAM detected — model quality will be limited"
    fi
}

RAM_GB=$(detect_ram_gb)
MODEL=$(recommend_model "$RAM_GB")
log "Detected ${RAM_GB} GB RAM → recommended model: $MODEL"

# ── Install method ──────────────────────────────────────────────
USE_UV=false
if command -v uv &>/dev/null; then
    USE_UV=true
    log "Found uv — will use it for faster installation"
fi

INSTALL_DIR="${FUSION_MLX_INSTALL_DIR:-$HOME/.fusion-mlx}"
VENV_DIR="$INSTALL_DIR/.venv"

# ── Install ─────────────────────────────────────────────────────
if [[ "$USE_UV" == true ]]; then
    log "Installing fusion-mlx via uv..."
    if ! command -v fusion-mlx &>/dev/null; then
        uv tool install fusion-mlx 2>/dev/null || {
            log "uv tool install failed, falling back to pip in venv..."
            USE_UV=false
        }
    else
        log "fusion-mlx already installed via uv, updating..."
        uv tool upgrade fusion-mlx 2>/dev/null || true
    fi
fi

if [[ "$USE_UV" == false ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtual environment at $VENV_DIR..."
        "$PYTHON" -m venv "$VENV_DIR"
    fi

    VENV_PYTHON="$VENV_DIR/bin/python"

    log "Upgrading pip..."
    "$VENV_PYTHON" -m pip install --upgrade pip --quiet 2>/dev/null

    log "Installing fusion-mlx..."
    "$VENV_PYTHON" -m pip install fusion-mlx --quiet 2>/dev/null || {
        err "pip install failed. Try manually:"
        err "  $VENV_PYTHON -m pip install fusion-mlx"
        exit 1
    }

    # Symlink into a PATH-visible location
    LINK_DIR="$INSTALL_DIR/bin"
    mkdir -p "$LINK_DIR"
    ln -sf "$VENV_DIR/bin/fusion-mlx" "$LINK_DIR/fusion-mlx" 2>/dev/null || true
    ln -sf "$VENV_DIR/bin/fm" "$LINK_DIR/fm" 2>/dev/null || true

    # Shell profile injection
    PROFILE_INJECTED=false
    for rcfile in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile"; do
        if [[ -f "$rcfile" ]] && ! grep -q "fusion-mlx/bin" "$rcfile" 2>/dev/null; then
            echo "" >> "$rcfile"
            echo "# fusion-mlx" >> "$rcfile"
            echo "export PATH=\"\$HOME/.fusion-mlx/bin:\$PATH\"" >> "$rcfile"
            PROFILE_INJECTED=true
            log "Added PATH to $rcfile"
            break
        fi
    done
fi

# ── Verify ──────────────────────────────────────────────────────
if command -v fusion-mlx &>/dev/null; then
    FM_CMD="fusion-mlx"
elif [[ -x "$LINK_DIR/fusion-mlx" ]]; then
    FM_CMD="$LINK_DIR/fusion-mlx"
elif [[ -x "$VENV_DIR/bin/fusion-mlx" ]]; then
    FM_CMD="$VENV_DIR/bin/fusion-mlx"
else
    err "Installation completed but fusion-mlx command not found in PATH"
    err "Restart your shell or run: source ~/.zshrc"
    exit 1
fi

INSTALLED_VER=$("$FM_CMD" --version 2>/dev/null || echo "unknown")
log "Installed fusion-mlx $INSTALLED_VER"

# ── Summary ─────────────────────────────────────────────────────
echo ""
printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
printf "${GREEN}  fusion-mlx %s installed!${NC}\n" "$INSTALLED_VER"
printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
echo ""
printf "  Quick start:\n"
echo ""
printf "  ${YELLOW}fusion-mlx chat${NC}              # interactive REPL (default: $MODEL)\n"
printf "  ${YELLOW}fusion-mlx serve %s${NC}   # OpenAI-compatible server\n" "$MODEL"
printf "  ${YELLOW}fusion-mlx doctor${NC}            # check environment health\n"
printf "  ${YELLOW}fusion-mlx models${NC}            # list available models\n"
echo ""
if [[ "$PROFILE_INJECTED" == true ]]; then
    printf "  ${YELLOW}source ~/.zshrc${NC}  (or restart your shell to update PATH)\n"
    echo ""
fi
printf "  Docs: https://github.com/dahai80/fusion-mlx\n"
printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
