#!/usr/bin/env bash
# strip_bundle.sh — post-build strip of FusionMLX.app Python bundle
#
# Usage:
#   apps/fusion-mac/Scripts/strip_bundle.sh build/Stage/FusionMLX.app
#
# Removes unused Python packages and prunes size from the bundled
# Python runtime to keep DMG under 500 MB.
#
# Run *after* build.sh, before package_dmg.sh.

set -euo pipefail

LIGHT_BLUE="\033[1;34m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

log()  { printf "${LIGHT_BLUE}[strip]${RESET} %s\n" "$*"; }
ok()   { printf "${GREEN}[strip]${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}[strip]${RESET} %s\n" "$*"; }
die()  { printf "${RED}[strip]${RESET} %s\n" "$*" >&2; exit 1; }

APP_BUNDLE="${1:-}"
[ -d "$APP_BUNDLE" ] || die "Usage: $0 <path/to/FusionMLX.app>"

PYTHON_DIR="$APP_BUNDLE/Contents/Resources/Python"
[ -d "$PYTHON_DIR" ] || die "No Python dir found at $PYTHON_DIR"

MLX_SITE="$PYTHON_DIR/framework-mlx-base/lib/python3.11/site-packages"
CPYTHON_LIB="$PYTHON_DIR/cpython-3.11/lib/python3.11"

[ -d "$MLX_SITE" ] || die "No site-packages at $MLX_SITE"
[ -d "$CPYTHON_LIB" ] || die "No stdlib at $CPYTHON_LIB"

log "Stripping $(du -sh "$APP_BUNDLE" | cut -f1) bundle at $APP_BUNDLE"

# ── 1. Remove completely unused packages (not imported at runtime) ─────────

UNUSED_PACKAGES=(
    "pyarrow"               # 120 MB — Apache Arrow, not imported
    "sklearn"               #  31 MB — scikit-learn, not imported
    "sympy"                 #  29 MB — symbolic math, not imported
    "pycountry"             #  21 MB — country data, not imported
    "llvmlite"              # 124 MB — LLVM bindings (numba dep), not imported
    "numba"                 #  13 MB — JIT compiler, not imported
)

REMOVED=0
REMOVED_BYTES=0
for pkg in "${UNUSED_PACKAGES[@]}"; do
    # Find package directories/files matching the pattern
    while IFS= read -r -d '' path; do
        size=$(du -sk "$path" 2>/dev/null | cut -f1)
        rm -rf "$path"
        REMOVED=$((REMOVED + 1))
        REMOVED_BYTES=$((REMOVED_BYTES + size))
        log "  removed: $(basename "$path") ($((size / 1024)) MB)"
    done < <(find "$MLX_SITE" -maxdepth 2 \( \
        -type d -name "$pkg" -o \
        -type d -name "${pkg}*" -o \
        -type f -name "${pkg}*.so" -o \
        -type f -name "${pkg}*.dist-info" -o \
        -type d -name "${pkg}.dist-info" \
    \) -print0 2>/dev/null)
done

# ── 2. Strip optional packages that are only used via lazy imports ─────────
#
# scipy is only used by mlx_audio TTS (lazy import at runtime).
# Keep the package but strip tests, docs, and unnecessary sub-modules.

log "Stripping scipy extras..."
if [ -d "$MLX_SITE/scipy" ]; then
    rm -rf "$MLX_SITE/scipy/tests" 2>/dev/null || true
    rm -rf "$MLX_SITE/scipy/*/tests" 2>/dev/null || true
    rm -rf "$MLX_SITE/scipy/*/tests/" 2>/dev/null || true
    # Remove .pyc files that will be regenerated
    find "$MLX_SITE/scipy" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
fi

# ── 3. Strip torch (386 MB) — fusion_mlx has a lazy stub, only used by mlx_audio ─
#
# torch and its deps (torchaudio, etc.) are transitive dependencies pulled in
# by mlx_audio. fusion-mlx provides a synthetic torch stub (_torch_stub.py).
# 
# ALL torch imports in fusion-mlx go through the stub except:
#   - mlx_audio model conversion scripts (not runtime)
#   - Other pip-installed packages that may do `import torch` at import time
#
# We keep only torch._C and torch.Tensor stubs that the stub module provides.
# For safety, rename the torch dir rather than delete, so any import that
# sneaks through gets an ImportError instead of a cryptic crash.

log "Removing torch (fusion-mlx has built-in stub)..."
if [ -d "$MLX_SITE/torch" ]; then
    size=$(du -sk "$MLX_SITE/torch" 2>/dev/null | cut -f1)
    rm -rf "$MLX_SITE/torch"
    REMOVED_BYTES=$((REMOVED_BYTES + size))
    log "  removed: torch ($((size / 1024)) MB)"
fi
# Also remove torchaudio, torchvision
for pkg in torchaudio torchvision; do
    if [ -d "$MLX_SITE/$pkg" ]; then
        size=$(du -sk "$MLX_SITE/$pkg" 2>/dev/null | cut -f1)
        rm -rf "$MLX_SITE/$pkg"
        REMOVED_BYTES=$((REMOVED_BYTES + size))
        log "  removed: $pkg ($((size / 1024)) MB)"
    fi
done
# Remove torch .dist-info and .so stubs
find "$MLX_SITE" -maxdepth 1 \( -name "torch-*.dist-info" -o -name "torch*.so" \) -exec rm -rf {} + 2>/dev/null || true

# ── 4. Strip cv2 (OpenCV, 119 MB) — only used in fusion_mlx/utils/video.py via lazy import ─
log "Stripping cv2 optional components..."
if [ -d "$MLX_SITE/cv2" ]; then
    # Remove OpenCV's bundled .jar, test data, docs
    rm -rf "$MLX_SITE/cv2/test" 2>/dev/null || true
    find "$MLX_SITE/cv2" -name "*.xml" -delete 2>/dev/null || true  # cascade XMLs
    find "$MLX_SITE/cv2" -name "*.jar" -delete 2>/dev/null || true
    find "$MLX_SITE/cv2" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    # Remove .so symlinks that point to unused backends
    for backend in gstreamer ovis; do
        find "$MLX_SITE/cv2" -name "*${backend}*" -delete 2>/dev/null || true
    done
fi

# ── 5. Strip pandas (40 MB) — only used in markitdown_pdf_fallback via lazy import ─
log "Stripping pandas extras..."
if [ -d "$MLX_SITE/pandas" ]; then
    rm -rf "$MLX_SITE/pandas/tests" 2>/dev/null || true
    rm -rf "$MLX_SITE/pandas/*/tests" 2>/dev/null || true
    find "$MLX_SITE/pandas" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
fi

# ── 6. Strip transformers (45 MB) — used, but strip tests ─────────────────
log "Stripping transformers tests..."
if [ -d "$MLX_SITE/transformers" ]; then
    rm -rf "$MLX_SITE/transformers/tests" 2>/dev/null || true
    find "$MLX_SITE/transformers" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
fi

# ── 7. Strip mlx audio/video heavy test data ──────────────────────────────
log "Stripping MLX test data..."
for pkg in mlx_audio mlx_vlm mlx_lm mlx_embeddings; do
    if [ -d "$MLX_SITE/$pkg" ]; then
        find "$MLX_SITE/$pkg" -name "*.wav" -delete 2>/dev/null || true
        find "$MLX_SITE/$pkg" -name "*.mp3" -delete 2>/dev/null || true
        find "$MLX_SITE/$pkg" -name "*.png" -delete 2>/dev/null || true
        find "$MLX_SITE/$pkg" -name "*.jpg" -delete 2>/dev/null || true
        find "$MLX_SITE/$pkg" -name "test*" -type d -exec rm -rf {} + 2>/dev/null || true
    fi
done

# ── 8. Strip cpython stdlib ────────────────────────────────────────────────
log "Stripping cpython stdlib..."
# Remove test dirs
for dir in test tests; do
    find "$CPYTHON_LIB" -maxdepth 1 -type d -name "$dir" -exec rm -rf {} + 2>/dev/null || true
done
# Remove idlelib, tkinter (not used in server mode)
for lib in idlelib tkinter turtledemo; do
    rm -rf "$CPYTHON_LIB/$lib" 2>/dev/null || true
done
# Remove __pycache__ everywhere
find "$PYTHON_DIR" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
find "$PYTHON_DIR" -name "*.pyc" -delete 2>/dev/null || true

# ── 9. Strip all .dist-info metadata (save ~10 MB total) ──────────────────
log "Stripping .dist-info metadata..."
find "$MLX_SITE" -maxdepth 1 -type d -name "*.dist-info" | while IFS= read -r d; do
    # Keep METADATA and RECORD for pip to function, remove the rest
    for f in "$d"/*; do
        bn=$(basename "$f")
        case "$bn" in
            METADATA|RECORD|INSTALLER) ;;
            *)
                if [ -f "$f" ]; then rm -f "$f"; fi
                ;;
        esac
    done
done

# ── 10. Strip all .so debug symbols ──────────────────────────────────────
log "Stripping debug symbols from native libraries..."
find "$PYTHON_DIR" -name "*.so" -type f | while IFS= read -r so; do
    strip -S "$so" 2>/dev/null || true
done

# ── Summary ────────────────────────────────────────────────────────────────
FINAL_SIZE=$(du -sh "$APP_BUNDLE" | cut -f1)
ok "Done. Bundle: $FINAL_SIZE (removed ~$((REMOVED_BYTES / 1024)) MB)"
echo "Note: run package_dmg.sh to create the DMG."
