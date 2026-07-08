#!/usr/bin/env bash
# package_dmg.sh — package FusionMLX.app into a distributable DMG.
#
# Prerequisites:
#   - FusionMLX.app built via build.sh (at build/Stage/FusionMLX.app)
#   - If --build flag is given, runs build.sh first
#
# Usage:
#   Scripts/package_dmg.sh              # package existing build
#   Scripts/package_dmg.sh --build      # build + package
#
# Output:
#   build/dist/FusionMLX-<version>-macos<major>-<codename>.dmg
#
# DMG naming matches ReleasesChecker.findMatchingDMG:
#   FusionMLX-0.1.0-macos26-tahoe.dmg
#   FusionMLX-0.1.0-macos15-sequoia.dmg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

# --- Colors ---
LIGHT_BLUE="\033[1;34m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

log()  { printf "${LIGHT_BLUE}[dmg]${RESET} %s\n" "$*"; }
ok()   { printf "${GREEN}[dmg]${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}[dmg]${RESET} %s\n" "$*"; }
die()  { printf "${RED}[dmg ERROR]${RESET} %s\n" "$*" >&2; exit 1; }

# Shared embedded Mach-O signing helpers (per-file; replaces the deprecated
# `codesign --deep` which left stale page hashes on nested .so and caused
# the server to SIGKILL on launch - see note at the sign step below).
# shellcheck source=sign_utils.sh
source "$SCRIPT_DIR/sign_utils.sh"

# --- Parse args ---
DO_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --build) DO_BUILD=1 ;;
        *) die "unknown flag '$arg'" ;;
    esac
done

# --- Version ---
VERSION_FILE="$REPO_ROOT/fusion_mlx/_version.py"
[ -f "$VERSION_FILE" ] || die "missing $VERSION_FILE"
APP_VERSION=$(grep -oE '__version__[[:space:]]*=[[:space:]]*"[^"]+"' "$VERSION_FILE" | \
              sed -E 's/.*"([^"]+)".*/\1/')
[ -n "$APP_VERSION" ] || die "could not parse __version__ from $VERSION_FILE"

# --- macOS version + codename ---
MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)

case "$MACOS_MAJOR" in
    15) CODENAME="sequoia" ;;
    26) CODENAME="tahoe" ;;
    27) CODENAME="sunrise" ;;
    *)  CODENAME="macos${MACOS_MAJOR}" ;;
esac

DMG_NAME="FusionMLX-${APP_VERSION}-macos${MACOS_MAJOR}-${CODENAME}"
STAGED_APP="$PROJECT_DIR/build/Stage/FusionMLX.app"
DIST_DIR="$PROJECT_DIR/build/dist"
DMG_PATH="${DIST_DIR}/${DMG_NAME}.dmg"

log "Version: $APP_VERSION"
log "macOS:   $MACOS_MAJOR ($CODENAME)"
log "Output:  $DMG_PATH"

# --- Optionally build ---
if [ "$DO_BUILD" -eq 1 ]; then
    log "Running build.sh release…"
    "$SCRIPT_DIR/build.sh" release
fi

# --- Verify app bundle ---
[ -d "$STAGED_APP" ] || die "FusionMLX.app not found at $STAGED_APP — run with --build or build.sh first"
APP_BINARY="$STAGED_APP/Contents/MacOS/FusionMLX"
[ -x "$APP_BINARY" ] || die "App binary missing or not executable: $APP_BINARY"

# --- Strip Python bundle for size (target < 500 MB DMG) ---
STRIP_SCRIPT="$SCRIPT_DIR/strip_bundle.sh"
if [ -x "$STRIP_SCRIPT" ]; then
    log "Stripping Python bundle…"
    "$STRIP_SCRIPT" "$STAGED_APP"
else
    warn "strip_bundle.sh not found; skipping size optimization"
fi

APP_SIZE=$(du -sh "$STAGED_APP" | cut -f1)
log "App bundle after strip: $STAGED_APP ($APP_SIZE)"

# --- Ad-hoc sign (per-file embedded + flat bundle seal) ---
#
# strip_bundle.sh just mutated the bundle, so signatures must be re-applied
# AFTER stripping. `codesign --force --sign - --deep` is deprecated and does
# NOT reliably re-sign the nested .so/.dylib under Resources/Python - it
# leaves stale page hashes, and macOS SIGKILLs the server child (exit 9,
# "Code Signature Invalid" / "Invalid Page") on the first dlopen. Sign each
# embedded Mach-O explicitly, seal the bundle flat, then verify mlx.core so a
# broken bundle never ships.
PYTHON_DIR="$STAGED_APP/Contents/Resources/Python"
MLX_SITE="$PYTHON_DIR/framework-mlx-base/lib/python3.11/site-packages"
log "Ad-hoc signing embedded native code (per-file)…"
_sign_embedded_mach_o_files "$PYTHON_DIR"
codesign --force --sign - "$STAGED_APP/Contents/MacOS/fusion-cli" >/dev/null 2>&1
_verify_embedded_signatures "$MLX_SITE"
log "Ad-hoc resigning app bundle (flat seal)…"
codesign --force --sign - "$STAGED_APP"
xattr -dr com.apple.quarantine "$STAGED_APP" 2>/dev/null || true
ok "Signed"

# --- Create DMG ---
log "Creating DMG…"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Temp staging dir for DMG contents
DMG_STAGING=$(mktemp -d)
trap 'rm -rf "$DMG_STAGING"' EXIT

# Copy app + Applications symlink
ditto "$STAGED_APP" "$DMG_STAGING/FusionMLX.app"
ln -s /Applications "$DMG_STAGING/Applications"

# Create volume icon position DS_Store using AppleScript
# This gives the classic "drag app to Applications" layout
log "Setting DMG window layout…"
DMG_RW="${DIST_DIR}/${DMG_NAME}-rw.dmg"
DMG_SIZE=$(du -sm "$DMG_STAGING" | cut -f1)
DMG_SIZE=$((DMG_SIZE + 512))  # padding (HFS+ overhead needs ample slack on 2GB+ bundles)

hdiutil create \
    -srcfolder "$DMG_STAGING" \
    -volname "FusionMLX" \
    -fs HFS+ \
    -fsargs "-c c=64,a=16,e=16" \
    -format UDRW \
    -size "${DMG_SIZE}m" \
    "$DMG_RW" >/dev/null

# Mount and customize window
MOUNT_DIR=$(hdiutil attach "$DMG_RW" -readwrite -nobrowse | \
            grep "/Volumes/FusionMLX" | awk '{print $NF}')

# Set window size, icon positions via AppleScript
osascript <<APPLESCRIPT >/dev/null 2>&1 || warn "AppleScript window layout failed (non-fatal)"
tell application "Finder"
    tell disk "FusionMLX"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {400, 100, 1000, 500}
        set theViewOptions to the icon view options of container window
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 96
        set position of item "FusionMLX.app" of container window to {150, 200}
        set position of item "Applications" of container window to {450, 200}
        close
        open
        update without registering applications
        delay 2
    end tell
end tell
APPLESCRIPT

# Sync and detach
sync
hdiutil detach "$MOUNT_DIR" -quiet

# Convert to compressed read-only (LZMA for max compression)
log "Compressing DMG with LZMA…"
hdiutil convert "$DMG_RW" \
    -format ULFO \
    -imagekey lzma-level=9 \
    -o "$DMG_PATH" >/dev/null

rm -f "$DMG_RW"

# --- Verify ---
DMG_FINAL_SIZE=$(du -sh "$DMG_PATH" | cut -f1)
ok "DMG created: $DMG_PATH ($DMG_FINAL_SIZE)"

# Quick integrity check
hdiutil verify "$DMG_PATH" >/dev/null 2>&1 && ok "DMG verified" || warn "DMG verify failed"

echo
echo "To distribute:"
echo "  $DMG_PATH"
echo
echo "Users can:"
echo "  1. Double-click to mount"
echo "  2. Drag FusionMLX.app to /Applications"
echo "  3. Right-click → Open (first launch, bypasses Gatekeeper)"
echo "  4. Or: xattr -cr /Applications/FusionMLX.app"
