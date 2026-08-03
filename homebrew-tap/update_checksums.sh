#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?Usage: $0 <version>}"
FORMULA="homebrew-tap/Formula/fusion-mlx.rb"

if [ ! -f "$FORMULA" ]; then
    echo "ERROR: $FORMULA not found" >&2
    exit 1
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

replace_sha() {
    local placeholder="$1"
    local value="$2"
    sed -i '' "s/${placeholder}/${value}/" "$FORMULA"
    log "  ${placeholder} → ${value:0:12}..."
}

# Main tarball
log "Fetching release tarball SHA256 for v${VERSION}..."
TARBALL_URL="https://github.com/dahai80/fusion-mlx/archive/refs/tags/v${VERSION}.tar.gz"
TARBALL_SHA=$(curl -sL "$TARBALL_URL" | shasum -a 256 | cut -d' ' -f1)
if [ -z "$TARBALL_SHA" ]; then
    echo "ERROR: Could not fetch tarball from $TARBALL_URL" >&2
    echo "Make sure the tag v${VERSION} exists on GitHub." >&2
    exit 1
fi
replace_sha "PLACEHOLDER_SHA256" "$TARBALL_SHA"

# PyPI wheels — fetch SHA256 from PyPI JSON API
fetch_pypi_sha() {
    local pkg="$1"
    local version="$2"
    curl -sL "https://pypi.org/pypi/${pkg}/${version}/json" \
        | python3 -c "
import json, sys
data = json.load(sys.stdin)
for f in data.get('urls', []):
    if 'macosx_14_0_arm64' in f['filename'] and f['filename'].endswith('.whl'):
        print(f['digests']['sha256']); sys.exit(0)
for f in data.get('urls', []):
    if f['filename'].endswith('.whl') and 'any' in f['filename']:
        print(f['digests']['sha256']); sys.exit(0)
print('', file=sys.stderr); sys.exit(1)
" 2>/dev/null
}

# mlx
log "Fetching mlx SHA256..."
MLX_SHA=$(fetch_pypi_sha "mlx" "0.32.0")
if [ -n "$MLX_SHA" ]; then
    replace_sha "PLACEHOLDER_MLX_SHA256" "$MLX_SHA"
else
    log "  WARNING: Could not fetch mlx SHA256 — update manually"
fi

# safetensors
log "Fetching safetensors SHA256..."
ST_SHA=$(fetch_pypi_sha "safetensors" "0.5.3")
if [ -n "$ST_SHA" ]; then
    replace_sha "PLACEHOLDER_SAFETENSORS_SHA256" "$ST_SHA"
else
    log "  WARNING: Could not fetch safetensors SHA256 — update manually"
fi

# Git-pinned resources — compute SHA256 from archive
fetch_git_sha() {
    local repo="$1"
    local commit="$2"
    local archive_url="https://github.com/${repo}/archive/${commit}.tar.gz"
    curl -sL "$archive_url" | shasum -a 256 | cut -d' ' -f1
}

log "Computing git-pinned resource SHA256s..."

MLX_LM_COMMIT=$(grep -A2 'resource "mlx-lm"' "$FORMULA" | grep 'commit:' | sed 's/.*commit: "//;s/".*//')
MLX_EMB_COMMIT=$(grep -A2 'resource "mlx-embeddings"' "$FORMULA" | grep 'commit:' | sed 's/.*commit: "//;s/".*//')
MLX_VLM_COMMIT=$(grep -A2 'resource "mlx-vlm"' "$FORMULA" | grep 'commit:' | sed 's/.*commit: "//;s/".*//')
DFLASH_COMMIT=$(grep -A2 'resource "dflash-mlx"' "$FORMULA" | grep 'commit:' | sed 's/.*commit: "//;s/".*//')
MLX_AUDIO_COMMIT=$(grep -A2 'resource "mlx-audio"' "$FORMULA" | grep 'commit:' | sed 's/.*commit: "//;s/".*//')

for pair in \
    "ml-explore/mlx-lm:$MLX_LM_COMMIT:PLACEHOLDER_MLX_LM_SHA256" \
    "Blaizzy/mlx-embeddings:$MLX_EMB_COMMIT:PLACEHOLDER_MLX_EMB_SHA256" \
    "Blaizzy/mlx-vlm:$MLX_VLM_COMMIT:PLACEHOLDER_MLX_VLM_SHA256" \
    "bstnxbt/dflash-mlx:$DFLASH_COMMIT:PLACEHOLDER_DFLASH_SHA256" \
    "Blaizzy/mlx-audio:$MLX_AUDIO_COMMIT:PLACEHOLDER_MLX_AUDIO_SHA256"
do
    repo=$(echo "$pair" | cut -d: -f1)
    commit=$(echo "$pair" | cut -d: -f2)
    placeholder=$(echo "$pair" | cut -d: -f3)
    log "  $repo @ ${commit:0:8}..."
    sha=$(fetch_git_sha "$repo" "$commit")
    if [ -n "$sha" ]; then
        replace_sha "$placeholder" "$sha"
    else
        log "  WARNING: Could not fetch SHA for $repo — update manually"
    fi
done

# Update version in Formula if different
CURRENT_VERSION=$(grep -o 'v[0-9]\+\.[0-9]\+\.[0-9]\+' "$FORMULA" | head -1 | tr -d 'v')
if [ "$CURRENT_VERSION" != "$VERSION" ]; then
    sed -i '' "s/v${CURRENT_VERSION}/v${VERSION}/g" "$FORMULA"
    log "Version updated: ${CURRENT_VERSION} → ${VERSION}"
fi

# Verify no placeholders remain
REMAINING=$(grep -c 'PLACEHOLDER' "$FORMULA" || true)
if [ "$REMAINING" -gt 0 ]; then
    echo ""
    echo "WARNING: ${REMAINING} PLACEHOLDER(s) remain in $FORMULA"
    echo "Run: grep PLACEHOLDER $FORMULA"
else
    echo ""
    log "All SHA256 values populated. Formula ready."
fi
