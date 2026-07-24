#!/usr/bin/env bash
# Verify that git-pinned dependencies in pyproject.toml match expected commits.
# The commit SHA in a git+ URL IS the integrity check (pip has no --hash for VCS).
# Usage: ./scripts/verify_git_pins.sh
set -euo pipefail

ok=0
fail=0

check_pin() {
    local pkg="$1" expected="$2"
    local actual
    actual=$(grep -E "^\\s*\"?${pkg} @ git\\+" pyproject.toml \
        | sed -E 's/.*@([a-f0-9]{40}).*/\1/' \
        | head -1)
    if [ "$actual" = "$expected" ]; then
        echo "OK  ${pkg}=${actual}"
        ok=$((ok + 1))
    else
        echo "FAIL  ${pkg}: expected ${expected}, got ${actual:-MISSING}"
        fail=$((fail + 1))
    fi
}

check_pin "mlx-lm"         "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd"
check_pin "mlx-embeddings"  "32981fa4e8064ed664b52071789dd18271fe4206"
check_pin "mlx-vlm"         "f96138eef1f5ce7fb5d97f8dd41a664a195b5659"
check_pin "dflash-mlx"      "1ba671372b289c025b435c1a13aabb4bfb80b183"
check_pin "mlx-audio\[tts,stt,sts\]" "51753266e0a4f766fd5e6fbc46652224efc23981"

echo "---"
echo "${ok} OK, ${fail} FAIL"
if [ "$fail" -gt 0 ]; then
    exit 1
fi
