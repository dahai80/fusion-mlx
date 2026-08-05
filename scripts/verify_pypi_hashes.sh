#!/usr/bin/env bash
# Verify PyPI-pinned dependencies: the SHA256 of each pinned wheel/sdist on
# pypi.org MUST match the locked value below. This replaces the former
# git-commit-pin integrity check (git pins broke PyPI's no-direct-dependency
# policy, #348). The locked SHA256 values are the supply-chain integrity check.
#
# Usage: bash scripts/verify_pypi_hashes.sh
# Exits non-zero if any pinned file's PyPI-published SHA256 diverges from the
# locked value, or if a pinned version is missing from PyPI.
set -euo pipefail

ok=0
fail=0

# Each line: <package> <version> <expected_wheel_sha256> <expected_sdist_sha256>
# Expected values captured 2026-08-05 from https://pypi.org/pypi/<pkg>/<ver>/json
PINS=(
    "mlx-lm 0.31.3 758cfddf1180053b7613db76fad3d246a331a2a905808e1164a275621fc983b8 61eb0e3ba09444f77f874aff295401d7ccd20b39495cbbce0c782a15474ce733"
    "mlx-embeddings 0.1.0 3fe1feaa786d3b546ccd8909f6b4c22bd3bcce097616fce83173ededd30e6630 f80c1e1be26ff7bd22b15c1fba4cc03afd44c86e00a431b5fa75ffd7500affb1"
    "mlx-vlm 0.5.0 3351d6ccf609cbf57a4c8cd8308e9a1ce469883d8679d9968c6c6f77af016419 24563cd1b3a399fd941b2359100628306e2754db1b48780516d1283138258793"
    "dflash-mlx 0.1.7 896eb4fe1a2c509e3e12c217f369b88ca5736f6265e059a62ef5e33f95b27885 6b2f934db6992163e559749538fd8d1260054715df7281033456c3672e507c6b"
    "mlx-audio 0.4.3 6b87bf42d79d9ceb6b9310a77656b9b76429c2d6ddd89f634b2786c58a2e4721 8e87badf56a0f73bf91e3797b1195c01440a181cf0b64a2a08dc1bda4b037f54"
)

check_pin() {
    local pkg="$1" ver="$2" exp_wheel="$3" exp_sdist="$4"
    local url="https://pypi.org/pypi/${pkg}/${ver}/json"
    local json
    if ! json=$(curl -fsS --max-time 30 "$url" 2>/dev/null); then
        echo "FAIL  ${pkg}==${ver}: cannot fetch ${url}"
        fail=$((fail + 1))
        return
    fi
    local got_wheel got_sdist
    got_wheel=$(printf '%s' "$json" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(next((f['digests']['sha256'] for f in d.get('urls',[]) if f['packagetype']=='bdist_wheel'), ''))" 2>/dev/null || echo "")
    got_sdist=$(printf '%s' "$json" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(next((f['digests']['sha256'] for f in d.get('urls',[]) if f['packagetype']=='sdist'), ''))" 2>/dev/null || echo "")

    local bad=0
    if [ -z "$got_wheel" ]; then
        echo "FAIL  ${pkg}==${ver}: no bdist_wheel on PyPI"
        bad=1
    elif [ "$got_wheel" != "$exp_wheel" ]; then
        echo "FAIL  ${pkg}==${ver} wheel: expected ${exp_wheel}, got ${got_wheel}"
        bad=1
    fi
    if [ -z "$got_sdist" ]; then
        echo "FAIL  ${pkg}==${ver}: no sdist on PyPI"
        bad=1
    elif [ "$got_sdist" != "$exp_sdist" ]; then
        echo "FAIL  ${pkg}==${ver} sdist: expected ${exp_sdist}, got ${got_sdist}"
        bad=1
    fi
    if [ "$bad" -eq 0 ]; then
        echo "OK    ${pkg}==${ver}  wheel=${got_wheel:0:12}...  sdist=${got_sdist:0:12}..."
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
    fi
}

echo "Verifying ${#PINS[@]} PyPI-pinned dependencies against locked SHA256..."
for line in "${PINS[@]}"; do
    # shellcheck disable=SC2086
    check_pin $line
done

echo "---"
echo "${ok} OK, ${fail} FAIL"
if [ "$fail" -gt 0 ]; then
    exit 1
fi
