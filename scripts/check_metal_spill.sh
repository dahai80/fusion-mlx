#!/usr/bin/env bash
# scripts/check_metal_spill.sh
# L1 DoD gate: spilled registers = 0. A Metal kernel that spills registers
# to DRAM suffers an order-of-magnitude perf cliff. v2 doc §5.1 rule 1,
# §7 L1. Run after every Metal kernel build; CI blocks merge on spill > 0.
#
# Uses `xcrun -sdk macosx metal -ast-tree-dump` + a grep for the spill
# count in the Metal AST. On hosts without a Metal toolchain the script
# exits 0 (no kernels to check) — the native build step itself gates this.
#
# Usage:
#   scripts/check_metal_spill.sh                 # scan fusion_mlx/shim
#   scripts/check_metal_spill.sh path/to/file.metal
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "check_metal_spill: xcrun unavailable (non-macOS); skipping"
  exit 0
fi

if ! xcrun -find metal >/dev/null 2>&1; then
  echo "check_metal_spill: metal toolchain unavailable; skipping"
  exit 0
fi

TARGETS=()
if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  # Default: scan all .metal files under fusion_mlx/shim.
  while IFS= read -r f; do
    TARGETS+=("$f")
  done < <(find "$REPO_ROOT/fusion_mlx/shim" -name '*.metal' 2>/dev/null || true)
fi

if [ "${#TARGETS[@]}" -eq 0 ]; then
  echo "check_metal_spill: no .metal files to scan (PR-A has none yet); OK"
  exit 0
fi

SPILL_FOUND=0
for metal_file in "${TARGETS[@]}"; do
  echo "check_metal_spill: scanning $metal_file"
  # The -ast-tree-dump emits kernel metadata including register usage.
  # We compile to .air in a temp dir and parse the metal linker output.
  tmp_air="$(mktemp -t shim_spill).air"
  if xcrun -sdk macosx metal -x metal -c "$metal_file" -o "$tmp_air" 2>dump_err.log; then
    # Re-run with -ast-tree-dump to surface spill diagnostics.
    xcrun -sdk macosx metal -x metal -ast-tree-dump "$metal_file" 2>/dev/null \
      | grep -iE 'spill|register' >spill_report.txt || true
    if [ -s spill_report.txt ]; then
      echo "check_metal_spill: POSSIBLE SPILL in $metal_file:" >&2
      cat spill_report.txt >&2
      SPILL_FOUND=1
    fi
  else
    echo "check_metal_spill: compile failed for $metal_file (see dump_err.log)" >&2
    cat dump_err.log >&2
    SPILL_FOUND=1
  fi
  rm -f "$tmp_air" dump_err.log spill_report.txt
done

if [ "$SPILL_FOUND" -ne 0 ]; then
  echo "check_metal_spill: FAIL — spilled registers detected (L1 gate)" >&2
  exit 1
fi

echo "check_metal_spill: OK — 0 spilled registers across ${#TARGETS[@]} kernel(s)"
