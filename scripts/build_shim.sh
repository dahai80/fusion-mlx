#!/usr/bin/env bash
# scripts/build_shim.sh
# Build the fusion-mlx C++ Shim extension (nanobind + MLX + CMake).
#
# This does NOT go through setup.py (the glm_moe_dsa build path referenced
# a setup.py that was never committed). It invokes CMake directly and
# installs the _ext.*.so + metallib inplace next to fusion_mlx/shim/.
#
# The build is OPTIONAL: when _ext is absent, fusion_mlx/shim/fast.py
# degrades to the Python fallback. CI on hosts without MLX Metal (ubuntu)
# skips this script and runs the degrade path.
#
# Usage:
#   scripts/build_shim.sh            # configure + build
#   scripts/build_shim.sh --debug    # MLX_METAL_DEBUG=1
#   scripts/build_shim.sh --clean    # rm -rf build/shim first
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/fusion_mlx/shim/csrc"
BUILD_DIR="$REPO_ROOT/build/shim"
OUT_DIR="$REPO_ROOT/fusion_mlx/shim"

DEBUG=0
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --debug) DEBUG=1 ;;
    --clean) CLEAN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

PY_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "build_shim: python not found (PYTHON_BIN=$PY_BIN)" >&2
  exit 1
fi

# Verify nanobind + mlx are importable in the building interpreter — CMake
# shells out to `python -m nanobind --cmake_dir` and `python -m mlx --cmake-dir`.
if ! "$PY_BIN" -c "import nanobind, mlx" >/dev/null 2>&1; then
  echo "build_shim: nanobind/mlx not importable in $PY_BIN; install them in the build venv" >&2
  echo "  pip install nanobind mlx" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "build_shim: cmake not found" >&2
  exit 1
fi

if [ "$CLEAN" = "1" ]; then
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

DEPLOY_TARGET="$(sw_vers -productVersion 2>/dev/null | cut -d. -f1).0"

CMAKE_ARGS=(
  -G Ninja
  -S "$SRC_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$OUT_DIR"
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOY_TARGET"
)

if [ "$DEBUG" = "1" ]; then
  CMAKE_ARGS+=(-DMLX_METAL_DEBUG=ON)
  CMAKE_ARGS+=(-DCMAKE_BUILD_TYPE=Debug)
else
  CMAKE_ARGS+=(-DCMAKE_BUILD_TYPE=Release)
fi

echo "build_shim: cmake configure"
cmake "${CMAKE_ARGS[@]}"

echo "build_shim: cmake build"
cmake --build "$BUILD_DIR" --parallel

# Verify the inplace artifacts landed next to the package.
if ! ls "$OUT_DIR"/_ext*.so >/dev/null 2>&1; then
  echo "build_shim: WARN — _ext.*.so not found in $OUT_DIR (CMAKE_LIBRARY_OUTPUT_DIRECTORY mismatch?)" >&2
else
  echo "build_shim: OK — _ext.so inplace at $OUT_DIR"
fi

# metallib is optional in PR-A (no Metal kernels yet). Report if present.
if ls "$OUT_DIR"/fusion_mlx_shim_kernels.metallib >/dev/null 2>&1; then
  echo "build_shim: metallib present"
fi
