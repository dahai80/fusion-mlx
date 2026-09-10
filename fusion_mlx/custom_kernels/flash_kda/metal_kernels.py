# SPDX-License-Identifier: Apache-2.0
"""FlashKDA Metal kernel bridge.

Provides the Metal compute shader backend for KDA recurrence on Apple Silicon.
Falls back to reference if Metal kernel compilation fails or is unavailable.

Phase 2: Metal compute kernels. Strategy mirrors CUDA K1/K2:
- K1: Token-parallel gate computation + inverse (CHUNK=16 tiles)
- K2: Head-parallel recurrence (SIMD matrix multiply for outer product + query)

Metal-specific:
- simd_matrix_multiply for q^T * h (16x16 -> 16x16)
- simd_shuffle for intra-SIMD-group gate reduction
- bf16 on-chip state via simdgroup_matrix
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

_METAL_AVAILABLE = False
_KERNEL_LOADED = False
_FALLBACK_WARNED = False

_METAL_SRC_DIR = Path(__file__).parent / "metal"

_PLACEHOLDER_MARKERS = ("Placeholder:", "Will use simdgroup_matrix", "TODO", "FIXME")


def metal_available() -> bool:
    """Check if Metal FlashKDA kernel is available and operational."""
    return _METAL_AVAILABLE


def _has_placeholder_body(source: str) -> bool:
    """Detect kernel functions whose body is only comments (no real logic)."""
    import re

    kernel_re = re.compile(r"kernel\s+void\s+(\w+)\s*\(", re.MULTILINE)
    for m in kernel_re.finditer(source):
        pos = m.end()
        paren = 1
        while pos < len(source) and paren > 0:
            if source[pos] == "(":
                paren += 1
            elif source[pos] == ")":
                paren -= 1
            pos += 1
        while pos < len(source) and source[pos] in " \t\n\r":
            pos += 1
        if pos >= len(source) or source[pos] != "{":
            continue
        body_start = pos + 1
        brace = 1
        bpos = body_start
        while bpos < len(source) and brace > 0:
            if source[bpos] == "{":
                brace += 1
            elif source[bpos] == "}":
                brace -= 1
            bpos += 1
        body = source[body_start : bpos - 1]
        stripped = re.sub(r"//[^\n]*", "", body)
        stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
        stripped = stripped.strip()
        if not stripped:
            return True
        for marker in _PLACEHOLDER_MARKERS:
            if marker in body and len(stripped) < 80:
                return True
    return False


def _try_load_metal_kernels() -> bool:
    """Attempt to compile and load Metal kernels. Returns True on success."""
    global _METAL_AVAILABLE, _KERNEL_LOADED

    metal_file = _METAL_SRC_DIR / "flash_kda_kernels.metal"
    if not metal_file.exists():
        logger.info(
            "FlashKDA Metal kernel source not found at %s, using reference", metal_file
        )
        return False

    try:
        source = metal_file.read_text()
        if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
            logger.info("FlashKDA: mx.fast.metal_kernel not available, using reference")
            return False
        if _has_placeholder_body(source):
            logger.warning(
                "FlashKDA Metal kernel source has placeholder/empty kernel bodies, "
                "using reference implementation until kernels are fully implemented"
            )
            _KERNEL_LOADED = True
            _METAL_AVAILABLE = False
            return False
        _KERNEL_LOADED = True
        _METAL_AVAILABLE = True
        logger.info(
            "FlashKDA Metal kernel source found, will be JIT-compiled on first use"
        )
        return True
    except Exception as exc:
        logger.warning("FlashKDA Metal kernel load failed: %s, using reference", exc)
        _METAL_AVAILABLE = False
        return False


_METAL_AVAILABLE = _try_load_metal_kernels()


def fwd(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    scale: float = 1.0,
    A_log: mx.array | None = None,
    dt_bias: mx.array | None = None,
    lower_bound: float = -5.0,
    initial_state: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """FlashKDA forward using Metal kernels.

    Currently delegates to chunked reference until Metal shaders are
    compiled. The Metal kernel will be invoked here once loaded.
    """
    global _FALLBACK_WARNED
    from .reference import fwd as fwd_ref

    if not _FALLBACK_WARNED:
        logger.warning(
            "FlashKDA Metal kernel not operational, using reference implementation. "
            "This warning will not repeat."
        )
        _FALLBACK_WARNED = True
    else:
        logger.debug("FlashKDA delegating to reference")
    return fwd_ref(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound, initial_state)
