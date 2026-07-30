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
import os
from pathlib import Path
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_METAL_AVAILABLE = False
_KERNEL_LOADED = False

_METAL_SRC_DIR = Path(__file__).parent / "metal"


def metal_available() -> bool:
    """Check if Metal FlashKDA kernel is available."""
    return _METAL_AVAILABLE


def _try_load_metal_kernels() -> bool:
    """Attempt to compile and load Metal kernels. Returns True on success."""
    global _METAL_AVAILABLE, _KERNEL_LOADED

    metal_file = _METAL_SRC_DIR / "flash_kda_kernels.metal"
    if not metal_file.exists():
        logger.info("FlashKDA Metal kernel source not found at %s, using reference", metal_file)
        return False

    try:
        source = metal_file.read_text()
        if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
            logger.info("FlashKDA: mx.fast.metal_kernel not available, using reference")
            return False
        _KERNEL_LOADED = True
        _METAL_AVAILABLE = True
        logger.info("FlashKDA Metal kernel source found, will be JIT-compiled on first use")
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
    A_log: Optional[mx.array] = None,
    dt_bias: Optional[mx.array] = None,
    lower_bound: float = -5.0,
    initial_state: Optional[mx.array] = None,
) -> tuple[mx.array, mx.array]:
    """FlashKDA forward using Metal kernels.

    Currently delegates to chunked reference until Metal shaders are
    compiled. The Metal kernel will be invoked here once loaded.
    """
    from .reference import fwd as fwd_ref

    logger.debug("FlashKDA Metal kernel not yet operational, delegating to reference")
    return fwd_ref(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound, initial_state)
