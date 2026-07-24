# SPDX-License-Identifier: Apache-2.0
"""W4A8 fused matmul Python dispatch layer.

NOTE: mx.fast.metal_kernel cannot express complex tiled GEMM kernels with
simdgroup_matrix MMA — it generates a simple element-wise wrapper. The Metal
source (custom_kernels/metal/w4a8_fused_matmul.metal) compiles correctly via
xcrun but requires C++ extension integration for native dispatch.

Current path: fallback via mx.quantized_matmul (W4 only, no A8 benefit).
"""

import logging
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

_METAL_SRC = Path(__file__).parent.parent / "metal" / "w4a8_fused_matmul.metal"

_NATIVE_AVAILABLE = False


def is_native_available() -> bool:
    return _NATIVE_AVAILABLE


def w4a8_fused_matmul(
    x: mx.array,
    w_quantized: mx.array,
    w_scales: mx.array,
    w_biases: mx.array,
    *,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    if _NATIVE_AVAILABLE:
        raise NotImplementedError("Native W4A8 kernel requires C++ extension build")

    logger.debug("w4a8_fused_matmul fallback: using mx.quantized_matmul (fp16 activations)")
    return mx.quantized_matmul(
        x, w_quantized, w_scales, w_biases,
        group_size=group_size, bits=bits,
    )
