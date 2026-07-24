# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_METAL_SRC = Path(__file__).parent.parent / "metal" / "moe_ffn_fused.metal"

_NATIVE_AVAILABLE = False


def is_native_available() -> bool:
    return _NATIVE_AVAILABLE


def moe_ffn_fused(
    x: mx.array,
    w_gate: mx.array,
    w_up: mx.array,
    w_down: mx.array,
    scales_a: mx.array = None,
    scales_b_gate: mx.array = None,
    scales_b_up: mx.array = None,
    scales_b_down: mx.array = None,
    expert_idx: int = 0,
    group_size: int = 32,
) -> mx.array:
    logger.info(
        f"moe_ffn_fused: shape x={x.shape} w_gate={w_gate.shape} "
        f"w_up={w_up.shape} w_down={w_down.shape} expert={expert_idx}"
    )

    if _NATIVE_AVAILABLE:
        return _moe_ffn_fused_native(
            x, w_gate, w_up, w_down,
            scales_a, scales_b_gate, scales_b_up, scales_b_down,
            expert_idx, group_size,
        )

    return _moe_ffn_fused_fallback(x, w_gate, w_up, w_down)


def _moe_ffn_fused_fallback(
    x: mx.array,
    w_gate: mx.array,
    w_up: mx.array,
    w_down: mx.array,
) -> mx.array:
    gate = x @ w_gate.T
    up = x @ w_up.T
    hidden = nn.silu(gate) * up
    output = hidden @ w_down.T
    return output


def _moe_ffn_fused_native(
    x: mx.array,
    w_gate: mx.array,
    w_up: mx.array,
    w_down: mx.array,
    scales_a: mx.array,
    scales_b_gate: mx.array,
    scales_b_up: mx.array,
    scales_b_down: mx.array,
    expert_idx: int,
    group_size: int,
) -> mx.array:
    raise NotImplementedError(
        "Native MoE FFN kernel requires C++ extension integration. "
        "See custom_kernels/metal/moe_ffn_fused.metal for the Metal implementation. "
        "Use the fallback path (separate matmuls) or build the C++ extension."
    )
