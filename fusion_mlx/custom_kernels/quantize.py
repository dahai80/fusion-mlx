# SPDX-License-Identifier: Apache-2.0
"""多级量化方案统一入口 (NF4/FP8/bf16)."""

from __future__ import annotations

import logging
from typing import Any

import mlx.nn as nn

logger = logging.getLogger(__name__)


def quantize_linear(layer: nn.Linear, bits: int = 4, group_size: int = 32) -> nn.Module:
    if bits == 16:
        return layer
    if bits == 8:
        from .fp8_linear import FP8Linear
        from .fp8_linear import is_available as _fp8_ok

        if _fp8_ok():
            return FP8Linear.from_linear(layer)
        logger.warning("FP8 硬件不可用, 降级 bf16 (weight_bits=8 → 16)")
        return layer
    if bits == 4:
        try:
            from mlx_lm.tuner.quant import quantize

            q_layer = quantize(layer, bits=4, group_size=group_size)
            logger.info("NF4 量化成功: %s → group=%d", layer, group_size)
            return q_layer
        except Exception as e:
            logger.warning("NF4 量化失败 (%s), 降级 bf16", e)
            return layer
    logger.warning("未知 bits=%d, 降级 bf16", bits)
    return layer


def get_quantize_config(weight_bits: int = 4, kv_bits: int = 4) -> dict[str, Any]:
    from .fp8_linear import is_available as _fp8_ok

    return {
        "weight_bits": weight_bits,
        "kv_bits": kv_bits,
        "group_size": 32,
        "fp8_available": _fp8_ok(),
    }


def quantize_model(model: nn.Module, bits: int = 4, group_size: int = 32) -> nn.Module:
    """将模型量化为指定位数 (m5_optimizer.py 兼容入口).

    AtomCode fix #131 #132: 递归遍历所有子模块含 list 内嵌套 (2026-07-19).
    用 fp8_linear._iter_submodules 遍历含 list 属性的嵌套子模块.
    """
    from .fp8_linear import _iter_submodules

    for parent, key, name, module, container_kind in _iter_submodules(model):
        q = quantize_linear(module, bits=bits, group_size=group_size)
        if q is not module:
            if container_kind == "list":
                parent[int(key)] = q
            else:
                setattr(parent, key, q)
            logger.info("quantize_model: %s %d-bit 量化完成", name, bits)
    return model


__all__ = ["quantize_linear", "get_quantize_config", "quantize_model"]
