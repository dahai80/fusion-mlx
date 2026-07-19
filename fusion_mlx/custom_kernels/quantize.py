# SPDX-License-Identifier: Apache-2.0
"""多级量化方案统一入口 (NF4/FP8/bf16).

对接 fusion_mlx.custom_kernels.fp8_linear (FP8 算子) +
mlx-lm 原生 NF4 量化 (mlx-community/NF4 权重布局).

量化层级:
  - weight_bits=4: NF4 权重 (mlx-lm 原生 QuantizedLinear, 19B 常驻 14GB)
  - weight_bits=8: FP8 权重 (fp8_linear.FP8Linear, M5 Neural Accelerator)
  - weight_bits=16: bf16 (无量化, 精度最高)

KV Cache 量化:
  - kv_bits=4: TurboQuant 4-bit (mlx-lm 原生, 长上下文显存省 8×)
  - kv_bits=8: FP8 KV (兼容兜底)
"""
from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def quantize_linear(layer: nn.Linear, bits: int = 4, group_size: int = 32) -> nn.Module:
    """将 nn.Linear 量化为指定位数.

    Args:
        layer: nn.Linear 待量化层
        bits: 4=NF4, 8=FP8, 16=bf16 (不量化)
        group_size: NF4 分组大小 (默认 32, mlx-community 约定)

    Returns:
        量化后模块 (QuantizedLinear 或 FP8Linear 或原 Linear)
    """
    if bits == 16:
        return layer
    if bits == 8:
        from .fp8_linear import FP8Linear, is_available as _fp8_ok
        if _fp8_ok():
            return FP8Linear.from_linear(layer)
        logger.warning("FP8 硬件不可用, 降级 bf16 (weight_bits=8 → 16)")
        return layer
    if bits == 4:
        # NF4 走 mlx-lm 原生 QuantizedLinear (mlx-community 权重布局)
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
    """获取量化配置字典 (透传到 SkyReels DiT 加载器).

    Args:
        weight_bits: 权重位数 (4/8/16)
        kv_bits: KV Cache 位数 (4/8/16)

    Returns:
        dict: {weight_bits, kv_bits, group_size, fp8_available}
    """
    from .fp8_linear import is_available as _fp8_ok
    return {
        "weight_bits": weight_bits,
        "kv_bits": kv_bits,
        "group_size": 32,  # mlx-community NF4 约定
        "fp8_available": _fp8_ok(),
    }


__all__ = ["quantize_linear", "get_quantize_config", "quantize_model"]


def quantize_model(model: nn.Module, bits: int = 4, group_size: int = 32) -> nn.Module:
    """将模型量化为指定位数 (m5_optimizer.py 兼容入口).

    Args:
        model: MLX 模型
        bits: 4=NF4, 8=FP8, 16=bf16
        group_size: NF4 分组大小 (默认 32)

    Returns:
        量化后的模型
    """
    for name, module in list(model.__dict__.items()):
        if isinstance(module, nn.Linear):
            q = quantize_linear(module, bits=bits, group_size=group_size)
            if q is not module:
                setattr(model, name, q)
                logger.info("quantize_model: %s %d-bit 量化完成", name, bits)
    return model
