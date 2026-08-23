# SPDX-License-Identifier: Apache-2.0
# weight_norm 重参数化重建（PyTorch weight_norm → MLX 扁平权重）。
#
# 上游权重以 weight_norm parametrization 存储：weight_g + weight_v。
# PyTorch 重建公式：weight = weight_g * (weight_v / ||weight_v||)，
# 其中 ||weight_v|| 沿除第 0 维（输出通道）外的所有轴求范数。
# 推理阶段无需运行时重参数化，加载时一次性重建为扁平权重。
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def reconstruct_weight_norm(weight_g, weight_v):
    # weight_g: (out,) 或 (out,1,1) — 方向范数。
    # weight_v: 与目标 weight 同形状 — 方向。
    # 返回：扁平 weight = g * v / ||v||。
    #
    # PyTorch weight_norm 对 dim=0（默认）规范化：
    #   ||v|| = sqrt(sum(v**2, dim=all-but-0))，即每个输出通道的权重向量范数。
    # weight_g 形状广播到 weight_v。
    g = mx.array(weight_g, dtype=mx.float32)
    v = mx.array(weight_v, dtype=mx.float32)
    # 范数沿除 axis=0 外所有轴。
    axes = tuple(range(1, v.ndim))
    if axes:
        norm = mx.sqrt(mx.sum(v * v, axis=axes, keepdims=True) + 1e-12)
    else:
        # weight_v 为 1D（极少见），整体范数。
        norm = mx.sqrt(mx.sum(v * v) + 1e-12)
    # weight_g 可能是 (out,) 或 (out,1,1)，广播到 v 形状。
    while g.ndim < v.ndim:
        g = mx.expand_dims(g, -1)
    weight = g * v / norm
    logger.debug(
        "weight_norm reconstruct: g=%s v=%s -> %s", g.shape, v.shape, weight.shape
    )
    return weight


__all__ = ["reconstruct_weight_norm"]
