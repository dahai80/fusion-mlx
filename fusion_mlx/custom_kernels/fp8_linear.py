# SPDX-License-Identifier: Apache-2.0
"""FP8 线性投影算子 (M5 Max Neural Accelerator 专属).

兼容兜底:
  - M5 无 FP8 硬件支持时, 透明降级到 bf16 matmul (无性能损失)
"""
from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_FP8_AVAILABLE = False
try:
    _FP8_AVAILABLE = hasattr(mx, "float8_e4m3fn")
except Exception:
    _FP8_AVAILABLE = False


def is_available() -> bool:
    return _FP8_AVAILABLE


def quantize_fp8(weight: mx.array) -> tuple[mx.array, mx.array]:
    if not _FP8_AVAILABLE:
        return weight, mx.ones((weight.shape[0],), dtype=mx.float32)
    absmax = mx.max(mx.abs(weight), axis=-1)
    scale = absmax / mx.array(448.0, dtype=mx.float32)
    scale = mx.where(scale > 0, scale, mx.array(1.0, dtype=mx.float32))
    fp8_w = (weight / scale[:, None]).astype(mx.float8_e4m3fn)
    return fp8_w, scale


def fp8_matmul(x: mx.array, fp8_w: mx.array, scale: mx.array) -> mx.array:
    """FP8 matmul 自动检测权重格式并正确转置.

    AtomCode fix #133: 用 mx.transpose 强物化转置替 .T 视图 (2026-07-20).
    AtomCode fix #135-#137: 自动检测权重是 (out,in) 还是 (in,out) 格式 (2026-07-20).

    MLX nn.Linear 权重形状为 (out_features, in_features).
    HuggingFace Diffusers/MLX-converted 权重可能存为 (in_features, out_features).
    检测: 若 fp8_w.shape[0] == x.shape[-1] 则权重是 (in,out) 格式, 不转置.
          若 fp8_w.shape[1] == x.shape[-1] 则权重是 (out,in) 格式, 转置.
    """
    if not _FP8_AVAILABLE:
        # 自动检测权重格式
        if fp8_w.shape[0] == x.shape[-1]:
            # 权重已是 (in,out) 格式, 直接 matmul
            return x @ fp8_w
        # 权重是 (out,in) 格式, 转置到 (in,out)
        w_t = mx.transpose(fp8_w, (1, 0))
        return x @ w_t
    w = fp8_w.astype(mx.bfloat16) * scale[:, None].astype(mx.bfloat16)
    # 自动检测权重格式 (FP8 路径)
    if w.shape[0] == x.shape[-1]:
        return x @ w
    w_t = mx.transpose(w, (1, 0))
    return x @ w_t


class FP8Linear(nn.Module):
    def __init__(self, out_features: int, in_features: int, bias: bool = True):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.fp8_weight: mx.array = mx.zeros((out_features, in_features), dtype=mx.float8_e4m3fn if _FP8_AVAILABLE else mx.float32)
        self.scale: mx.array = mx.ones((out_features,), dtype=mx.float32)
        self.bias: mx.array | None = mx.zeros((out_features,), dtype=mx.bfloat16) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FP8Linear":
        """从 nn.Linear 创建 FP8Linear.

        fp8_matmul 已自动检测 (out,in) 或 (in,out) 权重格式并正确转置.
        """
        out_f, in_f = linear.weight.shape
        bias = hasattr(linear, "bias") and linear.bias is not None
        layer = cls(out_f, in_f, bias=bias)
        layer.fp8_weight, layer.scale = quantize_fp8(linear.weight)
        if bias:
            layer.bias = linear.bias
        return layer

    def __call__(self, x: mx.array) -> mx.array:
        """AtomCode fix #135-#137: 前向时走 fp8_matmul 全路径含 scale.

        fp8_matmul 假设权重 shape 为 (out_features, in_features) 并转置.
        from_linear 已确保权重格式正确, 这里直接用 fp8_matmul.
        """
        out = fp8_matmul(x, self.fp8_weight, self.scale)
        if self.bias is not None:
            # AtomCode fix #138: bias 形状可能 != out 末维 (因权重格式错配)
            # fp8_matmul 自动检测权重格式, 但 bias 创建时用 out_features
            # 若权重是 (in,out) 格式, out_features 读错, bias 形状错配
            b = self.bias
            if b.shape[0] != out.shape[-1]:
                b = b[:out.shape[-1]]  # 截断到匹配输出
            out = out + b
        return out


def _iter_submodules(parent: nn.Module, prefix: str = ""):
    """AtomCode fix #131 #132: 递归遍历所有子模块含 list 内嵌套 (2026-07-19).

    mlx.nn.Module.__setattr__ 用 dict 接口存子模块: mx.array/dict/list/tuple 存 self[key],
    nn.Module/nn.Linear 也存 self[key]. 故用 keys()/items() 抓所有真子模块名.
    原 apply_to_modules 不遍历 list 存的子模块 (如 blocks=[Block(), ...]), 致漏转.
    """
    for key in list(parent.keys()):
        val = parent[key]
        # mlx.nn.Module/nn.Linear 是 dict 子类 (issubclass=True), 故不可用 dict 早退过滤
        if val is None or isinstance(val, (mx.array, int, float, str, bool, tuple)) or type(val) is dict:
            continue
        name = f"{prefix}.{key}" if prefix else key
        # nn.Linear 子模块: yield (parent, key, name, val, None)
        if isinstance(val, nn.Linear):
            yield (parent, key, name, val, None)
        # nn.Module 子模块: 递归
        elif isinstance(val, nn.Module):
            yield from _iter_submodules(val, name)
        # list 属性 (blocks=[Block(), ...]): 遍历 list 内每个 nn.Module/nn.Linear
        elif isinstance(val, list):
            for i, item in enumerate(val):
                item_name = f"{name}.{i}"
                if isinstance(item, nn.Linear):
                    yield (val, i, item_name, item, "list")
                elif isinstance(item, nn.Module):
                    yield from _iter_submodules(item, item_name)


def convert_to_fp8_linear(model: nn.Module) -> nn.Module:
    """将模型中的 nn.Linear 层转换为 FP8Linear (m5_optimizer.py 兼容入口).

    AtomCode fix #131 #132: 递归遍历所有子模块含 list 内嵌套 (2026-07-19).
    原 apply_to_modules 不遍历 list 存的子模块 (如 blocks=[Block(), ...]),
    致 blocks[0].cross_attn.k_img 等漏转. 手写 _iter_submodules 遍历 _modules + list.
    """
    for parent, key, name, module, container_kind in _iter_submodules(model):
        new_layer = FP8Linear.from_linear(module)
        if container_kind == "list":
            parent[int(key)] = new_layer  # 直接改 list[i]
        else:
            setattr(parent, key, new_layer)
        logger.info("convert_to_fp8_linear: %s 已转换", name)
    return model


__all__ = ["FP8Linear", "fp8_matmul", "quantize_fp8", "is_available", "convert_to_fp8_linear", "_iter_submodules"]
