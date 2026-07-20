# SPDX-License-Identifier: Apache-2.0
"""FP8 线性投影算子 (M5 Max Neural Accelerator 专属).

兼容兜底:
  - M5 无 FP8 硬件支持时, 透明降级到 bf16 matmul (无性能损失)
"""

from __future__ import annotations

import logging

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
    """FP8 matmul: 权重恒为 (out_features, in_features), 转置后与 x 相乘 (等价 nn.Linear 的 x @ W.T).

    AtomCode fix #139 (2026-07-20): 移除 #137 引入的 (out,in)/(in,out) 自动检测.
    自动检测在 x.shape[-1] 恰好等于 fp8_w.shape[0] (out) 时误判为 (in,out) 跳过转置,
    对方阵权重 (如 SkyReels text_embedding.1 5120x5120) 产出 x@W 而非 x@W.T, 数学错误,
    连锁触发 #138 bias 截断 / #139 维度错配. 权重恒 (out,in), 恒转置, 与 nn.Linear 一致.
    AtomCode fix #133: 用 mx.transpose 强物化转置替 .T 视图.
    """
    if not _FP8_AVAILABLE:
        w_t = mx.transpose(fp8_w, (1, 0))
        return x @ w_t
    w = fp8_w.astype(mx.bfloat16) * scale[:, None].astype(mx.bfloat16)
    w_t = mx.transpose(w, (1, 0))
    return x @ w_t


class FP8Linear(nn.Module):
    def __init__(self, out_features: int, in_features: int, bias: bool = True):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.fp8_weight: mx.array = mx.zeros(
            (out_features, in_features),
            dtype=mx.float8_e4m3fn if _FP8_AVAILABLE else mx.float32,
        )
        self.scale: mx.array = mx.ones((out_features,), dtype=mx.float32)
        self.bias: mx.array | None = (
            mx.zeros((out_features,), dtype=mx.bfloat16) if bias else None
        )

    @property
    def weight(self) -> mx.array:
        # AtomCode fix #142: 对齐 nn.QuantizedLinear.weight 约定, 返回量化存储 fp8_weight.
        # 仅供值访问/内省; 计算 dtype 用 compute_dtype (fp8_matmul 实际 dtype, 非 float8).
        return self.fp8_weight

    @property
    def compute_dtype(self) -> mx.Dtype:
        # AtomCode fix #142: fp8_matmul 实际计算 dtype. _FP8_AVAILABLE 时 bf16, 否则 f32 兜底.
        # _linear_dtype 必须返回此值, 不能用 fp8_weight.dtype (float8) 否则 x.astype(float8) 错配.
        return mx.bfloat16 if _FP8_AVAILABLE else mx.float32

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> FP8Linear:
        """从 nn.Linear 创建 FP8Linear (权重恒 (out,in), __call__ 转置)."""
        out_f, in_f = linear.weight.shape
        bias = hasattr(linear, "bias") and linear.bias is not None
        layer = cls(out_f, in_f, bias=bias)
        layer.fp8_weight, layer.scale = quantize_fp8(linear.weight)
        if bias:
            layer.bias = linear.bias
        return layer

    def __call__(self, x: mx.array) -> mx.array:
        """前向: fp8_matmul (含 scale) + bias, 等价 nn.Linear 的 x @ W.T + b.

        AtomCode fix #139 (2026-07-20): 移除 #138 的 bias 截断.
        bias 恒为 (out_features,), fp8_matmul 恒输出 (..., out_features), 形状必然匹配;
        截断只在 fp8_matmul 产出错误维度时才 "需要", 本就是掩盖 #137 自动检测 bug 的胶布.
        """
        out = fp8_matmul(x, self.fp8_weight, self.scale)
        if self.bias is not None:
            out = out + self.bias
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
        if (
            val is None
            or isinstance(val, (mx.array, int, float, str, bool, tuple))
            or type(val) is dict
        ):
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


__all__ = [
    "FP8Linear",
    "fp8_matmul",
    "quantize_fp8",
    "is_available",
    "convert_to_fp8_linear",
    "_iter_submodules",
]
