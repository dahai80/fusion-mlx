# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 注意力层 (空间 Self-Attention / 时序 Temporal / Cross-Attn).

对接 fusion-mlx 底座:
  - fusion_mlx.custom_kernels.mfa_bridge.flash_attention
  - fusion_mlx.custom_kernels.xfuser_attention.MLXFastAttention

设计要点:
  - GQA 分组维度严格对齐原版权重 (num_heads 多于 num_kv_heads)
  - 空间 Self-Attention 非因果, 全局 window_size=(-1,-1)
  - 时序 Temporal-Attention 启用滑动窗口 SW-FA, 控制上下文降低显存
  - Cross-Attention 关闭因果掩码 (文本/Prompt 引导)
  - RoPE 用 complex64 算子, float32 保精度
"""

from __future__ import annotations

import logging
import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

try:
    from fusion_mlx.custom_kernels.mfa_bridge import (
        flash_attention as _mfa_flash_attention,
    )
    from fusion_mlx.custom_kernels.mfa_bridge import is_available as _mfa_available
except Exception:  # pragma: no cover - optional extension
    _mfa_flash_attention = None
    _mfa_available = None

try:
    from fusion_mlx.custom_kernels.xfuser_attention import (
        MLXFastAttention,
        FastAttnMethod,
        current_step as _fa_step,
        is_active as _fa_active,
    )
except Exception:  # pragma: no cover - xfuser strategy optional
    MLXFastAttention = None
    FastAttnMethod = None
    _fa_step = None
    _fa_active = None

from . import _device
from .common import WanRMSNorm, rope_apply, rope_params

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _linear_dtype(layer: nn.Module) -> mx.Dtype:
    """获取线性层权重的 dtype (处理 LoRA 包装)."""
    inner = getattr(layer, "linear", layer)
    if isinstance(inner, nn.QuantizedLinear):
        return inner.scales.dtype
    return inner.weight.dtype


def _sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    mask: mx.array | None = None,
    *,
    fast_attn: Any = None,
    step: int = 0,
    batch_size: int | None = None,
) -> mx.array:
    """统一注意力分派: fast_attn > MFA > mx.fast.sdpa."""
    if fast_attn is not None:
        return fast_attn(
            q, k, v, step, scale=scale, mask=mask, batch_size=batch_size
        )
    if _mfa_available is not None and _mfa_available():
        return _mfa_flash_attention(q, k, v, scale=scale, mask=mask)
    if mask is not None:
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask
        )
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


# ---------------------------------------------------------------------------
# 空间 Self-Attention (单帧内部, 非因果)
# ---------------------------------------------------------------------------
class WanSelfAttention(nn.Module):
    """空间 Self-Attention, 对齐 WanSelfAttention.

    特点:
      - Q/K/V 各为 nn.Linear(dim, dim)
      - qk_norm 用 WanRMSNorm (FP32 RMS 归一化)
      - RoPE 应用后转回 compute dtype
      - 非因果 (causal=False), window_size 默认 (-1,-1) 全局
      - GQA 支持: num_kv_heads < num_heads 时 K/V 重复
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = dim // num_heads
        self.kv_head_dim = dim // self.num_kv_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim // (num_heads // self.num_kv_heads))
        self.v = nn.Linear(dim, dim // (num_heads // self.num_kv_heads))
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else None
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else None

        self._fast_attn = None  # 可注入 xfuser MLXFastAttention

    def _repeat_kv(self, x: mx.array) -> mx.array:
        """GQA: K/V 从 num_kv_heads 重复到 num_heads."""
        if self.num_kv_heads == self.num_heads:
            return x
        # x: [B, num_kv_heads, L, D]
        b, nkv, l, d = x.shape
        groups = self.num_heads // self.num_kv_heads
        x = mx.broadcast_to(x[:, :, None, :, :], (b, nkv, groups, l, d))
        return x.reshape(b, nkv * groups, l, d)

    def __call__(
        self,
        x: mx.array,
        seq_lens: list,
        grid_sizes: list,
        freqs: mx.array,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        nkv = self.num_kv_heads

        w_dtype = _linear_dtype(self.q)
        x_w = x.astype(w_dtype)

        q = self.q(x_w)
        k = self.k(x_w)
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        q = q.reshape(b, s, n, d)
        k = k.reshape(b, s, nkv, d)
        v = self.v(x_w).reshape(b, s, nkv, d)

        # RoPE on Q/K (float32 precision)
        q = rope_apply(
            q.astype(mx.float32),
            grid_sizes,
            freqs,
        )
        k = rope_apply(
            k.astype(mx.float32),
            grid_sizes,
            freqs,
        )

        q = q.astype(w_dtype).transpose(0, 2, 1, 3)  # [B, N, L, D]
        k = k.astype(w_dtype).transpose(0, 2, 1, 3)  # [B, Nkv, L, D]
        v = v.transpose(0, 2, 1, 3)  # [B, Nkv, L, D]

        # GQA: expand kv to match q heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # 构建掩码 (padding mask for varlen)
        mask = attn_mask
        if mask is None and any(sl < s for sl in seq_lens):
            mask = mx.zeros((b, 1, 1, s), dtype=q.dtype)
            for i, sl in enumerate(seq_lens):
                mask[i, :, :, sl:] = -1e9

        # 滑动窗口掩码 (window_size > 0 时)
        ws = self.window_size[0]
        if ws > 0:
            i = mx.arange(s, dtype=mx.float32)[:, None]
            j = mx.arange(s, dtype=mx.float32)[None, :]
            sw_mask = mx.where(
                mx.abs(i - j) < ws, 0.0, -float("inf")
            ).astype(q.dtype)
            sw_mask = sw_mask.reshape(1, 1, s, s)
            mask = sw_mask if mask is None else mask + sw_mask

        # xfuser 步级策略
        fa = self._fast_attn
        if fa is not None and _fa_active is not None and _fa_active():
            out = _sdpa(
                q,
                k,
                v,
                self.scale,
                mask,
                fast_attn=fa,
                step=_fa_step(),
                batch_size=b,
            )
        else:
            out = _sdpa(q, k, v, self.scale, mask)

        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.o(out)


# ---------------------------------------------------------------------------
# 时序 Temporal-Attention (多帧联动, SW-FA)
# ---------------------------------------------------------------------------
class WanTemporalAttention(nn.Module):
    """时序 Temporal-Attention, 对齐 video 模型时序分支.

    特点:
      - 启用滑动窗口 SW-FA, window_size 控制上下文窗口
      - M5 Max 默认 window_size=32 帧, 旧设备 16 帧
      - 时序序列长度偏大, 强制走 mlx-mfa STEEL 内核
      - GQA 分组参数原样迁移
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = -1,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # 时序窗口: -1 全局, >0 滑动窗口
        self.window_size = window_size if window_size > 0 else _device.get_window_size_default()
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else None
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else None

        self._fast_attn = None

    def __call__(
        self,
        x: mx.array,
        temporal_len: int,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        """前向.

        Args:
            x: [B, L, dim], L = temporal_len * spatial_len
            temporal_len: 时序长度 (帧数)
        """
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        ws = self.window_size

        w_dtype = _linear_dtype(self.q)
        x_w = x.astype(w_dtype)

        q = self.q(x_w)
        k = self.k(x_w)
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        q = q.reshape(b, s, n, d).transpose(0, 2, 1, 3)  # [B, N, L, D]
        k = k.reshape(b, s, n, d).transpose(0, 2, 1, 3)
        v = self.v(x_w).reshape(b, s, n, d).transpose(0, 2, 1, 3)

        # 滑动窗口掩码 (时序方向)
        # 优化: 缓存 sw_mask 避每 block 重构造 (40 blocks × s²张量致 Metal 峰值翻倍)
        # + 缩 dtype float16 降显存 (V2V 146GB → 预期 < 100GB)
        mask = attn_mask
        if ws > 0:
            cache_key = (s, ws, q.dtype)
            cached = getattr(self, "_sw_mask_cache", None)
            if cached is None or cached[0] != cache_key:
                i = mx.arange(s, dtype=mx.float16)[:, None]
                j = mx.arange(s, dtype=mx.float16)[None, :]
                sw_mask = mx.where(
                    mx.abs(i - j) < ws, mx.array(0.0, dtype=mx.float16), mx.array(float("-inf"), dtype=mx.float16)
                ).astype(q.dtype)
                sw_mask = sw_mask.reshape(1, 1, s, s)
                self._sw_mask_cache = (cache_key, sw_mask)
            sw_mask = self._sw_mask_cache[1]
            mask = sw_mask if mask is None else mask + sw_mask

        fa = self._fast_attn
        if fa is not None and _fa_active is not None and _fa_active():
            out = _sdpa(
                q, k, v, self.scale, mask,
                fast_attn=fa, step=_fa_step(), batch_size=b,
            )
        else:
            out = _sdpa(q, k, v, self.scale, mask)

        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.o(out)


# ---------------------------------------------------------------------------
# Cross-Attention (文本/参考图引导)
# ---------------------------------------------------------------------------
class WanT2VCrossAttention(nn.Module):
    """T2V Cross-Attention: 文本 Prompt 引导画面生成, 关闭因果掩码."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else None
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else None

        self._fast_attn = None

    def prepare_kv(self, context: mx.array) -> tuple:
        """AtomCode 专题优化: 预算 cross-attn KV 缓存跨步复用 (2026-07-18).

        Args:
            context: [B, L_ctx, dim] 文本/参考 context

        Returns:
            (k, v) KV 缓存 tuple, 跨采样步复用避每步重算 KV 投影
        """
        b = context.shape[0]
        n, d = self.num_heads, self.head_dim
        w_dtype = _linear_dtype(self.k)
        ctx = context.astype(w_dtype)
        k = self.k(ctx)
        if self.norm_k is not None:
            k = self.norm_k(k)
        k = k.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        v = self.v(ctx).reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        return k, v

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        context_lens: list | None = None,
        kv_cache: tuple | None = None,
    ) -> mx.array:
        b = x.shape[0]
        n, d = self.num_heads, self.head_dim

        w_dtype = _linear_dtype(self.q)
        q = self.q(x.astype(w_dtype))
        if self.norm_q is not None:
            q = self.norm_q(q)
        q = q.reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            k, v = kv_cache
        else:
            ctx = context.astype(w_dtype)
            k = self.k(ctx)
            if self.norm_k is not None:
                k = self.norm_k(k)
            k = k.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
            v = self.v(ctx).reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        # 可选 context 掩码
        mask = None
        if context_lens is not None:
            ctx_len = k.shape[2]
            mask = mx.zeros((b, 1, 1, ctx_len), dtype=q.dtype)
            for i, cl in enumerate(context_lens):
                mask[i, :, :, cl:] = -1e9

        fa = self._fast_attn
        if fa is not None and _fa_active is not None and _fa_active():
            out = _sdpa(
                q, k, v, self.scale, mask,
                fast_attn=fa, step=_fa_step(), batch_size=b,
            )
        else:
            out = _sdpa(q, k, v, self.scale, mask)

        out = out.transpose(0, 2, 1, 3).reshape(b, -1, n * d)
        return self.o(out)


class WanI2VCrossAttention(WanT2VCrossAttention):
    """I2V Cross-Attention: 参考图引导, 额外 k_img/v_img 分支.

    原版 WanI2VCrossAttention:
      - 拆分 context_img = context[:, :257] / context = context[:, 257:]
      - k_img/v_img 独立投影
      - img_x = attention(q, k_img, v_img)
      - x = attention(q, k, v) + img_x
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(dim, num_heads, qk_norm, eps)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else None

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        context_lens: list | None = None,
        kv_cache: tuple | None = None,
    ) -> mx.array:
        b = x.shape[0]
        n, d = self.num_heads, self.head_dim

        w_dtype = _linear_dtype(self.q)
        q = self.q(x.astype(w_dtype))
        if self.norm_q is not None:
            q = self.norm_q(q)
        q = q.reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        # 拆分参考图 context (前 257 个 token) 和文本 context
        img_len = 257
        context_img = context[:, :img_len]
        context_txt = context[:, img_len:]

        # 文本分支
        ctx_txt = context_txt.astype(w_dtype)
        k = self.k(ctx_txt)
        if self.norm_k is not None:
            k = self.norm_k(k)
        k = k.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        v = self.v(ctx_txt).reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        # 参考图分支
        ctx_img = context_img.astype(w_dtype)
        k_img = self.k_img(ctx_img)
        if self.norm_k_img is not None:
            k_img = self.norm_k_img(k_img)
        k_img = k_img.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        v_img = self.v_img(ctx_img).reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        # 两路注意力
        out_txt = _sdpa(q, k, v, self.scale)
        out_img = _sdpa(q, k_img, v_img, self.scale)
        out = out_txt + out_img

        out = out.transpose(0, 2, 1, 3).reshape(b, -1, n * d)
        return self.o(out)


# ---------------------------------------------------------------------------
# Cross-Attention 注册表
# ---------------------------------------------------------------------------
WAN_CROSSATTENTION_CLASSES: dict[str, type] = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


__all__ = [
    "WanSelfAttention",
    "WanTemporalAttention",
    "WanT2VCrossAttention",
    "WanI2VCrossAttention",
    "WAN_CROSSATTENTION_CLASSES",
    "_sdpa",
    "_linear_dtype",
]
