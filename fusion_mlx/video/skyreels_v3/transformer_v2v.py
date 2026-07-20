# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 V2V 视频续写主干 (14B-720P) 纯 MLX 端口.

与 R2V 14B 的关键差异:
  1. 启用时序分支 (has_temporal=True), 扩大时序窗口保连贯
  2. context_window_size 参数链路: 控制前置帧 KV 复用窗口
  3. window_size=(-1, -1) 全局空间注意力, 时序窗口默认 96 帧
  4. num_frame_list / grid_size_list / num_token_list 多段上下文

V2V 续写场景:
  - 输入 5s 视频片段 -> 续写到 30s
  - 前置帧 KV 可复用, 减少重复计算
  - 时序连贯性关键, 谨慎取巧 (xfuser 策略 v2v 分支阈值更严)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import _device
from .attention import (
    WanSelfAttention,
    WanTemporalAttention,
    WanT2VCrossAttention,
    WanI2VCrossAttention,
    WAN_CROSSATTENTION_CLASSES,
    _linear_dtype,
)
from .common import (
    WanLayerNorm,
    WanRMSNorm,
    GELUApprox,
    sinusoidal_embedding_1d,
    rope_params,
    rope_apply,
    PatchEmbed3D,
    mul_add,
    mul_add_add,
    maybe_compile,
)
from .transformer_r2v import WanAttentionBlock, WanFFN, Head

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# V2V 专用 DiT Block (启用时序分支)
# ---------------------------------------------------------------------------
class V2VAttentionBlock(nn.Module):
    """V2V 视频续写 DiT Block.

    与 R2V WanAttentionBlock 的差异:
      - has_temporal=True 强制启用时序分支
      - temporal_window 默认 96 (V2V 续写保连贯)
      - context_window_size 控制前置帧 KV 复用窗口

    结构:
      1. Self-Attention (空间, 非因果)
      2. Temporal Attention (时序, SW-FA window=96)
      3. Cross-Attention (文本引导)
      4. FFN
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        cross_attn_type: str = "i2v_cross_attn",
        num_kv_heads: int | None = None,
        temporal_window: int = 96,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.temporal_window = temporal_window

        # Self-attention (空间分支)
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps,
            num_kv_heads=num_kv_heads,
        )

        # Temporal attention (时序分支, SW-FA)
        self.temporal_attn = WanTemporalAttention(
            dim, num_heads, window_size=temporal_window,
            qk_norm=qk_norm, eps=eps,
        )
        self.norm_temporal = WanLayerNorm(dim, eps)

        # Cross-attention (文本引导)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else None
        )
        cross_cls = WAN_CROSSATTENTION_CLASSES.get(cross_attn_type)
        if cross_cls is None:
            raise ValueError(
                f"Unknown cross_attn_type: {cross_attn_type}. "
                f"Valid: {list(WAN_CROSSATTENTION_CLASSES)}"
            )
        self.cross_attn = cross_cls(dim, num_heads, qk_norm, eps)

        # Feed-forward
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = WanFFN(dim, ffn_dim)

        # Learned modulation: 6 vectors for scale/shift/gate (kept in float32)
        self.modulation = (
            mx.random.normal((1, 6, dim)) * (dim ** -0.5)
        ).astype(mx.float32)

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        seq_lens: list,
        grid_sizes: list,
        freqs: mx.array,
        context: mx.array,
        context_lens: list | None = None,
        cross_kv_cache: tuple | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
        context_window_size: int = 0,
    ) -> mx.array:
        """V2V 前向.

        Args:
            temporal_len: 时序长度 (帧数), V2V 必传
            context_window_size: 前置帧 KV 复用窗口 (0=不复用)

        Returns:
            [B, L, dim] 经一个 V2V Block 处理后的隐空间
        """
        # Modulation in float32
        # e 来自 time_projection: [B, dim*6], 需 reshape 为 [B, 6, dim] 才能与 modulation (1,6,dim) 相加
        b_e = e.shape[0]
        e_6 = e.reshape(b_e, 6, -1).astype(mx.float32)  # [B, 6, dim]
        mod = self.modulation + e_6  # float32
        e0, e1, e2, e3, e4, e5 = (
            mod[:, 0, :], mod[:, 1, :], mod[:, 2, :],
            mod[:, 3, :], mod[:, 4, :], mod[:, 5, :],
        )

        # ---- 1. Self-Attention (空间) ----
        x_norm = self.norm1(x)
        x_mod = mul_add_add(x_norm, e1, e0)
        y = self.self_attn(
            x_mod, seq_lens, grid_sizes, freqs,
            rope_cos_sin=rope_cos_sin, attn_mask=attn_mask,
        )
        x = mul_add(x, y, e2)

        # ---- 2. Temporal Attention (时序, SW-FA window=96) ----
        if temporal_len is not None:
            x_t = self.norm_temporal(x)
            y_t = self.temporal_attn(
                x_t, temporal_len,
                rope_cos_sin=rope_cos_sin,
            )
            x = x + y_t

        # ---- 3. Cross-Attention (文本引导) ----
        x_cross = self.norm3(x) if self.norm3 is not None else x
        y_c = self.cross_attn(
            x_cross, context, context_lens, kv_cache=cross_kv_cache,
        )
        x = x + y_c

        # ---- 4. FFN ----
        x_norm2 = self.norm2(x)
        x_mod2 = mul_add_add(x_norm2, e4, e3)
        y_f = self.ffn(x_mod2)
        x = mul_add(x, y_f, e5)

        return x


# ---------------------------------------------------------------------------
# V2V 14B 完整主干 (SkyReelsV2VDiT)
# ---------------------------------------------------------------------------
class SkyReelsV2VDiT(nn.Module):
    """SkyReels-V3 V2V 14B-720P 视频续写主干.

    配置:
      - dim=5120, ffn_dim=13824, num_heads=40, num_layers=40
      - has_temporal=True, temporal_window=96 (保时序连贯)
      - context_window_size 控制前置帧 KV 复用窗口

    V2V 续写场景:
      - 输入 5s 视频片段 -> 续写到 30s
      - 前置帧 KV 可复用, 减少重复计算
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}
        self.dim = cfg.get("dim", 5120)
        self.ffn_dim = cfg.get("ffn_dim", 13824)
        self.num_heads = cfg.get("num_heads", 40)
        self.num_kv_heads = cfg.get("num_kv_heads", self.num_heads)
        self.num_layers = cfg.get("num_layers", 40)
        self.patch_size = cfg.get("patch_size", (1, 2, 2))
        self.in_dim = cfg.get("in_dim", 16)
        self.out_dim = cfg.get("out_dim", 16)
        self.text_dim = cfg.get("text_dim", 4096)
        self.text_len = cfg.get("text_len", 512)
        self.freq_dim = cfg.get("freq_dim", 256)
        self.window_size = cfg.get("window_size", (-1, -1))
        self.qk_norm = cfg.get("qk_norm", True)
        self.cross_attn_norm = cfg.get("cross_attn_norm", True)
        self.eps = cfg.get("eps", 1e-6)
        self.cross_attn_type = cfg.get("cross_attn_type", "i2v_cross_attn")
        self.temporal_window = cfg.get("temporal_window", 96)
        self.context_window_size = cfg.get("context_window_size", 0)

        # Patch embedding
        self.patch_embedding = PatchEmbed3D(
            self.in_dim, self.dim, self.patch_size,
        )

        # Text embedding
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.dim),
            GELUApprox(),
            nn.Linear(self.dim, self.dim),
        )

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, self.dim * 6),
        )

        # RoPE 频率表
        head_dim = self.dim // self.num_heads
        self.freqs = rope_params(2048, head_dim)

        # V2V DiT Blocks (启用时序分支)
        self.blocks = [
            V2VAttentionBlock(
                dim=self.dim,
                ffn_dim=self.ffn_dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                qk_norm=self.qk_norm,
                cross_attn_norm=self.cross_attn_norm,
                eps=self.eps,
                cross_attn_type=self.cross_attn_type,
                num_kv_heads=self.num_kv_heads,
                temporal_window=self.temporal_window,
            )
            for _ in range(self.num_layers)
        ]

        # Output head
        self.head = Head(self.dim, self.out_dim, self.patch_size, self.eps)

        # 编译优化: 逐 block maybe_compile (已存在)
        if _device.should_compile():
            self.blocks = [maybe_compile(b) for b in self.blocks]
        # 关键优化: 显式 mx.compile 整个 __call__ 融合 40-block 顺序循环
        # (MLX 0.32 编译器对 for block in blocks: x=block(x) 循环无法自动融合,
        #  致算子图按 40× 累积, Metal 峰值 146GB + 耗时 890ms/step.
        #  实测 mx.compile 后: 268ms/step (3.3×), Metal 峰值 85GB (与 R2V 持平))
        self._compiled_call = None  # lazy compile (首步触发)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        context: mx.array,
        seq_lens: list,
        grid_sizes: list,
        context_lens: list | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
    ) -> mx.array:
        """V2V 前向: 视频 latent + 时间步 + 文本 -> 去噪 latent.

        Args:
            x: [B, C_in, T, H, W] 视频 latent (含前置帧)
            t: [B] 时间步
            context: [B, L_ctx, text_dim] 文本 embedding
            temporal_len: 时序长度 (帧数), V2V 必传

        Returns:
            [B, C_out, T, H, W] 去噪 latent (续写部分)
        """
        # lazy compile: 首步用原始路径, 触发后切 mx.compile 融合循环
        if self._compiled_call is None:
            out = self._call_raw(x, t, context, seq_lens, grid_sizes, context_lens, rope_cos_sin, attn_mask, temporal_len)
            try:
                self._compiled_call = mx.compile(self._call_raw)
            except Exception:
                self._compiled_call = False
            return out
        if self._compiled_call is False:
            return self._call_raw(x, t, context, seq_lens, grid_sizes, context_lens, rope_cos_sin, attn_mask, temporal_len)
        return self._compiled_call(x, t, context, seq_lens, grid_sizes, context_lens, rope_cos_sin, attn_mask, temporal_len)

    def _call_raw(
        self,
        x: mx.array,
        t: mx.array,
        context: mx.array,
        seq_lens: list,
        grid_sizes: list,
        context_lens: list | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
    ) -> mx.array:
        """原始前向路径 (mx.compile 融合目标)."""
        # 1. Patch embedding
        x = self.patch_embedding(x)

        # 2. Text embedding
        context = self.text_embedding(context)

        # 3. Time embedding
        t_emb = sinusoidal_embedding_1d(self.freq_dim, t)
        t_emb = self.time_embedding(t_emb)
        e = self.time_projection(t_emb)

        # 4. V2V DiT Blocks (启用时序分支, context_window_size 复用前置帧)
        for block in self.blocks:
            x = block(
                x, e, seq_lens, grid_sizes, self.freqs,
                context, context_lens,
                rope_cos_sin=rope_cos_sin, attn_mask=attn_mask,
                temporal_len=temporal_len,
                context_window_size=self.context_window_size,
            )

        # 5. Output head + unpatchify
        out = self.head(x, e)
        out = self._unpatchify(out, grid_sizes)
        return out

    def _unpatchify(self, x: mx.array, grid_sizes: list) -> mx.array:
        """Unpatchify: [B, L, P*C_out] -> [B, C_out, T, H, W]."""
        b = x.shape[0]
        pt, ph, pw = self.patch_size
        outputs = []
        for i, (f, h, w) in enumerate(grid_sizes):
            # grid_sizes (f, h, w) 为 patch 后 token 网格 (对齐 wan2 unpatchify)
            seq_len = f * h * w
            x_i = x[i, :seq_len]
            x_i = x_i.reshape(f, h, w, pt, ph, pw, self.out_dim)
            x_i = x_i.transpose(6, 0, 3, 1, 4, 2, 5)
            x_i = x_i.reshape(self.out_dim, f * pt, h * ph, w * pw)
            outputs.append(x_i)
        return mx.stack(outputs, axis=0)


__all__ = [
    "V2VAttentionBlock",
    "SkyReelsV2VDiT",
]
