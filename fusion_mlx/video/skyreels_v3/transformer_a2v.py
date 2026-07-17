# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 A2V 数字人主干 (19B-720P) 纯 MLX 端口.

A2V 与 R2V/V2V 的关键差异:
  1. 新增音频 Embedding 分支 (wav2vec2 + xlm_roberta)
  2. dim=6144, ffn_dim=24576, num_heads=48, num_layers=60 (19B)
  3. 音频驱动口型同步, 启用时序分支保嘴型连贯
  4. cross_attn 同时接收文本 + 音频 context

A2V 数字人场景:
  - 输入: 音频 + 参考图 + 文本 Prompt
  - 输出: 数字人说话视频 (口型同步)
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
from .transformer_r2v import WanFFN, Head

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 音频 Embedding 分支
# ---------------------------------------------------------------------------
class AudioEmbedding(nn.Module):
    """音频 Embedding 分支, 对齐 wav2vec2 + xlm_roberta 融合.

    结构:
      - audio_proj: Linear(audio_dim, dim)
      - text_proj: Linear(text_dim, dim)
      - fusion: LayerNorm + Linear + GELU + Linear (音频+文本融合)
    """

    def __init__(
        self,
        dim: int,
        audio_dim: int = 1024,  # wav2vec2 hidden size
        text_dim: int = 4096,  # xlm_roberta hidden size
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.audio_proj = nn.Linear(audio_dim, dim)
        self.text_proj = nn.Linear(text_dim, dim)
        self.norm = WanLayerNorm(dim, eps)
        self.fusion_fc1 = nn.Linear(dim, dim)
        self.act = GELUApprox()
        self.fusion_fc2 = nn.Linear(dim, dim)

    def __call__(
        self,
        audio_embeds: mx.array,
        text_embeds: mx.array,
    ) -> mx.array:
        """融合音频 + 文本 embedding.

        Args:
            audio_embeds: [B, L_audio, audio_dim]
            text_embeds: [B, L_text, text_dim]

        Returns:
            [B, L_audio+L_text, dim] 融合 context
        """
        audio_ctx = self.audio_proj(audio_embeds)  # [B, L_audio, dim]
        text_ctx = self.text_proj(text_embeds)  # [B, L_text, dim]
        ctx = mx.concatenate([audio_ctx, text_ctx], axis=1)  # [B, L_audio+L_text, dim]
        # fusion
        ctx_norm = self.norm(ctx)
        return self.fusion_fc2(self.act(self.fusion_fc1(ctx_norm)))


# ---------------------------------------------------------------------------
# A2V 专用 DiT Block (音频驱动)
# ---------------------------------------------------------------------------
class A2VAttentionBlock(nn.Module):
    """A2V 数字人 DiT Block, 启用时序分支保嘴型连贯.

    结构:
      1. Self-Attention (空间, 非因果)
      2. Temporal Attention (时序, SW-FA, window=32)
      3. Cross-Attention (音频+文本融合 context 引导)
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
        temporal_window: int = 32,
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

        # Temporal attention (时序分支, SW-FA window=32)
        self.temporal_attn = WanTemporalAttention(
            dim, num_heads, window_size=temporal_window,
            qk_norm=qk_norm, eps=eps,
        )
        self.norm_temporal = WanLayerNorm(dim, eps)

        # Cross-attention (音频+文本融合 context 引导)
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
        context: mx.array,  # 音频+文本融合 context
        context_lens: list | None = None,
        cross_kv_cache: tuple | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
    ) -> mx.array:
        """A2V 前向.

        Args:
            temporal_len: 时序长度 (帧数), A2V 必传 (保嘴型连贯)
            context: 音频+文本融合 context (来自 AudioEmbedding)

        Returns:
            [B, L, dim] 经一个 A2V Block 处理后的隐空间
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

        # ---- 2. Temporal Attention (时序, SW-FA window=32) ----
        if temporal_len is not None:
            x_t = self.norm_temporal(x)
            y_t = self.temporal_attn(
                x_t, temporal_len,
                rope_cos_sin=rope_cos_sin,
            )
            x = x + y_t

        # ---- 3. Cross-Attention (音频+文本融合 context 引导) ----
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
# A2V 19B 完整主干 (SkyReelsA2VDiT)
# ---------------------------------------------------------------------------
class SkyReelsA2VDiT(nn.Module):
    """SkyReels-V3 A2V 19B-720P 数字人主干.

    配置:
      - dim=6144, ffn_dim=24576, num_heads=48, num_layers=60 (19B)
      - has_temporal=True, temporal_window=32 (保嘴型连贯)
      - 音频分支: AudioEmbedding (wav2vec2 + xlm_roberta 融合)

    A2V 数字人场景:
      - 输入: 音频 + 参考图 + 文本 Prompt
      - 输出: 数字人说话视频 (口型同步)
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}
        self.dim = cfg.get("dim", 6144)
        self.ffn_dim = cfg.get("ffn_dim", 24576)
        self.num_heads = cfg.get("num_heads", 48)
        self.num_kv_heads = cfg.get("num_kv_heads", self.num_heads)
        self.num_layers = cfg.get("num_layers", 60)
        self.patch_size = cfg.get("patch_size", (1, 2, 2))
        self.in_dim = cfg.get("in_dim", 16)
        self.out_dim = cfg.get("out_dim", 16)
        self.text_dim = cfg.get("text_dim", 4096)
        self.audio_dim = cfg.get("audio_dim", 1024)
        self.text_len = cfg.get("text_len", 512)
        self.freq_dim = cfg.get("freq_dim", 256)
        self.window_size = cfg.get("window_size", (-1, -1))
        self.qk_norm = cfg.get("qk_norm", True)
        self.cross_attn_norm = cfg.get("cross_attn_norm", True)
        self.eps = cfg.get("eps", 1e-6)
        self.cross_attn_type = cfg.get("cross_attn_type", "i2v_cross_attn")
        self.temporal_window = cfg.get("temporal_window", 32)

        # Patch embedding
        self.patch_embedding = PatchEmbed3D(
            self.in_dim, self.dim, self.patch_size,
        )

        # Text embedding (基础文本, 与 AudioEmbedding 融合后作为 context)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.dim),
            GELUApprox(),
            nn.Linear(self.dim, self.dim),
        )

        # Audio embedding 分支 (数字人专用)
        self.audio_embedding = AudioEmbedding(
            dim=self.dim,
            audio_dim=self.audio_dim,
            text_dim=self.text_dim,
            eps=self.eps,
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

        # A2V DiT Blocks (启用时序分支)
        self.blocks = [
            A2VAttentionBlock(
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

        # 编译优化
        if _device.should_compile():
            self.blocks = [maybe_compile(b) for b in self.blocks]

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        audio_embeds: mx.array,
        text_embeds: mx.array,
        seq_lens: list,
        grid_sizes: list,
        context_lens: list | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
    ) -> mx.array:
        """A2V 前向: 视频 latent + 时间步 + 音频 + 文本 -> 去噪 latent.

        Args:
            x: [B, C_in, T, H, W] 视频 latent
            t: [B] 时间步
            audio_embeds: [B, L_audio, audio_dim] 音频 embedding (wav2vec2)
            text_embeds: [B, L_text, text_dim] 文本 embedding (xlm_roberta)
            temporal_len: 时序长度 (帧数), A2V 必传

        Returns:
            [B, C_out, T, H, W] 去噪 latent
        """
        # 1. Patch embedding
        x = self.patch_embedding(x)

        # 2. Audio + Text 融合 context (数字人专用)
        context = self.audio_embedding(audio_embeds, text_embeds)

        # 3. Time embedding
        t_emb = sinusoidal_embedding_1d(self.freq_dim, t)
        t_emb = self.time_embedding(t_emb)
        e = self.time_projection(t_emb)

        # 4. A2V DiT Blocks (启用时序分支)
        for block in self.blocks:
            x = block(
                x, e, seq_lens, grid_sizes, self.freqs,
                context, context_lens,
                rope_cos_sin=rope_cos_sin, attn_mask=attn_mask,
                temporal_len=temporal_len,
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
            h_p, w_p = h // ph, w // pw
            seq_len = f * h_p * w_p
            x_i = x[i, :seq_len]
            x_i = x_i.reshape(f, h_p, w_p, pt, ph, pw, self.out_dim)
            x_i = x_i.transpose(6, 0, 3, 1, 4, 2, 5)
            x_i = x_i.reshape(self.out_dim, f * pt, h_p * ph, w_p * pw)
            outputs.append(x_i)
        return mx.stack(outputs, axis=0)


__all__ = [
    "AudioEmbedding",
    "A2VAttentionBlock",
    "SkyReelsA2VDiT",
]
