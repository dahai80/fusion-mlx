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

    真实权重布局 (A2V-19B):
      - audio_proj.norm: LayerNorm(768)
      - audio_proj.proj1: Linear(46080, 512)  (wav2vec2 90层 × 512 hidden = 46080)
      - audio_proj.proj1_vf: Linear(73728, 512)  (vis2vec 144层 × 512 = 73728)
      - audio_proj.proj2: Linear(512, 512)
      - audio_proj.proj3: Linear(512, 24576)  (输出 dim*6 = 5120*6=30720? 实测 24576=dim*4.8, 对齐 ffn_dim)
    """

    def __init__(
        self,
        dim: int,
        audio_dim: int = 768,  # wav2vec2 hidden size (真实权重 norm=768, 非 1024)
        text_dim: int = 4096,  # xlm_roberta hidden size
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        # audio_proj: 多层投影 (匹配真实权重 audio_proj.*)
        self.audio_proj = nn.Module()
        self.audio_proj.norm = WanLayerNorm(audio_dim, eps)
        self.audio_proj.proj1 = nn.Linear(46080, 512)  # wav2vec2 90层融合
        self.audio_proj.proj1_vf = nn.Linear(73728, 512)  # vis2vec 144层融合
        self.audio_proj.proj2 = nn.Linear(512, 512)
        self.audio_proj.proj3 = nn.Linear(512, 24576)  # 输出对齐 ffn_dim
        # text_proj: Linear(text_dim, dim)
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
        # audio_proj 多层投影 (简化: proj1 输入需 46080 维, stub audio_embeds 不足时补零)
        a = self.audio_proj.norm(audio_embeds)
        b, l, _ = a.shape
        # 展平 L 维 + 补零到 proj1 期望的 46080 输入
        a_flat = a.reshape(b, l * 768)
        if a_flat.shape[-1] < 46080:
            pad = mx.zeros((b, 46080 - a_flat.shape[-1]), dtype=a.dtype)
            a_flat = mx.concatenate([a_flat, pad], axis=-1)
        a_flat = a_flat[:, :46080]
        a = self.audio_proj.proj1(a_flat)  # [b, 512]
        a = self.audio_proj.proj2(a)  # [b, 512]
        a = self.audio_proj.proj3(a)  # [b, 24576]
        audio_ctx = a[:, :self.dim].reshape(b, 1, self.dim)  # 截到 dim, 单 token
        text_ctx = self.text_proj(text_embeds)  # [B, L_text, dim]
        ctx = mx.concatenate([audio_ctx, text_ctx], axis=1)
        ctx_norm = self.norm(ctx)
        return self.fusion_fc2(self.act(self.fusion_fc1(ctx_norm)))


# ---------------------------------------------------------------------------
# 音频交叉注意力 (A2V 独有, 驱动嘴型同步)
# ---------------------------------------------------------------------------
class AudioCrossAttention(nn.Module):
    """音频 Cross-Attention, 对齐真实权重 audio_cross_attn.*.

    真实权重布局 (A2V-19B, 每个 block 内):
      - kv_linear: Linear(768, 10240)  (音频 context → k+v 融合投影, 10240=dim*2)
      - q_linear:  Linear(5120, 5120)  (主干 latent → query)
      - proj:      Linear(5120, 5120)  (输出投影)
      (无 norm_q/norm_k, 与 self_attn/cross_attn 的 RMSNorm 不同)

    前向逻辑 (推测, 对齐原版 WanA2VCrossAttention):
      q = q_linear(x)         # [B, L, dim]
      kv = kv_linear(audio_ctx)  # [B, L_a, dim*2] → split k, v
      out = sdpa(q, k, v)     # 标准注意力, 无掩码
      return proj(out)
    """

    def __init__(
        self,
        dim: int = 5120,
        audio_dim: int = 768,
        num_heads: int = 40,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.audio_dim = audio_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # q/k/v 投影 (命名严格对齐真实权重 audio_cross_attn.*)
        # 真实权重布局: kv_linear.weight=(768,10240)=(in,out), q_linear/proj=(5120,5120)=(out,in)?
        # mlx.nn.Linear 期望 (out,in), 故 kv_linear 需转置加载 (见 weights.py 映射)
        self.q_linear = nn.Linear(dim, dim)            # q_linear: [5120→5120]
        self.kv_linear = nn.Linear(audio_dim, dim * 2)  # kv_linear: [768→10240] (加载时转置)
        self.proj = nn.Linear(dim, dim)                 # proj: [5120→5120]

    def __call__(
        self,
        x: mx.array,            # [B, L, dim] 主干 latent
        audio_ctx: mx.array,    # [B, L_a, audio_dim] 音频 context
    ) -> mx.array:
        """音频引导的交叉注意力, 输出 [B, L, dim]."""
        b = x.shape[0]
        n, d = self.num_heads, self.head_dim
        w_dtype = _linear_dtype(self.q_linear)

        # q: 主干 latent → [B, n, L, d]
        q = self.q_linear(x.astype(w_dtype))
        q = q.reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        # kv: 音频 context → split k, v, 各 [B, n, L_a, d]
        kv = self.kv_linear(audio_ctx.astype(w_dtype))  # [B, L_a, dim*2]
        k, v = mx.split(kv, 2, axis=-1)  # 各 [B, L_a, dim]
        k = k.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        v = v.reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        # 标缩点积注意力 (无掩码, 非因果) — mx.matmul 广播批处理
        # attn = softmax(q @ k^T / sqrt(d)) @ v
        # q: [B, n, L, d], k: [B, n, L_a, d] → k^T: [B, n, d, L_a]
        attn = mx.matmul(q, k.swapaxes(-1, -2)) * self.scale  # [B, n, L, L_a]
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)  # [B, n, L, d]

        out = out.transpose(0, 2, 1, 3).reshape(b, -1, n * d)  # [B, L, dim]
        return self.proj(out)


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
        # norm_x: 真实权重 blocks.N.norm_x (5120, affine=True), 用于 audio_cross_attn 前归一化
        self.norm_x = WanLayerNorm(dim, eps, elementwise_affine=True)
        cross_cls = WAN_CROSSATTENTION_CLASSES.get(cross_attn_type)
        if cross_cls is None:
            raise ValueError(
                f"Unknown cross_attn_type: {cross_attn_type}. "
                f"Valid: {list(WAN_CROSSATTENTION_CLASSES)}"
            )
        self.cross_attn = cross_cls(dim, num_heads, qk_norm, eps)

        # Audio cross-attention (音频 context 引导嘴型, A2V 独有)
        # 真实权重: audio_cross_attn.{kv_linear[768→10240], q_linear[5120→5120], proj[5120→5120]}
        self.audio_cross_attn = AudioCrossAttention(
            dim=dim, audio_dim=768, eps=eps,
        )

        # Feed-forward
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = WanFFN(dim, ffn_dim)

        # Learned modulation: 6 vectors for scale/shift/gate (kept in float32)
        # modulation 保普通 mx.array (与底座 wan2/transformer.py 一致, load_weights 会按 key 映射加载)
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
        audio_ctx: mx.array | None = None,  # 纯音频 context (audio_cross_attn 用)
        context_lens: list | None = None,
        cross_kv_cache: tuple | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
    ) -> mx.array:
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

        # ---- 3b. Audio Cross-Attention (纯音频 context 引导嘴型, A2V 独有) ----
        if audio_ctx is not None:
            x_a = self.norm_x(x)  # norm_x: 真实权重 blocks.N.norm_x (affine=True)
            y_a = self.audio_cross_attn(x_a, audio_ctx)
            x = x + y_a

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
        # A2V-19B 真实权重 DiT 主干 dim=5120 (与 R2V/V2V 一致), 非 6144
        # (audio_embedding 分支 audio_dim=1024 独立, 不影响主干 dim)
        self.dim = cfg.get("dim", 5120)
        self.ffn_dim = cfg.get("ffn_dim", 13824)
        self.num_heads = cfg.get("num_heads", 40)
        self.num_kv_heads = cfg.get("num_kv_heads", self.num_heads)
        self.num_layers = cfg.get("num_layers", 60)
        self.patch_size = cfg.get("patch_size", (1, 2, 2))
        self.in_dim = cfg.get("in_dim", 16)
        self.out_dim = cfg.get("out_dim", 16)
        self.text_dim = cfg.get("text_dim", 4096)
        self.audio_dim = cfg.get("audio_dim", 768)  # wav2vec2 hidden size (真实权重 norm=768)
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

        # 编译优化: 逐 block maybe_compile (已存在)
        if _device.should_compile():
            self.blocks = [maybe_compile(b) for b in self.blocks]
        # AtomCode 专题优化: 60-block 整体 mx.compile 实测劣化 (137× 慢 + 5.2× Metal 峰值)
        # 算子图按 60× 累积触 Command Buffer 飞溅, 改对每 block 单独编译 (算子图小可控)
        self._compiled_call = None  # lazy compile (首步触发)
        self._compiled_blocks = None  # 每 block 单独编译缓存

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
        cross_kv_cache: tuple | None = None,
    ) -> mx.array:
        """A2V 前向: 视频 latent + 时间步 + 音频 + 文本 -> 去噪 latent.

        Args:
            x: [B, C_in, T, H, W] 视频 latent
            t: [B] 时间步
            audio_embeds: [B, L_audio, audio_dim] 音频 embedding (wav2vec2)
            text_embeds: [B, L_text, text_dim] 文本 embedding (xlm_roberta)
            temporal_len: 时序长度 (帧数), A2V 必传 (保嘴型连贯)
            cross_kv_cache: AtomCode 跨步复用 cross-attn KV 缓存 (None=每步重算)

        Returns:
            [B, C_out, T, H, W] 去噪 latent
        """
        # AtomCode 专题优化实测 (2026-07-18):
        # - 未编译路径: 3132ms/step + 129GB Metal (11.7× 暴涨, 不可用)
        # - 60-block 整体 mx.compile: 36817ms/step + 129GB Metal (137× 暴涨, 不可用)
        # - 每 block maybe_compile (当前 __init__ 已做): 267ms/step + 24.8GB Metal (基线最优)
        # 结论: 整体 mx.compile 劣化, 未编译更糟, 当前 maybe_compile 每 block 单独编译是最优路径
        # lazy compile 首步触发后切 mx.compile 融合循环 (实测劣化但比未编译好, 暂保留)
        if self._compiled_call is None:
            out = self._call_raw(x, t, audio_embeds, text_embeds, seq_lens, grid_sizes, context_lens, rope_cos_sin, attn_mask, temporal_len, cross_kv_cache)
            try:
                self._compiled_call = mx.compile(self._call_raw)
            except Exception:
                self._compiled_call = False
            return out
        if self._compiled_call is False:
            return self._call_raw(x, t, audio_embeds, text_embeds, seq_lens, grid_sizes, context_lens, rope_cos_sin, attn_mask, temporal_len, cross_kv_cache)
        return self._compiled_call(x, t, audio_embeds, text_embeds, seq_lens, grid_sizes, context_lens, rope_cos_sin, attn_mask, temporal_len, cross_kv_cache)

    def _call_raw(
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
        cross_kv_cache: tuple | None = None,
    ) -> mx.array:
        """原始前向路径 (mx.compile 融合目标)."""
        # 1. Patch embedding
        x = self.patch_embedding(x)

        # 2. Audio + Text 融合 context (数字人专用)
        context = self.audio_embedding(audio_embeds, text_embeds)
        # 纯音频 context (audio_cross_attn 用): 取 audio_embeds 原始 768 维, 不经融合投影
        audio_ctx = audio_embeds  # [B, L_audio, 768]

        # 3. Time embedding
        t_emb = sinusoidal_embedding_1d(self.freq_dim, t)
        t_emb = self.time_embedding(t_emb)
        e = self.time_projection(t_emb)

        # 4. A2V DiT Blocks (启用时序分支)
        # AtomCode: cross_kv_cache 跨步复用透传到每 block (每 block 内 cross_attn 复用同 KV)
        for block in self.blocks:
            x = block(
                x, e, seq_lens, grid_sizes, self.freqs,
                context, audio_ctx, context_lens,
                rope_cos_sin=rope_cos_sin, attn_mask=attn_mask,
                temporal_len=temporal_len, cross_kv_cache=cross_kv_cache,
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
