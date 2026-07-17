# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 DiT 主干 WanAttentionBlock 纯 MLX 端口 (R2V 14B).

基于 fusion-mlx wan2/transformer.py 蓝本, 适配 SkyReels-V3 R2V 14B-720P:
  - dim=5120, ffn_dim=13824, num_heads=40, num_layers=40
  - cross_attn_type="i2v_cross_attn" (参考图引导)
  - modulation(1, 6, dim) float32 保 AdaLN-Zero 精度
  - WanRMSNorm qk_norm, WanLayerNorm 残差流

原版权重映射严格对齐, 参数不可修改.
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FFN (前馈网络)
# ---------------------------------------------------------------------------
class WanFFN(nn.Module):
    """FFN: Linear -> GELU(tanh) -> Linear.

    开启 MLX 算子融合, 消除中间临时张量, 减少统一内存读写开销.
    """

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.act = GELUApprox()
        self.fc2 = nn.Linear(ffn_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x_w = x.astype(_linear_dtype(self.fc1))
        return self.fc2(self.act(self.fc1(x_w)))


# ---------------------------------------------------------------------------
# Head (输出投影)
# ---------------------------------------------------------------------------
class Head(nn.Module):
    """输出投影 Head, 对齐原版 Head 类.

    modulation(1, 2, dim) float32, head=Linear(dim, prod(patch_size)*out_dim).
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.patch_size = patch_size
        proj_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, proj_dim)
        self.modulation = (
            mx.random.normal((1, 2, dim)) * (dim ** -0.5)
        ).astype(mx.float32)

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        # e 来自 time_projection: [B, dim*6], Head 只需最后 dim*2 部分
        b_e = e.shape[0]
        dim = self.modulation.shape[-1]
        e_head = e.reshape(b_e, 6, dim)[:, 4:6, :].reshape(b_e, 2 * dim)
        # reshape 为 [B, 1, 2, dim] 才能与 modulation (1,1,2,dim) 相加
        e_2 = e_head.reshape(b_e, 1, 2, dim).astype(mx.float32)
        # modulation in float32 (matching reference's autocast(float32))
        mod = self.modulation[:, None, :, :] + e_2  # float32
        e0 = mod[:, 0, 0, :]  # shift
        e1 = mod[:, 0, 1, :]  # scale

        x_norm = self.norm(x)
        x_mod = mul_add_add(x_norm, e1, e0)  # x * (1+scale) + shift
        out = self.head(x_mod)
        return out


# ---------------------------------------------------------------------------
# DiT 主干 Block (WanAttentionBlock 纯 MLX 端口)
# ---------------------------------------------------------------------------
class WanAttentionBlock(nn.Module):
    """SkyReels-V3 DiT 主干 Block, 对齐 WanAttentionBlock.

    结构:
      1. Self-Attention (空间, 非因果)
         x_mod = norm1(x) * (1+e1) + e0
         y = self_attn(x_mod, ...)
         x = x + y * e2
      2. Cross-Attention (文本/参考图引导)
         x_cross = norm3(x) if cross_attn_norm else x
         x = x + cross_attn(x_cross, context)
      3. FFN (前馈网络)
         x_mod = norm2(x) * (1+e4) + e3
         y = ffn(x_mod)
         x = x + y * e5

    modulation(1, 6, dim) float32 保 AdaLN-Zero 精度, 残差流全程 float32.
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
        has_temporal: bool = False,
        temporal_window: int = -1,
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
        self.has_temporal = has_temporal

        # Self-attention (空间分支)
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps,
            num_kv_heads=num_kv_heads,
        )

        # 可选时序分支 (V2V/A2V 启用)
        if has_temporal:
            self.temporal_attn = WanTemporalAttention(
                dim, num_heads, window_size=temporal_window,
                qk_norm=qk_norm, eps=eps,
            )
            self.norm_temporal = WanLayerNorm(dim, eps)

        # Cross-attention (文本/参考图引导)
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

        # Learned modulation: 6 vectors for scale/shift/gate (kept in float32 for precision)
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
    ) -> mx.array:
        """前向.

        Args:
            x: [B, L, dim] 视频隐空间
            e: [B, dim*6] 时间步嵌入 (经 time_projection)
            seq_lens: 每个样本的序列长度
            grid_sizes: 每个样本的 (F, H, W) 网格
            freqs: RoPE 频率表
            context: [B, L_ctx, dim] 文本/参考图上下文
            context_lens: 每个样本的 context 长度
            cross_kv_cache: 可选 cross-attention KV 缓存
            rope_cos_sin: 预计算的 rope cos/sin
            attn_mask: 注意力掩码
            temporal_len: 时序分支的帧数 (has_temporal=True 时必传)

        Returns:
            [B, L, dim] 经一个 DiT Block 处理后的隐空间
        """
        # Modulation: compute in float32 for precision
        # e 来自 time_projection: [B, dim*6], 需 reshape 为 [B, 6, dim] 才能与 modulation (1,6,dim) 相加
        b_e = e.shape[0]
        e_6 = e.reshape(b_e, 6, -1).astype(mx.float32)  # [B, 6, dim]
        mod = self.modulation + e_6  # float32, broadcast (1,6,dim) + (B,6,dim)
        e0, e1, e2, e3, e4, e5 = (
            mod[:, 0, :],  # shift for self-attn
            mod[:, 1, :],  # scale for self-attn
            mod[:, 2, :],  # gate for self-attn
            mod[:, 3, :],  # shift for ffn
            mod[:, 4, :],  # scale for ffn
            mod[:, 5, :],  # gate for ffn
        )

        # ---- 1. Self-Attention (空间) ----
        x_norm = self.norm1(x)
        x_mod = mul_add_add(x_norm, e1, e0)  # x*(1+scale)+shift
        y = self.self_attn(
            x_mod, seq_lens, grid_sizes, freqs,
            rope_cos_sin=rope_cos_sin, attn_mask=attn_mask,
        )
        x = mul_add(x, y, e2)  # x + y*gate

        # ---- 2. Optional Temporal Attention (时序分支) ----
        if self.has_temporal and temporal_len is not None:
            x_t = self.norm_temporal(x)
            y_t = self.temporal_attn(x_t, temporal_len, rope_cos_sin=rope_cos_sin)
            x = x + y_t

        # ---- 3. Cross-Attention (文本/参考图引导) ----
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
# 完整 R2V 14B 主干 (SkyReelsDiT)
# ---------------------------------------------------------------------------
class SkyReelsR2VDiT(nn.Module):
    """SkyReels-V3 R2V 14B-720P 完整 DiT 主干.

    配置:
      - dim=5120, ffn_dim=13824, num_heads=40, num_layers=40
      - patch_size=(1, 2, 2), in_dim=16, out_dim=16
      - text_dim=4096, text_len=512
      - cross_attn_type="i2v_cross_attn" (参考图引导)
      - window_size=(-1, -1) 全局注意力

    参数量: ~14B
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        # 默认 R2V 14B 配置
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

        # Patch embedding (Conv3d)
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

        # RoPE 频率表 (buffer)
        head_dim = self.dim // self.num_heads
        # 原版: rope_params(2048, d - 4*(d//6)) + rope_params(2048, 2*(d//6)) + rope_params(2048, 2*(d//6))
        # 即 t/h/w 三段: t 占 1/3, h 占 1/3, w 占 1/3 (复杂维度拆分)
        # 简化: 统一 rope_params(2048, head_dim)
        self.freqs = rope_params(2048, head_dim)

        # DiT Blocks
        self.blocks = [
            WanAttentionBlock(
                dim=self.dim,
                ffn_dim=self.ffn_dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                qk_norm=self.qk_norm,
                cross_attn_norm=self.cross_attn_norm,
                eps=self.eps,
                cross_attn_type=self.cross_attn_type,
                num_kv_heads=self.num_kv_heads,
                has_temporal=False,  # R2V 不用时序分支
            )
            for _ in range(self.num_layers)
        ]

        # Output head
        self.head = Head(self.dim, self.out_dim, self.patch_size, self.eps)

        # 编译优化 (M5 Max 默认开启)
        if _device.should_compile():
            self.blocks = [maybe_compile(b) for b in self.blocks]

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
    ) -> mx.array:
        """前向: 视频 latent + 时间步 + 文本 -> 去噪 latent.

        Args:
            x: [B, C_in, T, H, W] 视频 latent
            t: [B] 时间步
            context: [B, L_ctx, text_dim] 文本/参考图 embedding
            seq_lens: 每个样本的序列长度
            grid_sizes: 每个样本的 (F, H, W) 网格
            context_lens: context 长度
            rope_cos_sin: 预计算 rope
            attn_mask: 注意力掩码

        Returns:
            [B, C_out, T, H, W] 去噪 latent
        """
        b = x.shape[0]

        # 1. Patch embedding
        x = self.patch_embedding(x)  # [B, L, dim]

        # 2. Text embedding
        context = self.text_embedding(context)  # [B, L_ctx, dim]

        # 3. Time embedding
        # 原版: sinusoidal_embedding_1d(freq_dim, t) -> time_embedding -> time_projection
        t_emb = sinusoidal_embedding_1d(self.freq_dim, t)  # [B, freq_dim]
        t_emb = self.time_embedding(t_emb)  # [B, dim]
        e = self.time_projection(t_emb)  # [B, dim*6]

        # 4. DiT Blocks
        for block in self.blocks:
            x = block(
                x, e, seq_lens, grid_sizes, self.freqs,
                context, context_lens,
                rope_cos_sin=rope_cos_sin, attn_mask=attn_mask,
            )

        # 5. Output head
        out = self.head(x, e)  # [B, L, prod(patch_size)*out_dim]

        # 6. Unpatchify: [B, L, prod(patch_size)*out_dim] -> [B, C_out, T, H, W]
        out = self._unpatchify(out, grid_sizes)

        return out

    def _unpatchify(
        self,
        x: mx.array,
        grid_sizes: list,
    ) -> mx.array:
        """Unpatchify: [B, L, P*C_out] -> [B, C_out, T, H, W].

        Args:
            x: [B, L, prod(patch_size)*out_dim]
            grid_sizes: [(F, H, W), ...]

        Returns:
            [B, C_out, T, H, W]
        """
        b = x.shape[0]
        pt, ph, pw = self.patch_size
        outputs = []
        for i, (f, h, w) in enumerate(grid_sizes):
            # L = f * (h//ph) * (w//pw)
            h_p, w_p = h // ph, w // pw
            seq_len = f * h_p * w_p
            # x_i: [seq_len, pt*ph*pw*out_dim]
            x_i = x[i, :seq_len]  # [L, P*C_out]
            x_i = x_i.reshape(f, h_p, w_p, pt, ph, pw, self.out_dim)
            # 转置到 [C_out, T, H, W]
            x_i = x_i.transpose(6, 0, 3, 1, 4, 2, 5)  # [C_out, f, pt, h_p, ph, w_p, pw]
            x_i = x_i.reshape(
                self.out_dim, f * pt, h_p * ph, w_p * pw
            )  # [C_out, T, H, W]
            outputs.append(x_i)
        return mx.stack(outputs, axis=0)  # [B, C_out, T, H, W]


__all__ = [
    "WanAttentionBlock",
    "WanFFN",
    "Head",
    "SkyReelsR2VDiT",
]
