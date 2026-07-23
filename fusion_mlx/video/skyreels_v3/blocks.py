# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 DiT Block 统一层 (R2V/V2V/A2V 共用).

重构目的 (issue #186): 三个变体原本各写一个独立 Block 文件, 90% 代码重复且
modulation 广播 bug (#186) 只在 R2V 修了, V2V/A2V 仍是旧 [B,6,dim] 切片
[B,dim] -> B>1 (CFG b=2 / 投机 b=2K) 时与 [B,L,dim] 广播失败. 统一到一个
SkyReelsDiTBlock, modulation fix 一处生效, 三变体自动继承.

变体差异 (构造参数控制):
  - has_temporal: R2V=False (可选), V2V/A2V=True (强制时序分支)
  - temporal_window: R2V=-1, V2V=96, A2V=32
  - audio_cross_attn: A2V=True (norm_x + AudioCrossAttention), 其余 False
  - text_dim: R2V 传 (#125 vestigial, 存不用), V2V/A2V 不传

B=1 数值等价: 旧 V2V/A2V modulation reshape [B,6,dim] 切片 [B,dim] 与新
[B,6,1,dim] 切片 [B,1,dim] 在 B=1 时广播到 [1,L,dim] 结果一致 -> 重构
B=1-neutral. B>1 新版修复 (旧版崩溃).

权重 key 严格保持: norm1/norm2/norm3/self_attn/cross_attn/ffn/modulation
+ (temporal_attn/norm_temporal) + (norm_x/audio_cross_attn). load_weights
按 key 名映射, 类合并对权重加载透明.
"""

from __future__ import annotations

import logging
import math

import mlx.core as mx
import mlx.nn as nn

from .attention import (
    WAN_CROSSATTENTION_CLASSES,
    WanSelfAttention,
    WanTemporalAttention,
    _linear_dtype,
)
from .common import GELUApprox, WanLayerNorm, mul_add, mul_add_add

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
        # issue #164: 对齐 diffusers SkyReelsV2Transformer3DModel.norm_out
        # (FP32LayerNorm affine=False 无参). 旧 Head.norm 有参 -> head.norm.weight
        # 无源权重停留 init.
        self.norm = WanLayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, proj_dim)
        self.modulation = (mx.random.normal((1, 2, dim)) * (dim**-0.5)).astype(
            mx.float32
        )

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        # e 来自 time_projection: [B, dim*6], Head 只需最后 dim*2 部分
        b_e = e.shape[0]
        dim = self.modulation.shape[-1]
        e_head = e.reshape(b_e, 6, dim)[:, 4:6, :].reshape(b_e, 2 * dim)
        # reshape 为 [B, 1, 2, dim] 才能与 modulation (1,1,2,dim) 相加
        e_2 = e_head.reshape(b_e, 1, 2, dim).astype(mx.float32)
        # modulation in float32 (matching reference's autocast(float32))
        mod = self.modulation[:, None, :, :] + e_2  # float32
        e0 = mod[:, 0, 0:1, :]  # shift [B,1,dim]
        e1 = mod[:, 0, 1:2, :]  # scale [B,1,dim]

        x_norm = self.norm(x)
        x_mod = mul_add_add(x_norm, e1, e0)  # x * (1+scale) + shift
        out = self.head(x_mod)
        return out


# ---------------------------------------------------------------------------
# 音频 Embedding 分支 (A2V 独有, 叶子模块)
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
        audio_ctx = a[:, : self.dim].reshape(b, 1, self.dim)  # 截到 dim, 单 token
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
      - kv_linear: Linear(768, 10240)  (音频 context -> k+v 融合投影, 10240=dim*2)
      - q_linear:  Linear(5120, 5120)  (主干 latent -> query)
      - proj:      Linear(5120, 5120)  (输出投影)
      (无 norm_q/norm_k, 与 self_attn/cross_attn 的 RMSNorm 不同)

    前向逻辑 (推测, 对齐原版 WanA2VCrossAttention):
      q = q_linear(x)         # [B, L, dim]
      kv = kv_linear(audio_ctx)  # [B, L_a, dim*2] -> split k, v
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
        self.scale = self.head_dim**-0.5
        # q/k/v 投影 (命名严格对齐真实权重 audio_cross_attn.*)
        # 真实权重布局: kv_linear.weight=(768,10240)=(in,out), q_linear/proj=(5120,5120)=(out,in)?
        # mlx.nn.Linear 期望 (out,in), 故 kv_linear 需转置加载 (见 weights.py 映射)
        self.q_linear = nn.Linear(dim, dim)  # q_linear: [5120->5120]
        self.kv_linear = nn.Linear(
            audio_dim, dim * 2
        )  # kv_linear: [768->10240] (加载时转置)
        self.proj = nn.Linear(dim, dim)  # proj: [5120->5120]

    def __call__(
        self,
        x: mx.array,  # [B, L, dim] 主干 latent
        audio_ctx: mx.array,  # [B, L_a, audio_dim] 音频 context
    ) -> mx.array:
        b = x.shape[0]
        n, d = self.num_heads, self.head_dim
        w_dtype = _linear_dtype(self.q_linear)

        # q: 主干 latent -> [B, n, L, d]
        q = self.q_linear(x.astype(w_dtype))
        q = q.reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        # kv: 音频 context -> split k, v, 各 [B, n, L_a, d]
        kv = self.kv_linear(audio_ctx.astype(w_dtype))  # [B, L_a, dim*2]
        k, v = mx.split(kv, 2, axis=-1)  # 各 [B, L_a, dim]
        k = k.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        v = v.reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        # 标缩点积注意力 (无掩码, 非因果) - mx.matmul 广播批处理
        attn = mx.matmul(q, k.swapaxes(-1, -2)) * self.scale  # [B, n, L, L_a]
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)  # [B, n, L, d]

        out = out.transpose(0, 2, 1, 3).reshape(b, -1, n * d)  # [B, L, dim]
        return self.proj(out)


# ---------------------------------------------------------------------------
# 统一 DiT Block (R2V/V2V/A2V 共用主干)
# ---------------------------------------------------------------------------
class SkyReelsDiTBlock(nn.Module):
    """SkyReels-V3 DiT 主干 Block (R2V/V2V/A2V 统一).

    结构:
      1. Self-Attention (空间, 非因果)
         x_mod = norm1(x) * (1+e1) + e0; y = self_attn(x_mod); x = x + y*e2
      2. Temporal Attention (has_temporal 且 temporal_len 非空时)
         x_t = norm_temporal(x); y_t = temporal_attn(x_t); x = x + y_t
      3. Cross-Attention (文本/参考图引导)
         x_cross = norm2(x) if cross_attn_norm else x; x = x + cross_attn(x_cross, context)
      3b. Audio Cross-Attention (audio_cross_attn 且 audio_ctx 非空时, A2V 独有)
         x_a = norm_x(x); x = x + audio_cross_attn(x_a, audio_ctx)
      4. FFN
         x_mod = norm3(x) * (1+e4) + e3; y = ffn(x_mod); x = x + y*e5

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
        text_dim: (
            int | None
        ) = None,  # AtomCode fix #125: k/v 投影输入维度 (vestigial, 存不用)
        audio_cross_attn: bool = False,  # A2V 独有: 启用音频交叉注意力 + norm_x
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
        self.temporal_window = temporal_window
        self.text_dim = text_dim  # vestigial (#125): 不参与前向, 仅存以兼容 R2V 构造
        self.audio_cross_attn_enabled = audio_cross_attn

        # Self-attention (空间分支)
        # issue #164: 对齐 diffusers SkyReelsV2TransformerBlock.norm1
        # (FP32LayerNorm elementwise_affine=False, 无可学习参数).
        self.norm1 = WanLayerNorm(dim, eps, elementwise_affine=False)
        self.self_attn = WanSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            num_kv_heads=num_kv_heads,
        )

        # 可选时序分支 (V2V/A2V 启用)
        if has_temporal:
            self.temporal_attn = WanTemporalAttention(
                dim,
                num_heads,
                window_size=temporal_window,
                qk_norm=qk_norm,
                eps=eps,
            )
            self.norm_temporal = WanLayerNorm(dim, eps)

        # Cross-attention (文本/参考图引导)
        # issue #164: cross-attn 前 norm = self.norm2 (affine=True, 对齐 diffusers block.norm2).
        self.norm2 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else None
        )
        cross_cls = WAN_CROSSATTENTION_CLASSES.get(cross_attn_type)
        if cross_cls is None:
            raise ValueError(
                f"Unknown cross_attn_type: {cross_attn_type}. "
                f"Valid: {list(WAN_CROSSATTENTION_CLASSES)}"
            )
        self.cross_attn = cross_cls(dim, num_heads, qk_norm, eps)

        # Audio cross-attention (A2V 独有): norm_x (真实权重 blocks.N.norm_x, affine=True)
        # + AudioCrossAttention (num_heads 硬编码 40, 对齐真实 14B dim=5120/head_dim=128).
        if audio_cross_attn:
            self.norm_x = WanLayerNorm(dim, eps, elementwise_affine=True)
            self.audio_cross_attn = AudioCrossAttention(
                dim=dim,
                audio_dim=768,
                eps=eps,
            )
        else:
            self.audio_cross_attn = None

        # Feed-forward
        # issue #164: ffn 前 norm = self.norm3 (affine=False 无参, 对齐 diffusers block.norm3).
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = WanFFN(dim, ffn_dim)

        # Learned modulation: 6 vectors for scale/shift/gate (kept in float32 for precision)
        self.modulation = (mx.random.normal((1, 6, dim)) * (dim**-0.5)).astype(
            mx.float32
        )

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
        audio_ctx: mx.array | None = None,
    ) -> mx.array:
        # Modulation: float32 保精度.
        # issue #186: e 来自 time_projection [B, dim*6], reshape 为 [B,6,1,dim] 使切片
        # 为 [B,1,dim] 广播到 x_norm [B,L,dim]. 旧 [B,6,dim] 切片 [B,dim] 在 B>1
        # (CFG b=2 / 投机 b=2K) 时与 [B,L,dim] 广播失败 (L!=B) -> b=2 CFG 崩溃.
        # B=1 数值不变 ([1,1,dim] 与 [1,dim] 广播到 [1,L,dim] 等价).
        b_e = e.shape[0]
        e_6 = e.reshape(b_e, 6, 1, -1).astype(mx.float32)  # [B, 6, 1, dim]
        mod = self.modulation[:, :, None, :] + e_6  # (1,6,1,dim) + (B,6,1,dim)
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
            x_mod,
            seq_lens,
            grid_sizes,
            freqs,
            rope_cos_sin=rope_cos_sin,
            attn_mask=attn_mask,
        )
        x = mul_add(x, y, e2)  # x + y*gate

        # ---- 1b. AnimateDiff Motion Module (after self-attn, before temporal) ----
        # Callers: AnimateDiff adapter inject() sets motion_module attr + animatediff_scale.
        # No motion_module attr = no change (backward compatible). Scale applied per-block.
        if hasattr(self, "motion_module") and temporal_len is not None:
            ad_scale = getattr(self, "animatediff_scale", 1.0)
            x = x + self.motion_module(x, temporal_len) * ad_scale

        # ---- 2. Optional Temporal Attention (时序分支) ----
        if self.has_temporal and temporal_len is not None:
            x_t = self.norm_temporal(x)
            y_t = self.temporal_attn(x_t, temporal_len, rope_cos_sin=rope_cos_sin)
            x = x + y_t

        # ---- 3. Cross-Attention (文本/参考图引导) ----
        # issue #164: cross-attn 前 norm = self.norm2 (对齐 diffusers block.norm2)
        x_cross = self.norm2(x) if self.norm2 is not None else x
        y_c = self.cross_attn(
            x_cross,
            context,
            context_lens,
            kv_cache=cross_kv_cache,
        )
        x = x + y_c

        # ---- 3b. Audio Cross-Attention (纯音频 context 引导嘴型, A2V 独有) ----
        if self.audio_cross_attn is not None and audio_ctx is not None:
            x_a = self.norm_x(x)  # norm_x: 真实权重 blocks.N.norm_x (affine=True)
            y_a = self.audio_cross_attn(x_a, audio_ctx)
            x = x + y_a

        # ---- 4. FFN ----
        # issue #164: ffn 前 norm = self.norm3 (对齐 diffusers block.norm3, 无参)
        x_norm3 = self.norm3(x)
        x_mod2 = mul_add_add(x_norm3, e4, e3)
        y_f = self.ffn(x_mod2)
        x = mul_add(x, y_f, e5)

        return x


# ---------------------------------------------------------------------------
# 变体薄子类: 保留公共名 + 各自默认参数, DiT 构造调用不变
# ---------------------------------------------------------------------------
class WanAttentionBlock(SkyReelsDiTBlock):
    """R2V 14B DiT Block (has_temporal 可选, 默认关, 传 text_dim)."""

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
        text_dim: int | None = None,  # AtomCode fix #125
    ):
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            cross_attn_type=cross_attn_type,
            num_kv_heads=num_kv_heads,
            has_temporal=has_temporal,
            temporal_window=temporal_window,
            text_dim=text_dim,
            audio_cross_attn=False,
        )


class V2VAttentionBlock(SkyReelsDiTBlock):
    """V2V 视频续写 DiT Block (强制时序分支, temporal_window 默认 96)."""

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
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            cross_attn_type=cross_attn_type,
            num_kv_heads=num_kv_heads,
            has_temporal=True,  # V2V 强制时序分支
            temporal_window=temporal_window,
            text_dim=None,
            audio_cross_attn=False,
        )


class A2VAttentionBlock(SkyReelsDiTBlock):
    """A2V 数字人 DiT Block (强制时序分支 + 音频交叉注意力, temporal_window 默认 32)."""

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
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            cross_attn_type=cross_attn_type,
            num_kv_heads=num_kv_heads,
            has_temporal=True,  # A2V 强制时序分支 (保嘴型连贯)
            temporal_window=temporal_window,
            text_dim=None,
            audio_cross_attn=True,  # A2V 独有: 音频交叉注意力
        )


__all__ = [
    "WanFFN",
    "Head",
    "AudioEmbedding",
    "AudioCrossAttention",
    "SkyReelsDiTBlock",
    "WanAttentionBlock",
    "V2VAttentionBlock",
    "A2VAttentionBlock",
]
