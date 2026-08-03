# SPDX-License-Identifier: Apache-2.0
# Open-Sora V2 MMDiT (modified Flux) transformer port to MLX.
# DoubleStreamBlock: separate img/txt streams with joint attention
# SingleStreamBlock: merged stream with fused qkv+mlp

import logging
import math

import mlx.core as mx
import mlx.nn as nn

from .rope import EmbedND, apply_rope

logger = logging.getLogger(__name__)


def _silu(x):
    return x * mx.sigmoid(x)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, x):
        return self.out_layer(_silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        x_f = x.astype(mx.float32)
        rrms = mx.rsqrt(mx.mean(x_f**2, axis=-1, keepdims=True) + self.eps)
        return (x_f * rrms).astype(x.dtype) * self.weight


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def __call__(self, q, k):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, fused_qkv=False):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        self.head_dim = dim // num_heads
        if fused_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.norm = QKNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, pe):
        B, L, _ = x.shape
        if self.fused_qkv:
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
            qkv = qkv.transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = (
                self.q_proj(x)
                .reshape(B, L, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            k = (
                self.k_proj(x)
                .reshape(B, L, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            v = (
                self.v_proj(x)
                .reshape(B, L, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
        q, k = self.norm(q, k)
        q, k = apply_rope(q, k, pe)
        scale = self.head_dim**-0.5
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.proj(out)


class Modulation(nn.Module):
    def __init__(self, dim, double=True):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def __call__(self, vec):
        out = self.lin(_silu(vec))
        parts = mx.split(out, self.multiplier, axis=-1)
        shift1, scale1, gate1 = parts[0], parts[1], parts[2]
        if self.is_double:
            shift2, scale2, gate2 = parts[3], parts[4], parts[5]
            return (shift1, scale1, gate1), (shift2, scale2, gate2)
        return (shift1, scale1, gate1), None


class DoubleStreamBlock(nn.Module):
    def __init__(
        self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=False, fused_qkv=False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias, fused_qkv)
        self.img_norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias, fused_qkv)
        self.txt_norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def __call__(self, img, txt, vec, pe):
        (img_s1, img_sc1, img_g1), (img_s2, img_sc2, img_g2) = self.img_mod(vec)
        (txt_s1, txt_sc1, txt_g1), (txt_s2, txt_sc2, txt_g2) = self.txt_mod(vec)

        img_mod = (1 + img_sc1[:, None, :]) * self.img_norm1(img) + img_s1[:, None, :]
        img_q = (
            self.img_attn.q_proj(img_mod)
            .reshape(img_mod.shape[0], img_mod.shape[1], self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        img_k = (
            self.img_attn.k_proj(img_mod)
            .reshape(img_mod.shape[0], img_mod.shape[1], self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        img_v = (
            self.img_attn.v_proj(img_mod)
            .reshape(img_mod.shape[0], img_mod.shape[1], self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k)

        txt_mod = (1 + txt_sc1[:, None, :]) * self.txt_norm1(txt) + txt_s1[:, None, :]
        txt_q = (
            self.txt_attn.q_proj(txt_mod)
            .reshape(txt_mod.shape[0], txt_mod.shape[1], self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        txt_k = (
            self.txt_attn.k_proj(txt_mod)
            .reshape(txt_mod.shape[0], txt_mod.shape[1], self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        txt_v = (
            self.txt_attn.v_proj(txt_mod)
            .reshape(txt_mod.shape[0], txt_mod.shape[1], self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

        q = mx.concatenate([txt_q, img_q], axis=2)
        k = mx.concatenate([txt_k, img_k], axis=2)
        v = mx.concatenate([txt_v, img_v], axis=2)

        q, k = apply_rope(q, k, pe)
        scale = self.head_dim**-0.5
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = mx.softmax(attn, axis=-1)
        attn_out = mx.matmul(attn, v)

        txt_len = txt_q.shape[2]
        txt_attn_out = (
            attn_out[:, :, :txt_len, :]
            .transpose(0, 2, 1, 3)
            .reshape(txt.shape[0], txt.shape[1], -1)
        )
        img_attn_out = (
            attn_out[:, :, txt_len:, :]
            .transpose(0, 2, 1, 3)
            .reshape(img.shape[0], img.shape[1], -1)
        )

        img = img + img_g1[:, None, :] * self.img_attn.proj(img_attn_out)
        img = img + img_g2[:, None, :] * self.img_mlp(
            (1 + img_sc2[:, None, :]) * self.img_norm2(img) + img_s2[:, None, :]
        )

        txt = txt + txt_g1[:, None, :] * self.txt_attn.proj(txt_attn_out)
        txt = txt + txt_g2[:, None, :] * self.txt_mlp(
            (1 + txt_sc2[:, None, :]) * self.txt_norm2(txt) + txt_s2[:, None, :]
        )

        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, fused_qkv=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.fused_qkv = fused_qkv

        if fused_qkv:
            self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_mlp = nn.Linear(hidden_size, hidden_size + self.mlp_hidden_dim)

        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(self.head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.modulation = Modulation(hidden_size, double=False)

    def __call__(self, x, vec, pe):
        (shift, scale, gate), _ = self.modulation(vec)
        x_mod = (1 + scale[:, None, :]) * self.pre_norm(x) + shift[:, None, :]

        B, L, _ = x_mod.shape
        if self.fused_qkv:
            combined = self.linear1(x_mod)
            qkv, mlp = mx.split(combined, [3 * self.hidden_size], axis=-1)
            qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim).transpose(
                2, 0, 3, 1, 4
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = (
                self.q_proj(x_mod)
                .reshape(B, L, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            k = (
                self.k_proj(x_mod)
                .reshape(B, L, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            v_mlp = self.v_mlp(x_mod)
            v, mlp = mx.split(v_mlp, [self.hidden_size], axis=-1)
            v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k = self.norm(q, k)
        q, k = apply_rope(q, k, pe)

        scale_val = self.head_dim**-0.5
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale_val
        attn = mx.softmax(attn, axis=-1)
        attn_out = mx.matmul(attn, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        mlp_activated = mlp * mx.sigmoid(1.702 * mlp)

        output = self.linear2(mx.concatenate([attn_out, mlp_activated], axis=-1))
        return x + gate[:, None, :] * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def __call__(self, x, vec):
        modulation = _silu(vec)
        modulation = self.adaLN_modulation(modulation)
        shift, scale = mx.split(modulation, 2, axis=-1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)


class MMDiTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        head_dim = config.hidden_size // config.num_heads
        pe_dim = head_dim
        assert (
            sum(config.axes_dim) == pe_dim
        ), f"axes_dim sum {sum(config.axes_dim)} != head_dim {pe_dim}"

        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(256, self.hidden_size)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size)

        if config.guidance_embed:
            self.guidance_in = MLPEmbedder(256, self.hidden_size)
        else:
            self.guidance_in = None

        if config.cond_embed:
            self.cond_in = nn.Linear(
                self.in_channels + self.patch_size**2, self.hidden_size, bias=True
            )
        else:
            self.cond_in = None

        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size)

        self.double_blocks = [
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                fused_qkv=config.fused_qkv,
            )
            for _ in range(config.depth)
        ]

        self.single_blocks = [
            SingleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=config.mlp_ratio,
                fused_qkv=config.fused_qkv,
            )
            for _ in range(config.depth_single_blocks)
        ]

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def __call__(
        self, img, img_ids, txt, txt_ids, timesteps, y_vec, cond=None, guidance=None
    ):
        img = self.img_in(img)
        if self.cond_in is not None:
            if cond is None:
                logger.warning("cond_embed=True but no cond provided, using zeros")
                cond = mx.zeros(
                    (img.shape[0], img.shape[1], self.in_channels + self.patch_size**2),
                    dtype=img.dtype,
                )
            img = img + self.cond_in(cond)

        vec = self.time_in(self._timestep_embedding(timesteps, 256))
        if self.guidance_in is not None:
            if guidance is None:
                logger.warning("guidance_embed=True but no guidance provided")
                guidance = mx.zeros((img.shape[0],), dtype=img.dtype)
            vec = vec + self.guidance_in(self._timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y_vec)

        txt = self.txt_in(txt)

        ids = mx.concatenate([txt_ids, img_ids], axis=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img, txt, vec, pe)

        x = mx.concatenate([txt, img], axis=1)
        for block in self.single_blocks:
            x = block(x, vec, pe)

        txt_len = txt.shape[1]
        img = x[:, txt_len:, :]

        return self.final_layer(img, vec)

    @staticmethod
    def _timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
        t = time_factor * t
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding.astype(t.dtype)
