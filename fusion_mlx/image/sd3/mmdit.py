import logging
import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

logger = logging.getLogger(__name__)


def timestep_embedding(
    t: mx.array, dim: int = 256, max_period: int = 10000
) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
    return emb


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu_approx(x)
        x = self.fc2(x)
        return x


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool = True):
        super().__init__()
        self.is_double = double
        self.linear = nn.Linear(dim, dim * (6 if double else 2))

    def __call__(self, vec: mx.array):
        out = self.linear(nn.silu(vec))
        if self.is_double:
            return mx.split(out, 6, axis=-1)
        return mx.split(out, 2, axis=-1)


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + scale[:, None]) + shift[:, None]


class JointAttention(nn.Module):
    def __init__(
        self, dim: int, heads: int, head_dim: int, context_pre_only: bool = False
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = dim
        self.context_pre_only = context_pre_only
        self.x_qkv = nn.Linear(dim, dim * 3)
        self.x_proj = nn.Linear(dim, dim)
        self.context_qkv = nn.Linear(dim, dim * 3)
        if not context_pre_only:
            self.context_proj = nn.Linear(dim, dim)

    def _split_qkv(self, qkv: mx.array, b: int, seq: int):
        qkv = mx.reshape(qkv, (b, seq, self.heads, self.head_dim * 3))
        q, k, v = mx.split(qkv, 3, axis=-1)
        return q, k, v

    def __call__(self, x: mx.array, context: mx.array):
        b, sx, _ = x.shape
        _, sc, _ = context.shape
        xq, xk, xv = self._split_qkv(self.x_qkv(x), b, sx)
        cq, ck, cv = self._split_qkv(self.context_qkv(context), b, sc)
        q = mx.concatenate([xq, cq], axis=1)
        k = mx.concatenate([xk, ck], axis=1)
        v = mx.concatenate([xv, cv], axis=1)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        scale = 1.0 / math.sqrt(self.head_dim)
        out = scaled_dot_product_attention(q, k, v, scale=scale)
        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (b, sx + sc, self.inner_dim))
        x_out = self.x_proj(out[:, :sx])
        if self.context_pre_only:
            return x_out, None
        ctx_out = self.context_proj(out[:, sx:])
        return x_out, ctx_out


class JointBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, context_pre_only: bool):
        super().__init__()
        self.context_pre_only = context_pre_only
        self.x_norm1 = nn.LayerNorm(dim, eps=1e-6, bias=False, affine=False)
        self.x_mod = Modulation(dim, double=True)
        self.context_norm1 = nn.LayerNorm(dim, eps=1e-6, bias=False, affine=False)
        self.context_mod = Modulation(dim, double=not context_pre_only)
        self.attn = JointAttention(
            dim, heads, head_dim, context_pre_only=context_pre_only
        )
        self.x_norm2 = nn.LayerNorm(dim, eps=1e-6, bias=False, affine=False)
        self.x_mlp = Mlp(dim, dim * 4)
        if not context_pre_only:
            self.context_norm2 = nn.LayerNorm(dim, eps=1e-6, bias=False, affine=False)
            self.context_mlp = Mlp(dim, dim * 4)

    def __call__(self, x: mx.array, context: mx.array, vec: mx.array):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.x_mod(vec)
        x_norm = modulate(self.x_norm1(x), shift_msa, scale_msa)
        if self.context_pre_only:
            c_shift, c_scale = self.context_mod(vec)
            c_norm = modulate(self.context_norm1(context), c_shift, c_scale)
            attn_x, _ = self.attn(x_norm, c_norm)
            x = x + gate_msa[:, None] * attn_x
            x_norm2 = modulate(self.x_norm2(x), shift_mlp, scale_mlp)
            x = x + gate_mlp[:, None] * self.x_mlp(x_norm2)
            return x, None
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.context_mod(vec)
        )
        c_norm = modulate(self.context_norm1(context), c_shift_msa, c_scale_msa)
        attn_x, attn_c = self.attn(x_norm, c_norm)
        x = x + gate_msa[:, None] * attn_x
        context = context + c_gate_msa[:, None] * attn_c
        x_norm2 = modulate(self.x_norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None] * self.x_mlp(x_norm2)
        c_norm2 = modulate(self.context_norm2(context), c_shift_mlp, c_scale_mlp)
        context = context + c_gate_mlp[:, None] * self.context_mlp(c_norm2)
        return x, context


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, bias=False, affine=False)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.mod = Modulation(dim, double=False)

    def __call__(self, x: mx.array, vec: mx.array) -> mx.array:
        shift, scale = self.mod(vec)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class Embedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )


class MMDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.inner_dim
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.pos_embed = mx.zeros((1, config.pos_embed_max_size**2, dim))
        self.t_embedder = Embedder(256, dim)
        self.y_embedder = Embedder(config.pooled_projection_dim, dim)
        self.context_embedder = nn.Linear(config.joint_attention_dim, dim)
        self.x_embedder_proj = nn.Conv2d(
            self.in_channels,
            dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.joint_blocks = [
            JointBlock(
                dim,
                heads,
                head_dim,
                context_pre_only=(i == config.num_layers - 1),
            )
            for i in range(config.num_layers)
        ]
        self.final_layer = FinalLayer(dim, self.patch_size, self.out_channels)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        pooled: mx.array,
        context: mx.array,
        h: int,
        w: int,
    ) -> mx.array:
        b = x.shape[0]
        x = self._patchify(x, h, w)
        context = self.context_embedder(context)
        t_emb = self.t_embedder.mlp(timestep_embedding(t, 256))
        y_emb = self.y_embedder.mlp(pooled)
        vec = t_emb + y_emb
        for block in self.joint_blocks:
            x, context = block(x, context, vec)
        x = self.final_layer(x, vec)
        return self._unpatchify(x, b, h, w)

    def _patchify(self, x: mx.array, h: int, w: int) -> mx.array:
        b = x.shape[0]
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.x_embedder_proj(x)
        x = mx.transpose(x, (0, 3, 1, 2))
        x = mx.reshape(x, (b, x.shape[1], h // self.patch_size * w // self.patch_size))
        x = mx.transpose(x, (0, 2, 1))
        pos = self._get_pos(h, w)
        return x + pos

    def _get_pos(self, h: int, w: int) -> mx.array:
        patch = self.patch_size
        hp, wp = h // patch, w // patch
        size = self.pos_embed.shape[1]
        max_side = int(round(math.sqrt(size)))
        pos = mx.reshape(
            self.pos_embed, (1, max_side, max_side, self.pos_embed.shape[-1])
        )
        pos = pos[:, :hp, :wp]
        return mx.reshape(pos, (1, hp * wp, -1))

    def _unpatchify(self, x: mx.array, b: int, h: int, w: int) -> mx.array:
        patch = self.patch_size
        hp, wp = h // patch, w // patch
        x = mx.reshape(x, (b, hp, wp, patch, patch, self.out_channels))
        x = mx.transpose(x, (0, 5, 1, 3, 2, 4))
        return mx.reshape(x, (b, self.out_channels, h, w))
