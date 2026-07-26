# SPDX-License-Identifier: Apache-2.0
# Cosmos DiT — dual-config: 7B T2V + 2B Predict2 I2V.
# 3D patchification, transformer blocks, adaptive layer-norm.

import logging
import math

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

COSMOS_7B_CONFIG = {
    "hidden_size": 4096,
    "num_heads": 32,
    "num_layers": 28,
    "mlp_ratio": 4.0,
    "patch_size": (2, 2, 2),
    "in_channels": 16,
    "out_channels": 16,
    "text_embed_dim": 4096,
    "rope_dim": 128,
    "caption_channels": 4096,
}

COSMOS_2B_CONFIG = {
    "hidden_size": 2048,
    "num_heads": 16,
    "num_layers": 14,
    "mlp_ratio": 4.0,
    "patch_size": (2, 2, 2),
    "in_channels": 16,
    "out_channels": 16,
    "text_embed_dim": 4096,
    "rope_dim": 64,
    "caption_channels": 4096,
}


def _silu(x):
    return x * mx.sigmoid(x)


class CosmosAdaLN(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, affine=False)
        self.adaLN_modulation = nn.Sequential(
            _SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def __call__(self, x, emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            self.adaLN_modulation(emb), 6, axis=-1
        )
        x_norm = self.norm(x) * (1 + scale_msa) + shift_msa
        return x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp


class _SiLU(nn.Module):
    def __call__(self, x):
        return _silu(x)


class CosmosAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, rope_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.rope_dim = rope_dim or self.head_dim

    def __call__(self, x, context=None, rope=None):
        B, L, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        if context is not None:
            k = (
                self.k_proj(context)
                .reshape(B, -1, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            v = (
                self.v_proj(context)
                .reshape(B, -1, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
        else:
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
        if rope is not None:
            q = _apply_rope(q, rope, self.rope_dim)
            if context is None:
                k = _apply_rope(k, rope, self.rope_dim)
        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


def _apply_rope(x, freqs, dim):
    if dim < x.shape[-1]:
        x_rot = x[..., :dim]
        x_pass = x[..., dim:]
    else:
        x_rot = x
        x_pass = None
    cos_f = mx.cos(freqs)
    sin_f = mx.sin(freqs)
    x1 = x_rot[..., ::2]
    x2 = x_rot[..., 1::2]
    cos_f = cos_f[..., ::2]
    sin_f = sin_f[..., ::2]
    rotated = mx.concatenate(
        [x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], axis=-1
    )
    if x_pass is not None:
        return mx.concatenate([rotated, x_pass], axis=-1)
    return rotated


class CosmosMLP(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner, bias=True)
        self.fc2 = nn.Linear(inner, hidden_size, bias=True)

    def __call__(self, x):
        return self.fc2(_silu(self.fc1(x)))


class CosmosDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, rope_dim=None):
        super().__init__()
        self.adaln = CosmosAdaLN(hidden_size)
        self.self_attn = CosmosAttention(hidden_size, num_heads, rope_dim)
        self.cross_attn = CosmosAttention(hidden_size, num_heads)
        self.mlp = CosmosMLP(hidden_size, mlp_ratio)

    def __call__(self, x, text_emb, adaln_emb, rope=None):
        x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(x, adaln_emb)
        attn_out = self.self_attn(x_norm, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm2 = self.adaln.norm(x) * (1 + scale_mlp) + shift_mlp
        cross_out = self.cross_attn(x_norm2, context=text_emb)
        x = x + cross_out
        mlp_out = self.mlp(self.adaln.norm(x) * (1 + scale_mlp) + shift_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class CosmosPatchEmbed3D(nn.Module):
    def __init__(self, in_channels=16, hidden_size=4096, patch_size=(2, 2, 2)):
        super().__init__()
        self.patch_size = patch_size
        pt, ph, pw = patch_size
        self.proj = nn.Linear(in_channels * pt * ph * pw, hidden_size, bias=True)

    def __call__(self, x):
        B, C, T, H, W = x.shape
        pt, ph, pw = self.patch_size
        ot = T // pt
        oh = H // ph
        ow = W // pw
        x = x.reshape(B, C, ot, pt, oh, ph, ow, pw)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(B, ot * oh * ow, C * pt * ph * pw)
        return self.proj(x), ot, oh, ow


class CosmosDiT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = config or COSMOS_7B_CONFIG
        self.hidden_size = cfg["hidden_size"]
        self.num_heads = cfg["num_heads"]
        self.num_layers = cfg["num_layers"]
        self.patch_size = cfg["patch_size"]
        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg["out_channels"]
        self.text_embed_dim = cfg["text_embed_dim"]
        self.rope_dim = cfg.get("rope_dim", self.hidden_size // self.num_heads)

        self.patch_embed = CosmosPatchEmbed3D(
            self.in_channels, self.hidden_size, self.patch_size
        )
        self.caption_proj = nn.Linear(self.text_embed_dim, self.hidden_size, bias=True)
        self.blocks = [
            CosmosDiTBlock(
                self.hidden_size,
                self.num_heads,
                cfg["mlp_ratio"],
                rope_dim=self.rope_dim,
            )
            for _ in range(self.num_layers)
        ]
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.final_linear = nn.Linear(
            self.hidden_size,
            self.out_channels
            * self.patch_size[0]
            * self.patch_size[1]
            * self.patch_size[2],
            bias=True,
        )
        self.t_embed = nn.Sequential(
            _SiLU(),
            nn.Linear(256, self.hidden_size, bias=True),
        )
        # Learnable position embedding
        self.pos_embed = None  # Set dynamically

    def _timestep_embedding(self, t, dim=256, max_period=10000):
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None] * freqs[None]
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

    def _compute_rope(self, ot, oh, ow, dtype=mx.float32):
        # 3D positional frequencies for RoPE
        t_coords = mx.arange(ot, dtype=dtype)
        h_coords = mx.arange(oh, dtype=dtype)
        w_coords = mx.arange(ow, dtype=dtype)
        freqs_t = t_coords[:, None] * mx.exp(
            -math.log(10000)
            * mx.arange(0, self.rope_dim // 3, 2, dtype=dtype)
            / (self.rope_dim // 3)
        )
        freqs_h = h_coords[:, None] * mx.exp(
            -math.log(10000)
            * mx.arange(0, self.rope_dim // 3, 2, dtype=dtype)
            / (self.rope_dim // 3)
        )
        freqs_w = w_coords[:, None] * mx.exp(
            -math.log(10000)
            * mx.arange(0, self.rope_dim - 2 * (self.rope_dim // 3), 2, dtype=dtype)
            / max(1, self.rope_dim - 2 * (self.rope_dim // 3))
        )
        # Build combined 3D rope for each spatial-temporal position
        grid_t, grid_h, grid_w = mx.meshgrid(t_coords, h_coords, w_coords)
        grid_t = grid_t.reshape(-1)
        grid_h = grid_h.reshape(-1)
        grid_w = grid_w.reshape(-1)
        # Simple 1D rope on flattened index
        indices = mx.arange(ot * oh * ow, dtype=dtype)
        dim = self.rope_dim
        freqs = indices[:, None] * mx.exp(
            -math.log(10000) * mx.arange(0, dim, 2, dtype=dtype) / dim
        )
        rope = mx.concatenate([freqs, freqs], axis=-1)  # (L, dim)
        return rope

    def __call__(self, x, timestep, text_emb, image_cond=None):
        B, C, T, H, W = x.shape
        # Patchify
        h, ot, oh, ow = self.patch_embed(x)
        L = ot * oh * ow
        # Timestep
        t_emb = self.t_embed(self._timestep_embedding(timestep))
        # Text projection
        text_proj = self.caption_proj(text_emb)  # (B, L_text, D)
        # I2V: add image conditioning
        if image_cond is not None:
            # image_cond: (B, C, 1, H, W) -> patchify and prepend
            img_patches, _, _, _ = self.patch_embed(image_cond)
            h = mx.concatenate([img_patches, h], axis=1)
        # RoPE
        rope = self._compute_rope(ot, oh, ow)
        # Transformer blocks
        for block in self.blocks:
            h = block(h, text_proj, t_emb, rope=rope)
        # Un-patchify
        h = self.final_norm(h)
        h = self.final_linear(h)
        pt, ph, pw = self.patch_size
        # If image_cond was prepended, strip those tokens
        if image_cond is not None:
            h = h[:, L:]
        h = h.reshape(B, ot, oh, ow, self.out_channels, pt, ph, pw)
        h = h.transpose(0, 4, 1, 5, 2, 6, 3, 7)
        h = h.reshape(B, self.out_channels, ot * pt, oh * ph, ow * pw)
        return h

    @classmethod
    def from_pretrained(cls, model_path, config=None, **kwargs):
        import os
        import glob

        dit = cls(config=config)
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not safetensor_files:
            logger.warning("no safetensors at %s, random init", model_path)
            return dit
        from mlx.utils import tree_flatten, tree_unflatten

        all_params = {}
        for sf in safetensor_files:
            from safetensors import safe_open

            with safe_open(sf, framework="mlx") as f:
                for key in f.keys():
                    all_params[key] = f.get_tensor(key)
        mapped = _remap_dit_weights(all_params)
        flat = tree_flatten(dit.parameters())
        loaded = {}
        for k, v in flat:
            if k in mapped:
                loaded[k] = mapped[k]
            else:
                loaded[k] = v
                logger.debug("dit: unmatched param %s", k)
        dit.update(tree_unflatten(loaded))
        return dit


def _remap_dit_weights(params):
    out = {}
    for k, v in params.items():
        nk = k
        # Diffusers naming -> our naming
        nk = nk.replace("transformer_blocks.", "blocks.")
        nk = nk.replace("attn1.", "self_attn.")
        nk = nk.replace("attn2.", "cross_attn.")
        nk = nk.replace("to_q.", "q_proj.")
        nk = nk.replace("to_k.", "k_proj.")
        nk = nk.replace("to_v.", "v_proj.")
        nk = nk.replace("to_out.0.", "out_proj.")
        nk = nk.replace("ff.net.0.proj.", "fc1.")
        nk = nk.replace("ff.net.2.", "fc2.")
        nk = nk.replace("norm1.", "adaln.norm.")
        nk = nk.replace("norm2.", "cross_attn_norm.")
        nk = nk.replace("adaLN_modulation.1.", "adaln.adaLN_modulation.1.")
        nk = nk.replace("pos_embed.", "pos_embed.")
        nk = nk.replace("caption_projection.", "caption_proj.")
        nk = nk.replace("proj_out.", "final_linear.")
        nk = nk.replace("norm_out.", "final_norm.")
        nk = nk.replace("adaln_input.", "t_embed.")
        out[nk] = v
    return out
