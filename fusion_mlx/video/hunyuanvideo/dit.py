# SPDX-License-Identifier: Apache-2.0
# HunyuanVideo DiT: dual-stream + single-stream transformer.
# Matches official weight format (856 keys).
# Called by: generate.py via HunyuanVideoDiT.from_pretrained()

import logging
import math

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

HUNYUAN_VIDEO_CONFIG = {
    "hidden_size": 3072,
    "num_heads": 24,
    "head_dim": 128,
    "num_layers": 20,
    "num_single_layers": 40,
    "num_refiner_layers": 2,
    "mlp_ratio": 4.0,
    "in_channels": 16,
    "out_channels": 16,
    "patch_size": (1, 2, 2),
    "text_embed_dim": 4096,
    "pooled_projection_dim": 768,
    "rope_theta": 256.0,
    "rope_axes_dim": (16, 56, 56),
    "guidance_embeds": True,
}


def _silu(x):
    return x * mx.sigmoid(x)


class _SiLU(nn.Module):
    def __call__(self, x):
        return _silu(x)


def _timestep_embedding(t, dim=256, max_period=10000):
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None] * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def _rms_norm(x, weight, eps=1e-6):
    x_type = x.dtype
    x = x.astype(mx.float32)
    rrms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * rrms).astype(x_type) * weight


def _apply_rope(x, freqs_cos, freqs_sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = mx.concatenate(
        [x1 * freqs_cos - x2 * freqs_sin, x1 * freqs_sin + x2 * freqs_cos],
        axis=-1,
    )
    return out


class _TimestepEmbed(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.in_layer = nn.Linear(256, hidden_size, bias=True)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, t):
        t_proj = _timestep_embedding(t)
        return self.out_layer(_silu(self.in_layer(t_proj)))


class _PooledEmbed(nn.Module):
    def __init__(self, hidden_size, in_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_size, bias=True)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, x):
        return self.out_layer(_silu(self.in_layer(x)))


class _TokenRefinerAttn(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=True)
        self.proj = nn.Linear(num_heads * head_dim, hidden_size, bias=True)

    def __call__(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.proj(out)


class _TokenRefinerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn = _TokenRefinerAttn(hidden_size, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Module()
        inner = hidden_size * 4
        setattr(self.mlp, "0", nn.Linear(hidden_size, inner, bias=True))
        setattr(self.mlp, "2", nn.Linear(inner, hidden_size, bias=True))
        self.adaLN_modulation = nn.Module()
        setattr(
            self.adaLN_modulation,
            "1",
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def __call__(self, x, emb):
        mlp_0 = getattr(self.mlp, "0")
        mlp_2 = getattr(self.mlp, "2")
        adaln_1 = getattr(self.adaLN_modulation, "1")
        gate_msa, gate_mlp = mx.split(adaln_1(_silu(emb)), 2, axis=-1)
        attn_out = self.self_attn(self.norm1(x))
        x = x + attn_out * mx.expand_dims(gate_msa, 1)
        mlp_out = mlp_2(_silu(mlp_0(self.norm2(x))))
        x = x + mlp_out * mx.expand_dims(gate_mlp, 1)
        return x


class _TokenRefiner(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, num_layers):
        super().__init__()
        self.input_embedder = nn.Linear(hidden_size, hidden_size, bias=True)
        self.t_embedder = _TimestepEmbed(hidden_size)
        container = nn.Module()
        for i in range(num_layers):
            block = _TokenRefinerBlock(hidden_size, num_heads, head_dim)
            setattr(container, str(i), block)
        self.blocks = container
        self.c_embedder = nn.Module()
        self.c_embedder.in_layer = nn.Linear(hidden_size, hidden_size, bias=True)
        self.c_embedder.out_layer = nn.Linear(hidden_size, hidden_size, bias=True)
        self.num_refiner_layers = num_layers

    def __call__(self, x, t):
        pooled = mx.mean(x, axis=1)
        emb = self.t_embedder(t) + self.c_embedder.out_layer(
            _silu(self.c_embedder.in_layer(pooled))
        )
        x = self.input_embedder(x)
        for i in range(self.num_refiner_layers):
            block = getattr(self.blocks, str(i))
            x = block(x, emb)
        return x


class _DoubleStreamAttn(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=True)
        self.proj = nn.Linear(num_heads * head_dim, hidden_size, bias=True)
        self.norm = nn.Module()
        self.norm.query_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm.key_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def _forward_attn(self, x, num_img_tokens=0, rope_cos=None, rope_sin=None):
        B, L, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = _rms_norm(q, self.norm.query_norm.weight)
        k = _rms_norm(k, self.norm.key_norm.weight)
        if rope_cos is not None and num_img_tokens > 0:
            q_img = q[:, :, :num_img_tokens]
            k_img = k[:, :, :num_img_tokens]
            q_img = _apply_rope(q_img, rope_cos, rope_sin)
            k_img = _apply_rope(k_img, rope_cos, rope_sin)
            q = mx.concatenate([q_img, q[:, :, num_img_tokens:]], axis=2)
            k = mx.concatenate([k_img, k[:, :, num_img_tokens:]], axis=2)
        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.proj(out)

    def __call__(self, img, txt, rope_cos=None, rope_sin=None):
        x = mx.concatenate([img, txt], axis=1)
        out = self._forward_attn(
            x, num_img_tokens=img.shape[1], rope_cos=rope_cos, rope_sin=rope_sin
        )
        img_out = out[:, : img.shape[1]]
        txt_out = out[:, img.shape[1] :]
        return img_out, txt_out


class _DoubleStreamMLP(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        setattr(self, "0", nn.Linear(hidden_size, inner, bias=True))
        setattr(self, "2", nn.Linear(inner, hidden_size, bias=True))

    def __call__(self, x):
        fc0 = getattr(self, "0")
        fc2 = getattr(self, "2")
        return fc2(_silu(fc0(x)))


class _DoubleStreamMod(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lin = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def __call__(self, emb):
        return mx.split(_silu(self.lin(emb)), 6, axis=-1)


class _DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, mlp_ratio=4.0):
        super().__init__()
        self.img_mod = _DoubleStreamMod(hidden_size)
        self.img_attn = _DoubleStreamAttn(hidden_size, num_heads, head_dim)
        self.img_mlp = _DoubleStreamMLP(hidden_size, mlp_ratio)
        self.txt_mod = _DoubleStreamMod(hidden_size)
        self.txt_attn = _DoubleStreamAttn(hidden_size, num_heads, head_dim)
        self.txt_mlp = _DoubleStreamMLP(hidden_size, mlp_ratio)
        self.img_norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.txt_norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.img_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.txt_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)

    def __call__(self, img, txt, emb, rope_cos=None, rope_sin=None):
        (
            shift_i,
            scale_i,
            gate_i,
            shift_i2,
            scale_i2,
            gate_i2,
        ) = self.img_mod(emb)
        (
            shift_t,
            scale_t,
            gate_t,
            shift_t2,
            scale_t2,
            gate_t2,
        ) = self.txt_mod(emb)
        img_normed = self.img_norm(img) * (
            1 + mx.expand_dims(scale_i, 1)
        ) + mx.expand_dims(shift_i, 1)
        txt_normed = self.txt_norm(txt) * (
            1 + mx.expand_dims(scale_t, 1)
        ) + mx.expand_dims(shift_t, 1)
        img_attn, txt_attn = self.img_attn(img_normed, txt_normed, rope_cos, rope_sin)
        img = img + img_attn * mx.expand_dims(gate_i, 1)
        txt = txt + txt_attn * mx.expand_dims(gate_t, 1)
        img_normed2 = self.img_norm2(img) * (
            1 + mx.expand_dims(scale_i2, 1)
        ) + mx.expand_dims(shift_i2, 1)
        txt_normed2 = self.txt_norm2(txt) * (
            1 + mx.expand_dims(scale_t2, 1)
        ) + mx.expand_dims(shift_t2, 1)
        img = img + self.img_mlp(img_normed2) * mx.expand_dims(gate_i2, 1)
        txt = txt + self.txt_mlp(txt_normed2) * mx.expand_dims(gate_t2, 1)
        return img, txt


class _SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        mlp_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(
            hidden_size, num_heads * head_dim * 3 + mlp_dim, bias=True
        )
        self.linear2 = nn.Linear(num_heads * head_dim + mlp_dim, hidden_size, bias=True)
        self.modulation = nn.Module()
        self.modulation.lin = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.norm = nn.Module()
        self.norm.query_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm.key_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self._ln = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)

    def __call__(self, x, emb, num_img_tokens=0, rope_cos=None, rope_sin=None):
        shift, scale, gate = mx.split(self.modulation.lin(_silu(emb)), 3, axis=-1)
        x_norm = self._ln(x)
        x_norm = x_norm * (1 + mx.expand_dims(scale, 1)) + mx.expand_dims(shift, 1)
        linear1_out = self.linear1(x_norm)
        attn_dim = self.num_heads * self.head_dim * 3
        qkv = linear1_out[:, :, :attn_dim]
        mlp_h = _silu(linear1_out[:, :, attn_dim:])
        B, L, _ = qkv.shape
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = _rms_norm(q, self.norm.query_norm.weight)
        k = _rms_norm(k, self.norm.key_norm.weight)
        if rope_cos is not None and num_img_tokens > 0:
            q_img = q[:, :, :num_img_tokens]
            k_img = k[:, :, :num_img_tokens]
            q_img = _apply_rope(q_img, rope_cos, rope_sin)
            k_img = _apply_rope(k_img, rope_cos, rope_sin)
            q = mx.concatenate([q_img, q[:, :, num_img_tokens:]], axis=2)
            k = mx.concatenate([k_img, k[:, :, num_img_tokens:]], axis=2)
        scale_f = self.head_dim**-0.5
        attn = (q * scale_f) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn, axis=-1)
        attn_out = attn @ v
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        combined = mx.concatenate([attn_out, mlp_h], axis=-1)
        out = self.linear2(combined)
        x = x + mx.expand_dims(gate, 1) * out
        return x


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        pt, ph, pw = patch_size
        self.adaLN_modulation = nn.Module()
        setattr(
            self.adaLN_modulation,
            "1",
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear = nn.Linear(hidden_size, pt * ph * pw * out_channels, bias=True)

    def __call__(self, x, emb):
        adaln_1 = getattr(self.adaLN_modulation, "1")
        shift, scale = mx.split(adaln_1(_silu(emb)), 2, axis=-1)
        x = self.norm(x) * (1 + mx.expand_dims(scale, 1)) + mx.expand_dims(shift, 1)
        return self.linear(x)


class HunyuanVideoDiT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = config or HUNYUAN_VIDEO_CONFIG
        self.hidden_size = cfg["hidden_size"]
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg.get("head_dim", self.hidden_size // self.num_heads)
        self.num_layers = cfg.get("num_layers", 20)
        self.num_single_layers = cfg.get("num_single_layers", 40)
        self.num_refiner_layers = cfg.get("num_refiner_layers", 2)
        self.patch_size = cfg["patch_size"]
        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg.get("out_channels", self.in_channels)
        self.text_embed_dim = cfg["text_embed_dim"]
        self.pooled_projection_dim = cfg.get("pooled_projection_dim", 768)
        self.rope_theta = cfg.get("rope_theta", 256.0)
        self.rope_axes_dim = cfg.get("rope_axes_dim", (16, 56, 56))
        self.guidance_embeds = cfg.get("guidance_embeds", True)

        pt, ph, pw = self.patch_size
        self.img_in = nn.Module()
        self.img_in.proj = nn.Module()
        self.img_in.proj.weight = mx.zeros(
            (self.hidden_size, self.in_channels, pt, ph, pw), dtype=mx.float32
        )
        self.img_in.proj.bias = mx.zeros((self.hidden_size,), dtype=mx.float32)
        self._patch_size = (pt, ph, pw)

        self.txt_in = _TokenRefiner(
            self.hidden_size,
            self.num_heads,
            self.head_dim,
            self.num_refiner_layers,
        )
        self.txt_in.input_embedder = nn.Linear(
            self.text_embed_dim, self.hidden_size, bias=True
        )

        self.time_in = _TimestepEmbed(self.hidden_size)
        self.vector_in = _PooledEmbed(self.hidden_size, self.pooled_projection_dim)
        self.guidance_in = None
        if self.guidance_embeds:
            self.guidance_in = _TimestepEmbed(self.hidden_size)

        self.double_blocks = nn.Module()
        for i in range(self.num_layers):
            block = _DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                self.head_dim,
                cfg["mlp_ratio"],
            )
            setattr(self.double_blocks, str(i), block)

        self.single_blocks = nn.Module()
        for i in range(self.num_single_layers):
            block = _SingleStreamBlock(
                self.hidden_size,
                self.num_heads,
                self.head_dim,
                cfg["mlp_ratio"],
            )
            setattr(self.single_blocks, str(i), block)

        self.final_layer = _FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        )

    def _compute_rope(self, ot, oh, ow, dtype=mx.float32):
        axes_grids = []
        for size, dim in zip([ot, oh, ow], self.rope_axes_dim):
            pos = mx.arange(size, dtype=dtype)
            freqs = 1.0 / (self.rope_theta ** (mx.arange(0, dim, 2, dtype=dtype) / dim))
            grid = mx.outer(pos, freqs)
            axes_grids.append(grid)
        t_freqs, h_freqs, w_freqs = axes_grids
        t_4d = t_freqs[:, None, None, :]
        h_4d = h_freqs[None, :, None, :]
        w_4d = w_freqs[None, None, :, :]
        t_full = mx.broadcast_to(t_4d, (ot, oh, ow, t_freqs.shape[-1]))
        h_full = mx.broadcast_to(h_4d, (ot, oh, ow, h_freqs.shape[-1]))
        w_full = mx.broadcast_to(w_4d, (ot, oh, ow, w_freqs.shape[-1]))
        freqs = mx.concatenate([t_full, h_full, w_full], axis=-1)
        L = ot * oh * ow
        freqs = freqs.reshape(L, -1)
        freqs_cos = mx.cos(freqs)
        freqs_sin = mx.sin(freqs)
        return freqs_cos, freqs_sin

    def __call__(
        self, x, timestep, text_emb, pooled_emb=None, guidance=None, image_cond=None
    ):
        B, C, T, H, W = x.shape
        pt, ph, pw = self._patch_size
        ot = T // pt
        oh = H // ph
        ow = W // pw

        w = self.img_in.proj.weight
        b = self.img_in.proj.bias
        x_thw = x.transpose(0, 2, 3, 4, 1)
        w_thw = w.transpose(0, 2, 3, 4, 1)
        img = mx.conv3d(x_thw, w_thw, stride=(pt, ph, pw))
        img = img.transpose(0, 4, 1, 2, 3)
        img = img + b.reshape(1, -1, 1, 1, 1)
        img = img.reshape(B, self.hidden_size, ot * oh * ow).transpose(0, 2, 1)

        txt = self.txt_in(text_emb, timestep)

        t_emb = self.time_in(timestep)
        if pooled_emb is None:
            pooled_emb = mx.zeros((B, self.pooled_projection_dim), dtype=x.dtype)
        v_emb = self.vector_in(pooled_emb)
        emb = t_emb + v_emb
        if self.guidance_in is not None:
            if guidance is None:
                guidance = mx.zeros_like(timestep)
            g_emb = self.guidance_in(guidance)
            emb = emb + g_emb

        rope_cos, rope_sin = self._compute_rope(ot, oh, ow, dtype=x.dtype)
        num_img_tokens = img.shape[1]
        rope_cos_img = rope_cos[None, None, :, :]
        rope_sin_img = rope_sin[None, None, :, :]

        for i in range(self.num_layers):
            block = getattr(self.double_blocks, str(i))
            img, txt = block(img, txt, emb, rope_cos_img, rope_sin_img)

        combined = mx.concatenate([img, txt], axis=1)
        for i in range(self.num_single_layers):
            block = getattr(self.single_blocks, str(i))
            combined = block(combined, emb, num_img_tokens, rope_cos_img, rope_sin_img)
        img = combined[:, :num_img_tokens]
        txt = combined[:, num_img_tokens:]

        img = self.final_layer(img, emb)
        img = img.reshape(B, ot, oh, ow, self.out_channels, pt, ph, pw)
        img = img.transpose(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(B, self.out_channels, ot * pt, oh * ph, ow * pw)
        return img

    @classmethod
    def from_pretrained(cls, model_path, config=None, **kwargs):
        import glob
        import os

        dit = cls(config=config)
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not safetensor_files:
            logger.warning("no safetensors at %s, random init", model_path)
            return dit
        from mlx.utils import tree_flatten

        all_params = {}
        for sf in safetensor_files:
            weights = mx.load(sf)
            all_params.update(weights)

        mapped = _remap_dit_weights(all_params)
        flat = tree_flatten(dit.parameters())
        loaded = {}
        matched = 0
        unmatched = []
        for k, v in flat:
            if k in mapped:
                loaded[k] = (
                    mapped[k].astype(mx.float16)
                    if mapped[k].dtype != mx.float16
                    else mapped[k]
                )
                matched += 1
            else:
                loaded[k] = v
                unmatched.append(k)
        logger.info(
            "hunyuan dit: loaded %d/%d params from %s", matched, len(flat), model_path
        )
        if unmatched:
            logger.debug(
                "hunyuan dit: unmatched params (%d): %s", len(unmatched), unmatched[:20]
            )

        nested = {}
        for key, val in loaded.items():
            parts = key.split(".")
            d = nested
            for p in parts[:-1]:
                if p not in d:
                    d[p] = {}
                d = d[p]
            d[parts[-1]] = val
        dit.update(nested)
        return dit


def _remap_dit_weights(params):
    out = {}
    for k, v in params.items():
        nk = k
        if nk.startswith("model.model."):
            nk = nk[len("model.model.") :]
        # RMSNorm: official uses .scale, MLX uses .weight
        nk = nk.replace("norm.query_norm.scale", "norm.query_norm.weight")
        nk = nk.replace("norm.key_norm.scale", "norm.key_norm.weight")
        # Token refiner: official nests under individual_token_refiner
        nk = nk.replace("txt_in.individual_token_refiner.", "txt_in.")
        out[nk] = v
    return out
