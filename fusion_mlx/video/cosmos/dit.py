# SPDX-License-Identifier: Apache-2.0
# Cosmos DiT — rewritten to match official NVIDIA weight structure.
# Triple AdaLN (self_attn / cross_attn / mlp), QK-norm, cross-attn GQA,
# learnable positional embed + 3D RoPE, GELU MLP.

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
    "patch_size": (1, 2, 2),
    "in_channels": 16,
    "out_channels": 16,
    "text_embed_dim": 1024,
    "rope_dim": 128,
    "adaln_lora_dim": 256,
    "concat_padding_mask": True,
    "max_size": (128, 240, 240),
    "rope_scale": (2.0, 1.0, 1.0),
}

COSMOS_2B_CONFIG = {
    "hidden_size": 2048,
    "num_heads": 16,
    "num_layers": 28,
    "mlp_ratio": 4.0,
    "patch_size": (1, 2, 2),
    "in_channels": 16,
    "out_channels": 16,
    "text_embed_dim": 1024,
    "rope_dim": 128,
    "adaln_lora_dim": 256,
    "concat_padding_mask": True,
    "max_size": (128, 240, 240),
    "rope_scale": (2.0, 1.0, 1.0),
}


class _AdaLNZero(nn.Module):
    def __init__(self, hidden_size, lora_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear_1 = nn.Linear(hidden_size, lora_dim, bias=False)
        self.linear_2 = nn.Linear(lora_dim, 3 * hidden_size, bias=False)

    def __call__(self, x, embedded_t, temb=None):
        emb = nn.silu(embedded_t)
        emb = self.linear_1(emb)
        emb = self.linear_2(emb)
        if temb is not None:
            emb = emb + temb
        shift, scale, gate = mx.split(emb, 3, axis=-1)
        x_norm = self.norm(x)
        if shift.ndim == 2:
            shift = mx.expand_dims(shift, 1)
            scale = mx.expand_dims(scale, 1)
            gate = mx.expand_dims(gate, 1)
        x_norm = x_norm * (1 + scale) + shift
        return x_norm, gate


class _AdaLNOut(nn.Module):
    def __init__(self, hidden_size, lora_dim=256):
        super().__init__()
        self._hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear_1 = nn.Linear(hidden_size, lora_dim, bias=False)
        self.linear_2 = nn.Linear(lora_dim, 2 * hidden_size, bias=False)

    def __call__(self, x, embedded_t, temb=None):
        emb = nn.silu(embedded_t)
        emb = self.linear_1(emb)
        emb = self.linear_2(emb)
        if temb is not None:
            emb = emb + temb[..., : 2 * self._hidden_size]
        shift, scale = mx.split(emb, 2, axis=-1)
        x_norm = self.norm(x)
        if shift.ndim == 2:
            shift = mx.expand_dims(shift, 1)
            scale = mx.expand_dims(scale, 1)
        x_norm = x_norm * (1 + scale) + shift
        return x_norm


class _SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, rope_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.rope_dim = rope_dim or self.head_dim

    def __call__(self, x, rope_cos=None, rope_sin=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_cos is not None and rope_sin is not None:
            q = _apply_rope(q, rope_cos, rope_sin, self.rope_dim)
            k = _apply_rope(k, rope_cos, rope_sin, self.rope_dim)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.output_proj(out)


class _CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, text_embed_dim=1024):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(text_embed_dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(text_embed_dim, hidden_size, bias=False)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def __call__(self, x, context):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.output_proj(out)


class _MLP(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.layer1 = nn.Linear(hidden_size, inner, bias=False)
        self.layer2 = nn.Linear(inner, hidden_size, bias=False)

    def __call__(self, x):
        return self.layer2(nn.gelu(self.layer1(x)))


class _CosmosBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, rope_dim=None,
                 adaln_lora_dim=256, text_embed_dim=1024):
        super().__init__()
        self.adaln_modulation_self_attn = _AdaLNZero(hidden_size, adaln_lora_dim)
        self.self_attn = _SelfAttention(hidden_size, num_heads, rope_dim)
        self.adaln_modulation_cross_attn = _AdaLNZero(hidden_size, adaln_lora_dim)
        self.cross_attn = _CrossAttention(hidden_size, num_heads, text_embed_dim)
        self.adaln_modulation_mlp = _AdaLNZero(hidden_size, adaln_lora_dim)
        self.mlp = _MLP(hidden_size, mlp_ratio)

    def __call__(self, x, text_emb, embedded_t, temb, rope_cos=None, rope_sin=None,
                 extra_pos_emb=None):
        if extra_pos_emb is not None:
            x = x + extra_pos_emb
        x_norm, gate = self.adaln_modulation_self_attn(x, embedded_t, temb)
        attn_out = self.self_attn(x_norm, rope_cos, rope_sin)
        x = x + gate * attn_out
        x_norm, gate = self.adaln_modulation_cross_attn(x, embedded_t, temb)
        cross_out = self.cross_attn(x_norm, text_emb)
        x = x + gate * cross_out
        x_norm, gate = self.adaln_modulation_mlp(x, embedded_t, temb)
        mlp_out = self.mlp(x_norm)
        x = x + gate * mlp_out
        return x


class _PatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size=(1, 2, 2)):
        super().__init__()
        self.patch_size = patch_size
        pt, ph, pw = patch_size
        self.proj = nn.Linear(in_channels * pt * ph * pw, hidden_size, bias=False)

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


class _LearnablePosEmbed(nn.Module):
    def __init__(self, hidden_size, max_size=(128, 240, 240), patch_size=(1, 2, 2)):
        super().__init__()
        self.patch_size = patch_size
        max_t = max_size[0] // patch_size[0]
        max_h = max_size[1] // patch_size[1]
        max_w = max_size[2] // patch_size[2]
        self.pos_emb_t = mx.zeros((max_t, hidden_size))
        self.pos_emb_h = mx.zeros((max_h, hidden_size))
        self.pos_emb_w = mx.zeros((max_w, hidden_size))
        self._max_t = max_t
        self._max_h = max_h
        self._max_w = max_w

    def __call__(self, ot, oh, ow, B):
        emb_t = self.pos_emb_t[:ot][None, :, None, None, :]
        emb_t = mx.broadcast_to(emb_t, (B, ot, oh, ow, emb_t.shape[-1]))
        emb_h = self.pos_emb_h[:oh][None, None, :, None, :]
        emb_h = mx.broadcast_to(emb_h, (B, ot, oh, ow, emb_h.shape[-1]))
        emb_w = self.pos_emb_w[:ow][None, None, None, :, :]
        emb_w = mx.broadcast_to(emb_w, (B, ot, oh, ow, emb_w.shape[-1]))
        emb = emb_t + emb_h + emb_w
        emb = emb.reshape(B, ot * oh * ow, -1)
        norm = mx.linalg.norm(emb.astype(mx.float32), axis=-1, keepdims=True)
        n_elements = emb.shape[0] * emb.shape[1]
        norm = 1e-6 + norm * math.sqrt(emb.shape[-1] / n_elements)
        return (emb.astype(mx.float32) / norm).astype(emb.dtype)


class _TimestepEmbedding(nn.Module):
    def __init__(self, dim, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(dim, hidden_size, bias=False)
        self.linear_2 = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def __call__(self, t):
        t = nn.silu(self.linear_1(t))
        t = self.linear_2(t)
        return t


class _CosmosEmbedding(nn.Module):
    def __init__(self, dim, hidden_size):
        super().__init__()
        self.time_proj_dim = dim
        self.t_embedder = _TimestepEmbedding(dim, hidden_size)
        self.t_embedding_norm = nn.RMSNorm(dim, eps=1e-6)

    def __call__(self, timestep_emb):
        temb = self.t_embedder(timestep_emb)
        embedded_t = self.t_embedding_norm(timestep_emb)
        return temb, embedded_t


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
        self.adaln_lora_dim = cfg.get("adaln_lora_dim", 256)
        self.concat_padding_mask = cfg.get("concat_padding_mask", True)
        self.max_size = cfg.get("max_size", (128, 240, 240))
        self.rope_scale = cfg.get("rope_scale", (2.0, 1.0, 1.0))

        patch_in_ch = self.in_channels + (1 if self.concat_padding_mask else 0)
        self.x_embedder = _PatchEmbed(patch_in_ch, self.hidden_size, self.patch_size)
        self.learnable_pos_embed = _LearnablePosEmbed(
            self.hidden_size, self.max_size, self.patch_size
        )
        self.t_embedder = _CosmosEmbedding(self.hidden_size, self.hidden_size)
        self.blocks = nn.Module()
        for i in range(self.num_layers):
            block = _CosmosBlock(
                self.hidden_size,
                self.num_heads,
                cfg["mlp_ratio"],
                rope_dim=self.rope_dim,
                adaln_lora_dim=self.adaln_lora_dim,
                text_embed_dim=self.text_embed_dim,
            )
            setattr(self.blocks, str(i), block)
        self.final_layer = _AdaLNOut(self.hidden_size, self.adaln_lora_dim)
        self.final_linear = nn.Linear(
            self.hidden_size,
            self.out_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
            bias=False,
        )

    def _timestep_embedding(self, t, dim=256, max_period=10000):
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None] * freqs[None]
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

    def _compute_rope(self, ot, oh, ow, fps=None, base_fps=24, dtype=mx.float32):
        head_dim = self.rope_dim
        dim_h = head_dim // 6 * 2
        dim_w = head_dim // 6 * 2
        dim_t = head_dim - dim_h - dim_w
        h_ntk = self.rope_scale[1] ** (dim_h / (dim_h - 2))
        w_ntk = self.rope_scale[2] ** (dim_w / (dim_w - 2))
        t_ntk = self.rope_scale[0] ** (dim_t / (dim_t - 2))
        h_theta = 10000.0 * h_ntk
        w_theta = 10000.0 * w_ntk
        t_theta = 10000.0 * t_ntk
        seq = mx.arange(max(ot, oh, ow), dtype=dtype)
        h_freqs = 1.0 / (h_theta ** (mx.arange(0, dim_h, 2, dtype=dtype)[:dim_h // 2] / dim_h))
        w_freqs = 1.0 / (w_theta ** (mx.arange(0, dim_w, 2, dtype=dtype)[:dim_w // 2] / dim_w))
        t_freqs = 1.0 / (t_theta ** (mx.arange(0, dim_t, 2, dtype=dtype)[:dim_t // 2] / dim_t))
        emb_h = mx.outer(seq[:oh], h_freqs)[None, :, None, :]
        emb_h = mx.broadcast_to(emb_h, (ot, oh, ow, emb_h.shape[-1]))
        emb_w = mx.outer(seq[:ow], w_freqs)[None, None, :, :]
        emb_w = mx.broadcast_to(emb_w, (ot, oh, ow, emb_w.shape[-1]))
        if fps is not None:
            t_seq = seq[:ot] / fps * base_fps
        else:
            t_seq = seq[:ot]
        emb_t = mx.outer(t_seq, t_freqs)[:, None, None, :]
        emb_t = mx.broadcast_to(emb_t, (ot, oh, ow, emb_t.shape[-1]))
        freqs = mx.concatenate([emb_t, emb_h, emb_w] * 2, axis=-1)
        freqs_flat = freqs.reshape(-1, freqs.shape[-1])
        rope_cos = mx.cos(freqs_flat)
        rope_sin = mx.sin(freqs_flat)
        return rope_cos, rope_sin

    def __call__(self, x, timestep, text_emb, fps=None, padding_mask=None, condition_mask=None):
        B, C, T, H, W = x.shape
        if condition_mask is not None:
            x = mx.concatenate([x, condition_mask], axis=1)
        if self.concat_padding_mask and padding_mask is not None:
            pm = mx.broadcast_to(
                padding_mask[:, :, None, :, :],
                (B, 1, T, H, W),
            )
            x = mx.concatenate([x, pm], axis=1)
        h, ot, oh, ow = self.x_embedder(x)
        rope_cos, rope_sin = self._compute_rope(ot, oh, ow, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(ot, oh, ow, B)
        t_emb_input = self._timestep_embedding(timestep, dim=self.hidden_size)
        temb, embedded_t = self.t_embedder(t_emb_input)
        for i in range(self.num_layers):
            block = getattr(self.blocks, str(i))
            h = block(h, text_emb, embedded_t, temb, rope_cos, rope_sin, extra_pos_emb)
        h = self.final_layer(h, embedded_t, temb)
        h = self.final_linear(h)
        pt, ph, pw = self.patch_size
        h = h.reshape(B, ot, oh, ow, self.out_channels, pt, ph, pw)
        h = h.transpose(0, 4, 1, 5, 2, 6, 3, 7)
        h = h.reshape(B, self.out_channels, ot * pt, oh * ph, ow * pw)
        return h

    @classmethod
    def from_pretrained(cls, model_path, config=None, **kwargs):
        import glob
        import os

        dit = cls(config=config)
        if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
            safetensor_files = [model_path]
        else:
            safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not safetensor_files:
            logger.warning("cosmos dit: no safetensors at %s, random init", model_path)
            return dit
        all_params = {}
        for sf in safetensor_files:
            weights = mx.load(sf)
            all_params.update(weights)
        mapped = _remap_dit_weights(all_params, dit.hidden_size)
        model_params = dit.parameters()
        flat_model = _flatten_params(model_params)
        loaded = {}
        matched = 0
        shape_mismatches = []
        unmatched_model = []
        for k, v in flat_model.items():
            if k in mapped:
                w = mapped[k]
                if v.shape == w.shape:
                    loaded[k] = w.astype(v.dtype)
                    matched += 1
                else:
                    shape_mismatches.append((k, v.shape, w.shape))
                    loaded[k] = v
            else:
                loaded[k] = v
                unmatched_model.append(k)
        total_weight = len(mapped)
        logger.info(
            "cosmos dit: matched %d/%d model params from %d weight keys",
            matched, len(flat_model), total_weight,
        )
        if shape_mismatches:
            logger.warning(
                "cosmos dit: %d shape mismatches: %s",
                len(shape_mismatches),
                shape_mismatches[:10],
            )
        if unmatched_model:
            logger.debug("cosmos dit: unmatched model keys: %s", unmatched_model[:20])
        unmapped_weights = [k for k in mapped if k not in flat_model]
        if unmapped_weights:
            logger.debug("cosmos dit: unmapped weight keys: %s", unmapped_weights[:20])
        _update_module(dit, loaded)
        return dit


def _apply_rope(x, cos_f, sin_f, dim):
    if dim < x.shape[-1]:
        x_rot = x[..., :dim]
        x_pass = x[..., dim:]
    else:
        x_rot = x
        x_pass = None
    L = x_rot.shape[-2]
    cos_f = cos_f[:L]
    sin_f = sin_f[:L]
    x1 = x_rot[..., ::2]
    x2 = x_rot[..., 1::2]
    cos_1 = cos_f[..., ::2]
    sin_1 = sin_f[..., ::2]
    cos_2 = cos_f[..., 1::2]
    sin_2 = sin_f[..., 1::2]
    rotated = mx.concatenate(
        [x1 * cos_1 - x2 * sin_1, x1 * sin_2 + x2 * cos_2], axis=-1
    )
    if x_pass is not None:
        return mx.concatenate([rotated, x_pass], axis=-1)
    return rotated


def _flatten_params(params, prefix=""):
    out = {}
    if isinstance(params, dict):
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(_flatten_params(v, key))
    elif isinstance(params, mx.array):
        out[prefix] = params
    return out


def _update_module(module, flat_params):
    children = {}
    for k, v in flat_params.items():
        parts = k.split(".")
        if len(parts) == 1:
            if isinstance(v, mx.array):
                setattr(module, k, v)
            continue
        child_name = parts[0]
        rest = ".".join(parts[1:])
        if child_name not in children:
            children[child_name] = {}
        children[child_name][rest] = v
    for child_name, child_params in children.items():
        child = getattr(module, child_name, None)
        if child is not None and isinstance(child, nn.Module):
            _update_module(child, child_params)
        elif child is not None and isinstance(child, mx.array):
            if "." not in child_name and len(child_params) == 1 and "" in child_params:
                setattr(module, child_name, child_params[""])


def _remap_dit_weights(params, hidden_size):
    has_nvidia = any(k.startswith("net.") for k in params)
    has_diffusers = any(k.startswith("transformer_blocks.") for k in params)
    if has_nvidia:
        return _remap_nvidia(params, hidden_size)
    elif has_diffusers:
        return _remap_diffusers(params, hidden_size)
    return _remap_passthrough(params, hidden_size)


def _remap_nvidia(params, hidden_size):
    out = {}
    skip_prefixes = ("accum_",)
    for k, v in params.items():
        if any(k.startswith(f"net.{p}") for p in skip_prefixes):
            continue
        nk = k
        if nk.startswith("net."):
            nk = nk[4:]
        if "blocks." in nk:
            pass
        elif nk == "x_embedder.proj.1.weight":
            nk = "x_embedder.proj.weight"
        elif nk == "t_embedder.1.linear_1.weight":
            nk = "t_embedder.t_embedder.linear_1.weight"
        elif nk == "t_embedder.1.linear_2.weight":
            nk = "t_embedder.t_embedder.linear_2.weight"
        elif nk == "t_embedding_norm.weight":
            nk = "t_embedder.t_embedding_norm.weight"
        elif nk == "t_embedding_norm._extra_state":
            continue
        elif nk.startswith("final_layer."):
            if "adaln_modulation.1.weight" in nk:
                nk = "final_layer.linear_1.weight"
            elif "adaln_modulation.2.weight" in nk:
                nk = "final_layer.linear_2.weight"
            elif "linear.weight" in nk:
                nk = "final_linear.weight"
        elif nk.startswith("pos_embedder."):
            if nk == "pos_embedder.seq":
                continue
            elif nk == "pos_embedder.dim_spatial_range":
                continue
            elif nk == "pos_embedder.dim_temporal_range":
                continue
            elif "pos_emb_" in nk:
                nk = nk.replace("pos_embedder.", "learnable_pos_embed.")
        else:
            logger.debug("cosmos dit nvidia: unhandled key %s", k)
            continue
        if "blocks." in nk:
            nk = nk.replace("adaln_modulation_self_attn.1.weight",
                            "adaln_modulation_self_attn.linear_1.weight")
            nk = nk.replace("adaln_modulation_self_attn.2.weight",
                            "adaln_modulation_self_attn.linear_2.weight")
            nk = nk.replace("adaln_modulation_cross_attn.1.weight",
                            "adaln_modulation_cross_attn.linear_1.weight")
            nk = nk.replace("adaln_modulation_cross_attn.2.weight",
                            "adaln_modulation_cross_attn.linear_2.weight")
            nk = nk.replace("adaln_modulation_mlp.1.weight",
                            "adaln_modulation_mlp.linear_1.weight")
            nk = nk.replace("adaln_modulation_mlp.2.weight",
                            "adaln_modulation_mlp.linear_2.weight")
            nk = nk.replace("mlp.fc1.", "mlp.layer1.")
            nk = nk.replace("mlp.fc2.", "mlp.layer2.")
            nk = nk.replace("out_proj.", "output_proj.")
        if "_extra_state" in nk:
            continue
        out[nk] = v
    return out


def _remap_diffusers(params, hidden_size):
    out = {}
    for k, v in params.items():
        nk = k
        if nk.startswith("patch_embed.proj.weight"):
            nk = "x_embedder.proj.weight"
        elif nk.startswith("time_embed."):
            if "t_embedder.linear_1" in nk:
                nk = "t_embedder.t_embedder.linear_1.weight"
            elif "t_embedder.linear_2" in nk:
                nk = "t_embedder.t_embedder.linear_2.weight"
            elif "norm.weight" in nk:
                nk = "t_embedder.t_embedding_norm.weight"
        elif nk.startswith("norm_out."):
            if "linear_1" in nk:
                nk = "final_layer.linear_1.weight"
            elif "linear_2" in nk:
                nk = "final_layer.linear_2.weight"
        elif nk == "proj_out.weight":
            nk = "final_linear.weight"
        elif nk.startswith("learnable_pos_embed."):
            pass
        elif nk.startswith("transformer_blocks."):
            nk = _remap_block_diffusers(nk)
        else:
            logger.debug("cosmos dit diffusers: unhandled key %s", k)
            continue
        out[nk] = v
    return out


def _remap_block_diffusers(k):
    k = k.replace("transformer_blocks.", "blocks.")
    k = k.replace(".norm1.linear_1.", ".adaln_modulation_self_attn.linear_1.")
    k = k.replace(".norm1.linear_2.", ".adaln_modulation_self_attn.linear_2.")
    k = k.replace(".norm2.linear_1.", ".adaln_modulation_cross_attn.linear_1.")
    k = k.replace(".norm2.linear_2.", ".adaln_modulation_cross_attn.linear_2.")
    k = k.replace(".norm3.linear_1.", ".adaln_modulation_mlp.linear_1.")
    k = k.replace(".norm3.linear_2.", ".adaln_modulation_mlp.linear_2.")
    k = k.replace(".attn1.to_q.", ".self_attn.q_proj.")
    k = k.replace(".attn1.to_k.", ".self_attn.k_proj.")
    k = k.replace(".attn1.to_v.", ".self_attn.v_proj.")
    k = k.replace(".attn1.to_out.0.", ".self_attn.output_proj.")
    k = k.replace(".attn1.norm_q.", ".self_attn.q_norm.")
    k = k.replace(".attn1.norm_k.", ".self_attn.k_norm.")
    k = k.replace(".attn2.to_q.", ".cross_attn.q_proj.")
    k = k.replace(".attn2.to_k.", ".cross_attn.k_proj.")
    k = k.replace(".attn2.to_v.", ".cross_attn.v_proj.")
    k = k.replace(".attn2.to_out.0.", ".cross_attn.output_proj.")
    k = k.replace(".attn2.norm_q.", ".cross_attn.q_norm.")
    k = k.replace(".attn2.norm_k.", ".cross_attn.k_norm.")
    k = k.replace(".ff.net.0.proj.", ".mlp.layer1.")
    k = k.replace(".ff.net.2.", ".mlp.layer2.")
    return k


def _remap_passthrough(params, hidden_size):
    out = {}
    skip = ("_extra_state", "accum_")
    for k, v in params.items():
        if any(s in k for s in skip):
            continue
        out[k] = v
    return out
