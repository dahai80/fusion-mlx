# SPDX-License-Identifier: Apache-2.0
#
# Pure-MLX port of the LTX-Video 0.9.x Transformer3DModel.
#
# This is a direct port of ltx_video/models/transformers/transformer3d.py and
# ltx_video/models/transformers/attention.py (LTX-Video 0.9.x, MIT licensed),
# plus the diffusers helper math they lean on: AdaLayerNormSingle,
# PixArtAlphaCombinedTimestepSizeEmbeddings, PixArtAlphaTextProjection,
# Timesteps, TimestepEmbedding, RMSNorm, and the tanh-approximate GELU.
#
# No torch / diffusers / einops dependency. RoPE, RMSNorm, adaLN-single and
# scaled dot-product attention are reimplemented on mlx.core + mlx.nn only.
#
# Checkpoints are loaded from diffusers-format 0.9.x weights by applying the
# reference TRANSFORMER_KEYS_RENAME_DICT (proj_in->patchify_proj,
# time_embed->adaln_single, norm_q->q_norm, norm_k->k_norm) plus the small set
# of ltx_video-native sub-module names (to_out.0->out_proj, ff.net.0.proj->fc1,
# ff.net.2->fc2) so the same loader accepts both ltx_video and diffusers shards.

import glob
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

logger = logging.getLogger(__name__)

# diffusers -> ltx_video top-level key renames (TRANSFORMER_KEYS_RENAME_DICT).
_TRANSFORMER_KEYS_RENAME = {
    "proj_in": "patchify_proj",
    "time_embed": "adaln_single",
    "norm_q": "q_norm",
    "norm_k": "k_norm",
}

# ltx_video-native sub-module names that differ from this port's internal names.
_SUBKEY_RENAME = [
    (".to_out.0.", ".out_proj."),
    (".ff.net.0.proj.", ".ff.fc1."),
    (".ff.net.2.", ".ff.fc2."),
]

_MASK_NEG = -1e4


def _gelu_tanh(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + mx.tanh(c * (x + 0.044715 * x * x * x)))


def get_timestep_embedding(
    timesteps, dim, flip_sin_to_cos=True, downscale_freq_shift=0.0, max_period=10000
):
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32)
    exponent = exponent / (half - downscale_freq_shift)
    freqs = mx.exp(exponent)
    t = timesteps.astype(mx.float32).reshape(-1)
    emb = t[:, None] * freqs[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half:], emb[:, :half]], axis=-1)
    if dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


def precompute_freqs_cis(indices_grid, inner_dim, theta, max_pos):
    # indices_grid: [B, 3, N] (frame_idx, h_idx, w_idx) already scaled by the
    # caller (temporal axis divided by frame_rate). Returns (cos, sin) each
    # [B, N, inner_dim], matching ltx_video precompute_freqs_cis(spacing="exp").
    b = indices_grid.shape[0]
    n = indices_grid.shape[2]
    dim = inner_dim
    d6 = dim // 6
    fp = mx.stack([indices_grid[:, i] / float(max_pos[i]) for i in range(3)], axis=-1)
    fp = fp.astype(mx.float32)
    lin = mx.linspace(0.0, 1.0, d6, dtype=mx.float32)
    indices = mx.power(mx.array(theta, dtype=mx.float32), lin)
    indices = indices * (math.pi / 2.0)
    frac = fp[..., None] * 2.0 - 1.0
    freqs = indices[None, None, None, :] * frac
    freqs = mx.transpose(freqs, (0, 1, 3, 2)).reshape(b, n, d6 * 3)
    cos_freq = mx.repeat(mx.cos(freqs), 2, axis=-1)
    sin_freq = mx.repeat(mx.sin(freqs), 2, axis=-1)
    rem = dim % 6
    if rem != 0:
        cos_pad = mx.ones((b, n, rem), dtype=mx.float32)
        sin_pad = mx.zeros((b, n, rem), dtype=mx.float32)
        cos_freq = mx.concatenate([cos_pad, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_pad, sin_freq], axis=-1)
    return cos_freq, sin_freq


def apply_rotary_emb(x, cos, sin):
    # x: [..., D], cos/sin: [..., D] with each freq duplicated (interleaved).
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos1 = cos[..., 0::2]
    sin1 = sin[..., 0::2]
    out0 = x1 * cos1 - x2 * sin1
    out1 = x2 * cos1 + x1 * sin1
    return mx.stack([out0, out1], axis=-1).reshape(x.shape)


def _to_bias(mask):
    if mask is None:
        return None
    bias = (1.0 - mask.astype(mx.float32)) * _MASK_NEG
    return bias[:, None, None, :]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps, affine=True):
        super().__init__()
        self.eps = eps
        self._affine = affine
        if affine:
            self.weight = mx.ones((dim,))

    def __call__(self, x):
        xf = x.astype(mx.float32)
        var = mx.mean(xf * xf, axis=-1, keepdims=True)
        xf = xf * mx.rsqrt(var + self.eps)
        if self._affine:
            xf = self.weight * xf
        return xf.astype(x.dtype)


class _GELUProj(nn.Module):
    # diffusers GELU(approximate="tanh"): Linear then tanh-gelu. Disk key: net.0.proj
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def __call__(self, x):
        return _gelu_tanh(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, dim, inner_dim=None, bias=True):
        super().__init__()
        inner_dim = inner_dim if inner_dim is not None else dim * 4
        self.fc1 = _GELUProj(dim, inner_dim, bias=bias)
        self.fc2 = nn.Linear(inner_dim, dim, bias=bias)

    def __call__(self, x):
        return self.fc2(self.fc1(x))


class TimestepEmbedder(nn.Module):
    # diffusers TimestepEmbedding: linear_1 -> silu -> linear_2.
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def __call__(self, x):
        return self.linear_2(nn.silu(self.linear_1(x)))


class PixArtTimestepEmb(nn.Module):
    # diffusers PixArtAlphaCombinedTimestepSizeEmbeddings with
    # use_additional_conditions=False: only time_proj(256) + timestep_embedder.
    def __init__(self, embedding_dim):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder(256, embedding_dim)

    def __call__(self, timestep):
        t_proj = get_timestep_embedding(
            timestep, 256, flip_sin_to_cos=True, downscale_freq_shift=0.0
        )
        return self.timestep_embedder(t_proj)


class AdaLayerNormSingle(nn.Module):
    # diffusers AdaLayerNormSingle: emb -> silu -> linear(6*dim). Returns
    # (modulation [B,6*dim], embedded_timestep [B,dim]).
    def __init__(self, embedding_dim):
        super().__init__()
        self.emb = PixArtTimestepEmb(embedding_dim)
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)

    def __call__(self, timestep):
        embedded = self.emb(timestep)
        return self.linear(nn.silu(embedded)), embedded


class CaptionProjection(nn.Module):
    # diffusers PixArtAlphaTextProjection: linear_1 -> gelu(tanh) -> linear_2.
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, caption):
        return self.linear_2(_gelu_tanh(self.linear_1(caption)))


class LTXAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        cross_attention_dim,
        heads,
        dim_head,
        bias=True,
        qk_norm=True,
        use_rope=False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head**-0.5
        self.use_rope = use_rope
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.q_norm = (
            RMSNorm(self.inner_dim, eps=1e-5, affine=True) if qk_norm else None
        )
        self.k_norm = (
            RMSNorm(self.inner_dim, eps=1e-5, affine=True) if qk_norm else None
        )
        self.out_proj = nn.Linear(self.inner_dim, query_dim, bias=True)

    def __call__(
        self,
        hidden_states,
        cos=None,
        sin=None,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        b = hidden_states.shape[0]
        is_self = encoder_hidden_states is None
        ctx = hidden_states if is_self else encoder_hidden_states

        q = self.to_q(hidden_states)
        if self.q_norm is not None:
            q = self.q_norm(q)
        k = self.to_k(ctx)
        v = self.to_v(ctx)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if is_self and self.use_rope and cos is not None:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        sq = q.shape[1]
        sk = k.shape[1]
        q = q.reshape(b, sq, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(b, sk, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, sk, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attention_mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, sq, self.inner_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    # ltx_video BasicTransformerBlock: norm1 -> self-attn(+ada gate) -> norm2 ->
    # cross-attn(plain residual) -> norm2 -> ff(+ada gate). Pre-norms are
    # RMSNorm(affine=False); ada modulation comes from scale_shift_table +
    # timestep via AdaLN-single.
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        cross_attention_dim,
        eps,
        qk_norm=True,
        use_rope=False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps, affine=False)
        self.attn1 = LTXAttention(
            dim, dim, heads, dim_head, bias=True, qk_norm=qk_norm, use_rope=use_rope
        )
        self.norm2 = RMSNorm(dim, eps, affine=False)
        self.attn2 = LTXAttention(
            dim,
            cross_attention_dim,
            heads,
            dim_head,
            bias=True,
            qk_norm=qk_norm,
            use_rope=use_rope,
        )
        self.ff = FeedForward(dim, inner_dim=dim * 4)
        self.scale_shift_table = mx.zeros((6, dim))

    def __call__(
        self, x, cos, sin, t_mod, encoder_hidden_states, self_mask, cross_mask
    ):
        b = x.shape[0]
        d = x.shape[2]
        h = self.norm1(x)
        ada = self.scale_shift_table[None, None] + t_mod.reshape(b, 1, 6, d)
        shift_msa = ada[..., 0, :]
        scale_msa = ada[..., 1, :]
        gate_msa = ada[..., 2, :]
        shift_mlp = ada[..., 3, :]
        scale_mlp = ada[..., 4, :]
        gate_mlp = ada[..., 5, :]
        h = h * (1.0 + scale_msa) + shift_msa
        a = self.attn1(
            h, cos=cos, sin=sin, encoder_hidden_states=None, attention_mask=self_mask
        )
        x = x + gate_msa * a
        a = self.attn2(
            x, encoder_hidden_states=encoder_hidden_states, attention_mask=cross_mask
        )
        x = x + a
        h = self.norm2(x)
        h = h * (1.0 + scale_mlp) + shift_mlp
        f = self.ff(h)
        x = x + gate_mlp * f
        return x


@dataclass
class Transformer3DConfig:
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 28
    cross_attention_dim: int = 2048
    caption_channels: int = 4096
    attention_bias: bool = True
    norm_eps: float = 1e-6
    qk_norm: str = "rms_norm"
    standardization_norm: str = "rms_norm"
    positional_embedding_type: str = "rope"
    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: tuple = (20, 2048, 2048)
    timestep_scale_multiplier: float = 1000.0
    activation_fn: str = "gelu-approximate"

    @property
    def inner_dim(self):
        return self.num_attention_heads * self.attention_head_dim

    @classmethod
    def from_hf_config(cls, cfg):
        c = dict(cfg)
        get = lambda k, d: c.get(k, d)
        out = cls(
            num_attention_heads=int(get("num_attention_heads", 32)),
            attention_head_dim=int(get("attention_head_dim", 64)),
            in_channels=int(get("in_channels", 128)),
            out_channels=int(get("out_channels", 128)),
            num_layers=int(get("num_layers", 28)),
            cross_attention_dim=int(get("cross_attention_dim", 2048)),
            caption_channels=int(get("caption_channels", 4096)),
            attention_bias=bool(get("attention_bias", True)),
            norm_eps=float(get("norm_eps", 1e-6)),
            qk_norm=str(get("qk_norm", "rms_norm")),
            standardization_norm=str(get("standardization_norm", "rms_norm")),
            positional_embedding_type=str(get("positional_embedding_type", "rope")),
            positional_embedding_theta=float(
                get("positional_embedding_theta", 10000.0)
            ),
            positional_embedding_max_pos=tuple(
                get("positional_embedding_max_pos", [20, 2048, 2048])
            ),
            timestep_scale_multiplier=float(get("timestep_scale_multiplier", 1000.0)),
            activation_fn=str(get("activation_fn", "gelu-approximate")),
        )
        return out


class Transformer3DModel(nn.Module):
    def __init__(self, config: Transformer3DConfig):
        super().__init__()
        self.cfg = config
        d = config.inner_dim
        self.patchify_proj = nn.Linear(config.in_channels, d, bias=True)
        self.transformer_blocks = [
            TransformerBlock(
                dim=d,
                heads=config.num_attention_heads,
                dim_head=config.attention_head_dim,
                cross_attention_dim=config.cross_attention_dim,
                eps=config.norm_eps,
                qk_norm=(config.qk_norm == "rms_norm"),
                use_rope=(config.positional_embedding_type == "rope"),
            )
            for _ in range(config.num_layers)
        ]
        self.norm_out = nn.LayerNorm(d, affine=False, eps=config.norm_eps)
        self.scale_shift_table = mx.zeros((2, d))
        self.proj_out = nn.Linear(d, config.out_channels, bias=True)
        self.adaln_single = AdaLayerNormSingle(d)
        self.caption_projection = (
            CaptionProjection(config.caption_channels, d)
            if config.caption_channels
            else None
        )
        self.inner_dim = d
        self.theta = config.positional_embedding_theta
        self.max_pos = tuple(config.positional_embedding_max_pos)
        self.timestep_scale_multiplier = config.timestep_scale_multiplier

    def __call__(
        self,
        hidden_states,
        indices_grid,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        encoder_attention_mask=None,
    ):
        t0 = time.time()
        b = hidden_states.shape[0]
        n = hidden_states.shape[1]
        h = self.patchify_proj(hidden_states)

        if self.timestep_scale_multiplier:
            timestep = (
                timestep.reshape(-1).astype(mx.float32) * self.timestep_scale_multiplier
            )
        cos, sin = precompute_freqs_cis(
            indices_grid, self.inner_dim, self.theta, self.max_pos
        )

        t_mod, embedded_t = self.adaln_single(timestep)
        t_mod = t_mod.reshape(b, 1, 6, self.inner_dim)
        embedded_t = embedded_t.reshape(b, 1, self.inner_dim)

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)

        self_mask = _to_bias(attention_mask)
        cross_mask = _to_bias(encoder_attention_mask)

        for block in self.transformer_blocks:
            h = block(h, cos, sin, t_mod, encoder_hidden_states, self_mask, cross_mask)

        sst = self.scale_shift_table[None, None] + embedded_t[:, :, None]
        shift = sst[..., 0, :]
        scale = sst[..., 1, :]
        h = self.norm_out(h)
        h = h * (1.0 + scale) + shift
        h = self.proj_out(h)

        out_shape = tuple(h.shape)
        logger.info(
            "transformer: fwd batch=%d tokens=%d out_channels=%d dt=%.3fs",
            b,
            n,
            self.cfg.out_channels,
            time.time() - t0,
        )
        return h

    @classmethod
    def from_pretrained(cls, model_path, dtype=mx.float32) -> "Transformer3DModel":
        t0 = time.time()
        model_path = Path(model_path)
        cfg_file = model_path / "config.json"
        if not cfg_file.exists():
            cfg_file = model_path / "transformer" / "config.json"
        cfg = json.loads(cfg_file.read_text())
        config = Transformer3DConfig.from_hf_config(cfg)
        model = cls(config)

        shards = sorted(
            glob.glob(str(model_path / "diffusion_pytorch_model*.safetensors"))
        )
        if not shards:
            shards = sorted(
                glob.glob(
                    str(
                        model_path
                        / "transformer"
                        / "diffusion_pytorch_model*.safetensors"
                    )
                )
            )
        if not shards:
            raise FileNotFoundError(f"transformer: no safetensors in {model_path}")

        logger.info(
            "transformer: load path=%s layers=%d heads=%d dim=%d dtype=%s shards=%d",
            model_path.name,
            config.num_layers,
            config.num_attention_heads,
            config.inner_dim,
            dtype,
            len(shards),
        )
        raw = {}
        for shard in shards:
            with safe_open(shard, framework="numpy") as f:
                for k in f.keys():
                    raw[k] = f.get_tensor(k)

        mapped = _map_transformer_weights(raw)
        pairs = [(k, mx.array(v).astype(dtype)) for k, v in mapped.items()]
        n_params = sum(int(p[1].size) for p in pairs)
        model.load_weights(pairs, strict=False)
        mx.eval(model.parameters())
        missing, unexpected = _audit_weights(model, mapped)
        logger.info(
            "transformer: ready params=%.2fB mapped=%d missing=%d unexpected=%d dt=%.2fs",
            n_params / 1e9,
            len(mapped),
            missing,
            unexpected,
            time.time() - t0,
        )
        if missing:
            logger.warning(
                "transformer: missing %d weight tensors (init defaults)", missing
            )
        return model


def _map_transformer_weights(raw):
    out = {}
    for k, v in raw.items():
        nk = k
        if nk.startswith("transformer."):
            nk = nk[len("transformer.") :]
        for src, dst in _TRANSFORMER_KEYS_RENAME.items():
            if nk == src or nk.startswith(src + "."):
                nk = dst + nk[len(src) :]
                break
        for src, dst in _SUBKEY_RENAME:
            if src in nk:
                nk = nk.replace(src, dst)
        out[nk] = v
    return out


def _audit_weights(model, mapped):
    model_keys = set(_flatten_keys(model.parameters()))
    mapped_keys = set(mapped.keys())
    missing = len(model_keys - mapped_keys)
    unexpected = len(mapped_keys - model_keys)
    if unexpected:
        sample = sorted(mapped_keys - model_keys)[:5]
        logger.warning("transformer: unexpected keys (first 5): %s", sample)
    return missing, unexpected


def _flatten_keys(tree, prefix=""):
    keys = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            keys.extend(_flatten_keys(v, prefix + k + "."))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            keys.extend(_flatten_keys(v, prefix + str(i) + "."))
    else:
        if prefix:
            keys.append(prefix[:-1])
    return keys
