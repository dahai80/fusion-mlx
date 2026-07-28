# SPDX-License-Identifier: Apache-2.0
import logging
import math

import mlx.core as mx
import mlx.nn as nn

from .config import CogVideoXConfig
from .rope import apply_rope, compute_3d_rope

logger = logging.getLogger(__name__)


def _sinusoidal_embedding(
    timesteps: mx.array, dim: int, flip_sin_to_cos: bool = True, freq_shift: int = 0
) -> mx.array:
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - freq_shift)
    emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
    emb = timesteps.astype(mx.float32)[..., None] * emb[None, :]
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if dim % 2 == 1:
        emb = mx.concatenate([emb, mx.zeros_like(emb[..., :1])], axis=-1)
    return emb


class CogVideoXPatchEmbed(nn.Module):
    def __init__(self, config: CogVideoXConfig):
        super().__init__()
        self.config = config
        p = config.patch_size
        pt = config.patch_size_t
        inner_dim = config.inner_dim

        if pt is not None:
            self.proj = nn.Linear(
                pt * p * p * config.in_channels, inner_dim, bias=config.patch_bias
            )
        else:
            self.proj = nn.Conv2d(
                config.in_channels,
                inner_dim,
                kernel_size=(p, p),
                stride=(p, p),
                bias=config.patch_bias,
            )

        self.text_proj = nn.Linear(config.text_embed_dim, inner_dim)

        self.use_positional_embeddings = (
            config.use_rotary_positional_embeddings is False
        )
        self.use_learned_positional_embeddings = (
            config.use_learned_positional_embeddings
        )

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            text_len = config.max_text_seq_length
            video_len = (
                (config.sample_frames // (pt if pt else 1))
                * (config.sample_height // p)
                * (config.sample_width // p)
            )
            total_len = text_len + video_len
            self.pos_embedding = mx.zeros((1, total_len, inner_dim))

    def __call__(self, text_embeds: mx.array, image_embeds: mx.array) -> mx.array:
        b, num_frames, c, h, w = image_embeds.shape
        p = self.config.patch_size
        pt = self.config.patch_size_t
        inner_dim = self.config.inner_dim

        text_embeds = self.text_proj(text_embeds)

        if pt is not None:
            image_embeds = image_embeds.transpose(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                b, num_frames // pt, pt, h // p, p, w // p, p, c
            )
            image_embeds = image_embeds.transpose(0, 1, 3, 5, 7, 2, 4, 6)
            image_embeds = image_embeds.reshape(
                b, (num_frames // pt) * (h // p) * (w // p), pt * p * p * c
            )
            image_embeds = self.proj(image_embeds)
        else:
            # MLX conv2d expects NHWC: (N, H, W, C)
            image_embeds = image_embeds.transpose(0, 1, 3, 4, 2)  # (b, f, h, w, c)
            image_embeds = image_embeds.reshape(b * num_frames, h, w, c)
            image_embeds = self.proj(image_embeds)  # (b*f, oh, ow, oc)
            oh, ow = image_embeds.shape[1], image_embeds.shape[2]
            image_embeds = image_embeds.reshape(b, num_frames, oh, ow, inner_dim)
            image_embeds = image_embeds.reshape(b, num_frames * oh * ow, inner_dim)

        embeds = mx.concatenate([text_embeds, image_embeds], axis=1)

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if (
                hasattr(self, "pos_embedding")
                and self.pos_embedding.shape[1] == embeds.shape[1]
            ):
                embeds = embeds + self.pos_embedding

        return embeds


class CogVideoXLayerNormZero(nn.Module):
    def __init__(
        self,
        time_embed_dim: int,
        dim: int,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.lin = nn.Linear(time_embed_dim, 6 * dim, bias=bias)
        self.norm = nn.LayerNorm(dim, eps=norm_eps, affine=norm_elementwise_affine)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array, temb: mx.array
    ) -> tuple:
        temb = self.lin(nn.silu(temb))
        chunks = mx.split(temb, 6, axis=-1)
        shift_msa, scale_msa, gate_msa, shift_enc, scale_enc, gate_enc = [
            c.squeeze(axis=-1) if c.shape[-1] == 1 else c for c in chunks
        ]

        norm_hidden = (
            self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        )
        norm_encoder = (
            self.norm(encoder_hidden_states) * (1 + scale_enc[:, None])
            + shift_enc[:, None]
        )

        return norm_hidden, norm_encoder, gate_msa, gate_enc


class CogVideoXAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        bias: bool = True,
        out_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=out_bias)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        rope_cos: mx.array | None = None,
        rope_sin: mx.array | None = None,
    ) -> tuple:
        b = hidden_states.shape[0]
        text_len = encoder_hidden_states.shape[1]

        # Concatenate text + video, compute qkv from combined
        combined = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
        q = self.q_proj(combined)
        k = self.k_proj(combined)
        v = self.v_proj(combined)

        q = q.reshape(b, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if rope_cos is not None and rope_sin is not None:
            rope_cos_exp = rope_cos[None, None, :, :]
            rope_sin_exp = rope_sin[None, None, :, :]
            q_video = apply_rope(q[:, :, text_len:, :], rope_cos_exp, rope_sin_exp)
            q = mx.concatenate([q[:, :, :text_len, :], q_video], axis=2)

            k_video = apply_rope(k[:, :, text_len:, :], rope_cos_exp, rope_sin_exp)
            k = mx.concatenate([k[:, :, :text_len, :], k_video], axis=2)

        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = (
            (attn @ v)
            .transpose(0, 2, 1, 3)
            .reshape(b, -1, self.num_heads * self.head_dim)
        )

        out = self.out_proj(out)
        out_text = out[:, :text_len]
        out_video = out[:, text_len:]
        return out_video, out_text


class CogVideoXFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int,
        activation_fn: str = "gelu-approximate",
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = [
            nn.Linear(dim, inner_dim, bias=bias),
            nn.GELU(approximate="tanh") if "tanh" in activation_fn else nn.GELU(),
            nn.Linear(inner_dim, dim, bias=bias),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.net:
            x = layer(x)
        return x


class CogVideoXBlock(nn.Module):
    def __init__(self, config: CogVideoXConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.norm1 = CogVideoXLayerNormZero(
            config.time_embed_dim,
            config.inner_dim,
            config.norm_elementwise_affine,
            config.norm_eps,
            bias=True,
        )
        self.attn1 = CogVideoXAttention(
            config.inner_dim,
            config.num_attention_heads,
            config.attention_head_dim,
            bias=config.attention_bias,
            out_bias=True,
        )
        self.norm2 = CogVideoXLayerNormZero(
            config.time_embed_dim,
            config.inner_dim,
            config.norm_elementwise_affine,
            config.norm_eps,
            bias=True,
        )
        self.ff = CogVideoXFeedForward(
            config.inner_dim,
            config.ff_inner_dim,
            config.activation_fn,
            bias=True,
            dropout=config.dropout,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        rope_cos: mx.array | None = None,
        rope_sin: mx.array | None = None,
    ) -> tuple:
        text_len = encoder_hidden_states.shape[1]

        norm_hidden, norm_encoder, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )
        attn_video, attn_text = self.attn1(
            norm_hidden, norm_encoder, rope_cos, rope_sin
        )
        hidden_states = hidden_states + gate_msa[:, None] * attn_video
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_msa[:, None] * attn_text
        )

        norm_hidden, norm_encoder, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )
        ff_input = mx.concatenate([norm_encoder, norm_hidden], axis=1)
        ff_output = self.ff(ff_input)
        hidden_states = hidden_states + gate_ff[:, None] * ff_output[:, text_len:]
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_ff[:, None] * ff_output[:, :text_len]
        )

        return hidden_states, encoder_hidden_states


class CogVideoXTransformer3DModel(nn.Module):
    def __init__(self, config: CogVideoXConfig):
        super().__init__()
        self.config = config

        self.patch_embed = CogVideoXPatchEmbed(config)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.time_proj = lambda t: _sinusoidal_embedding(
            t, config.inner_dim, config.flip_sin_to_cos, config.freq_shift
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.inner_dim, config.time_embed_dim),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, config.time_embed_dim),
        )

        self.transformer_blocks = [
            CogVideoXBlock(config, layer_idx=i) for i in range(config.num_layers)
        ]

        self.norm_final = nn.LayerNorm(
            config.inner_dim, eps=config.norm_eps, affine=config.norm_elementwise_affine
        )
        self.norm_out = CogVideoXLayerNormZero(
            config.time_embed_dim,
            config.inner_dim,
            config.norm_elementwise_affine,
            config.norm_eps,
            bias=True,
        )
        self.proj_out = nn.Linear(
            config.inner_dim,
            config.out_channels * config.patch_size * config.patch_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        rope_cos: mx.array | None = None,
        rope_sin: mx.array | None = None,
    ) -> mx.array:
        b, f, c, h, w = hidden_states.shape
        p = self.config.patch_size
        pt = self.config.patch_size_t

        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb)

        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_len = self.config.max_text_seq_length
        encoder_hidden_states = hidden_states[:, :text_len]
        hidden_states = hidden_states[:, text_len:]

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, emb, rope_cos, rope_sin
            )

        hidden_states = self.norm_final(hidden_states)
        norm_hidden, _, gate, _ = self.norm_out(
            hidden_states, encoder_hidden_states, emb
        )
        hidden_states = gate[:, None] * norm_hidden
        hidden_states = self.proj_out(hidden_states)

        hf = h // p
        wf = w // p
        ff = f // pt if pt else f
        out_ch = self.config.out_channels

        hidden_states = hidden_states.reshape(b, ff, hf, wf, out_ch, p, p)
        hidden_states = hidden_states.transpose(0, 1, 4, 2, 5, 3, 6)
        output = hidden_states.reshape(b, f, out_ch, h, w)

        return output

    def _precompute_rope(
        self, num_frames: int, height: int, width: int
    ) -> tuple[mx.array, mx.array]:
        p = self.config.patch_size
        pt = self.config.patch_size_t if self.config.patch_size_t is not None else 1
        return compute_3d_rope(
            num_frames // pt,
            height // p,
            width // p,
            patch_size=1,
            head_dim=self.config.attention_head_dim,
            spatial_interpolation_scale=self.config.spatial_interpolation_scale,
            temporal_interpolation_scale=self.config.temporal_interpolation_scale,
        )
