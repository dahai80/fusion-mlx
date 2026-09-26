# SPDX-License-Identifier: Apache-2.0
# ACE-Step MLX DiT model (issue #988): patch embed, timestep embed,
# condition embedder, 24 DiT layers, output AdaLN + de-patchify.
# Ported from ACE-Step/Ace-Step1.5 modeling_acestep_v15_turbo.py (MIT).
# No torch deps — pure mlx.nn.
from __future__ import annotations

import logging
import math

import mlx.core as mx
import mlx.nn as nn

from .blocks import DiTLayer, create_4d_mask, rope_freqs
from .config import AceStepConfig

logger = logging.getLogger(__name__)


class TimestepEmbedding(nn.Module):
    # Sinusoidal timestep embedding + 2-layer MLP. Returns (temb, timestep_proj).
    # temb: (B, time_embed_dim). timestep_proj: (B, 6, time_embed_dim) for AdaLN
    # modulation (6 chunks: shift/scale/gate x2 for self-attn + MLP).
    def __init__(self, in_channels: int, time_embed_dim: int, scale: float = 1000.0):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6, bias=True)
        self.in_channels = in_channels
        self.scale = scale

    def timestep_embedding(
        self, t: mx.array, dim: int, max_period: float = 10000.0
    ) -> mx.array:
        t = t.astype(mx.float32) * self.scale
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None] * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate(
                [emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1
            )
        return emb

    def __call__(self, t: mx.array) -> tuple[mx.array, mx.array]:
        t_freq = self.timestep_embedding(t, self.in_channels)
        temb = self.linear_1(t_freq)
        temb = nn.silu(temb)
        temb = self.linear_2(temb)
        timestep_proj = self.time_proj(nn.silu(temb))
        timestep_proj = mx.reshape(timestep_proj, (timestep_proj.shape[0], 6, -1))
        return temb, timestep_proj


class AceStepDiTModel(nn.Module):
    # Music DiT: patch-embed (Conv1d stride=patch_size) -> 24 DiT layers
    # (self-attn + cross-attn + MLP, AdaLN-modulated) -> output AdaLN + de-patchify.
    # Bidirectional (non-causal) attention: full (global) or sliding-window (local).
    # Conditioned on text+lyric+timbre encoder hidden states via cross-attn.
    def __init__(self, config: AceStepConfig):
        super().__init__()
        self.config = config
        in_channels = config.in_channels
        inner_dim = config.hidden_size
        patch_size = config.patch_size
        self.patch_size = patch_size
        self.inner_dim = inner_dim

        # patch embedding: Conv1d over time, kernel=stride=patch_size
        self.proj_in_weight = mx.zeros((inner_dim, in_channels, patch_size))
        self.proj_in_bias = mx.zeros((inner_dim,))
        # de-patchify: ConvTranspose1d
        # ConvTranspose1d weight: (in_channels=inner_dim, out_channels=audio_dim, K)
        self.proj_out_weight = mx.zeros(
            (inner_dim, config.audio_acoustic_hidden_dim, patch_size)
        )
        self.proj_out_bias = mx.zeros((config.audio_acoustic_hidden_dim,))

        self.layers = [
            DiTLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ]

        self.time_embed = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
        self.time_embed_r = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
        self.condition_embedder = nn.Linear(inner_dim, inner_dim, bias=True)

        self.norm_out = nn.RMSNorm(inner_dim, eps=config.rms_norm_eps)
        # output AdaLN: 2 chunks (shift, scale)
        self.scale_shift_table = mx.random.normal((1, 2, inner_dim)) / (inner_dim**0.5)

    def _conv1d(self, x: mx.array, w: mx.array, b: mx.array, stride: int) -> mx.array:
        # x: (B, C_in, T). w: (C_out, C_in, K). im2col conv1d.
        B, Cin, T = x.shape
        Cout, _, K = w.shape
        if T < K:
            return mx.zeros((B, Cout, 0), dtype=x.dtype) + b[:, None]
        out_T = (T - K) // stride + 1
        idxs = mx.arange(K)[:, None] + mx.arange(out_T)[None, :] * stride  # (K, out_T)
        patches = x[:, :, idxs]  # (B, Cin, K, out_T)
        w_flat = mx.reshape(w, (Cout, Cin * K))
        patches_flat = mx.reshape(mx.swapaxes(patches, 1, 2), (B, out_T, Cin * K))
        out = mx.matmul(patches_flat, w_flat.T)  # (B, out_T, Cout)
        out = mx.swapaxes(out, 1, 2)  # (B, Cout, out_T)
        out = out + b[:, None]
        return out

    def _conv_transpose1d(
        self, x: mx.array, w: mx.array, b: mx.array, stride: int
    ) -> mx.array:
        # x: (B, Cin, T). w: (Cin, Cout, K) (ConvTranspose1d layout). stride=K.
        B, Cin, T = x.shape
        _, Cout, K = w.shape
        out_T = T * stride
        # y[b, c_out, t*K+k] = sum_cin x[b,cin,t]*w[cin,c_out,k]; w shape (Cin, Cout, K)
        out = mx.einsum("bct,cok->bokt", x, w)  # (B, Cout, K, T)
        out = mx.reshape(out, (B, Cout, out_T))
        out = out + b[:, None]
        return out

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        timestep_r: mx.array,
        attention_mask: mx.array | None,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: mx.array | None,
        context_latents: mx.array,
    ) -> mx.array:
        # timestep: (B,), timestep_r: (B,)
        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r  # (B, 6, D)

        # concat context latents (source latents + chunk masks) with hidden along feat
        hidden_states = mx.concatenate([context_latents, hidden_states], axis=-1)
        original_seq_len = hidden_states.shape[1]
        # pad seq to multiple of patch_size
        pad_length = 0
        if original_seq_len % self.patch_size != 0:
            pad_length = self.patch_size - (original_seq_len % self.patch_size)
            hidden_states = mx.pad(
                hidden_states,
                [(0, 0), (0, pad_length), (0, 0)],
                constant_values=0.0,
            )

        # patch embed: (B, T, C) -> (B, C, T) -> conv1d -> (B, inner_dim, T//p) -> (B, T//p, inner_dim)
        x = mx.swapaxes(hidden_states, 1, 2)
        x = self._conv1d(x, self.proj_in_weight, self.proj_in_bias, self.patch_size)
        hidden_states = mx.swapaxes(x, 1, 2)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        seq_len = hidden_states.shape[1]
        cos, sin = rope_freqs(self.config.head_dim, seq_len, self.config.rope_theta)

        # masks: full (global, bidirectional) for full_attention layers,
        # sliding (local, |i-j|<=W) for sliding_attention layers.
        # cross-attn mask: (B, 1, seq_len, enc_seq_len).
        full_mask = create_4d_mask(
            seq_len,
            hidden_states.dtype,
            attention_mask=attention_mask,
            sliding_window=None,
        )
        sliding_mask = None
        if self.config.use_sliding_window:
            sliding_mask = create_4d_mask(
                seq_len,
                hidden_states.dtype,
                attention_mask=attention_mask,
                sliding_window=self.config.sliding_window,
            )
        enc_seq_len = encoder_hidden_states.shape[1]
        max_len = max(seq_len, enc_seq_len)
        enc_mask_4d = create_4d_mask(
            max_len, hidden_states.dtype, attention_mask=None, sliding_window=None
        )
        enc_mask_4d = enc_mask_4d[:, :, :seq_len, :enc_seq_len]
        if encoder_attention_mask is not None:
            # (B, enc_seq) -> (B, 1, 1, enc_seq) additive
            em = encoder_attention_mask.astype(hidden_states.dtype)
            em = mx.reshape(em, (em.shape[0], 1, 1, em.shape[-1]))
            neg = mx.log(mx.zeros((), dtype=hidden_states.dtype))
            enc_pad = mx.where(em > 0, mx.zeros_like(enc_mask_4d), neg)
            enc_mask_4d = enc_mask_4d + enc_pad

        for layer in self.layers:
            at = layer.attention_type
            self_mask = (
                sliding_mask
                if at == "sliding_attention" and sliding_mask is not None
                else full_mask
            )
            hidden_states = layer(
                hidden_states,
                position_embeddings=(cos, sin),
                temb=timestep_proj,
                self_mask=self_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_mask=enc_mask_4d,
            )

        # output AdaLN: (1, 2, D) + temb(B, D)->(B, 1, D) broadcast over 2 chunks
        mod = self.scale_shift_table + temb[:, None, :]
        shift, scale = [mx.squeeze(c, axis=1) for c in mx.split(mod, 2, axis=1)]
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None, :]) + shift[:, None, :]
        )
        # de-patchify: (B, T//p, inner_dim) -> (B, inner_dim, T//p) -> convT1d -> (B, audio_dim, T) -> (B, T, audio_dim)
        x = mx.swapaxes(hidden_states, 1, 2)
        x = self._conv_transpose1d(
            x, self.proj_out_weight, self.proj_out_bias, self.patch_size
        )
        hidden_states = mx.swapaxes(x, 1, 2)
        hidden_states = hidden_states[:, :original_seq_len, :]
        return hidden_states


__all__ = ["TimestepEmbedding", "AceStepDiTModel"]
