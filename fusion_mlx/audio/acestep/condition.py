# SPDX-License-Identifier: Apache-2.0
# ACE-Step MLX condition encoders (issue #988): lyric encoder, timbre encoder,
# attention pooler, condition encoder (packs text+lyric+timbre for cross-attn).
# Ported from ACE-Step/Ace-Step1.5 modeling_acestep_v15_turbo.py (MIT).
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from .blocks import MLP, Attention, create_4d_mask, rope_freqs
from .config import AceStepConfig

logger = logging.getLogger(__name__)


class EncoderLayer(nn.Module):
    # Qwen3-style encoder layer: self-attn (bidirectional) + MLP, residual.
    # No AdaLN (condition encoders are not timestep-modulated).
    def __init__(self, config: AceStepConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx, is_cross=False)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MLP(config)
        self.attention_type = config.layer_types[layer_idx]

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(
            h, mask=attention_mask, position_embeddings=position_embeddings
        )
        hidden_states = residual + h
        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h = self.mlp(h)
        hidden_states = residual + h
        return hidden_states


def _run_encoder_layers(
    layers,
    hidden_states: mx.array,
    attention_mask: mx.array | None,
    config: AceStepConfig,
    use_sliding: bool,
) -> mx.array:
    seq_len = hidden_states.shape[1]
    cos, sin = rope_freqs(config.head_dim, seq_len, config.rope_theta)
    full_mask = create_4d_mask(
        seq_len, hidden_states.dtype, attention_mask=attention_mask
    )
    sliding_mask = None
    if use_sliding and config.use_sliding_window:
        sliding_mask = create_4d_mask(
            seq_len,
            hidden_states.dtype,
            attention_mask=attention_mask,
            sliding_window=config.sliding_window,
        )
    for layer in layers:
        at = layer.attention_type
        m = (
            sliding_mask
            if at == "sliding_attention" and sliding_mask is not None
            else full_mask
        )
        hidden_states = layer(
            hidden_states, position_embeddings=(cos, sin), attention_mask=m
        )
    return hidden_states


class LyricEncoder(nn.Module):
    # 8-layer Qwen3-style bidirectional encoder over lyric text embeddings.
    def __init__(self, config: AceStepConfig):
        super().__init__()
        self.embed_tokens = nn.Linear(config.text_hidden_dim, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            EncoderLayer(config, i)
            for i in range(config.num_lyric_encoder_hidden_layers)
        ]
        self.config = config

    def __call__(self, inputs_embeds: mx.array, attention_mask: mx.array) -> mx.array:
        h = self.embed_tokens(inputs_embeds)
        h = _run_encoder_layers(
            self.layers, h, attention_mask, self.config, use_sliding=True
        )
        return self.norm(h)


class AttentionPooler(nn.Module):
    # Pools patch sequences via a special CLS-like token attending to all patches.
    # Input: (B, T, P, D). Output: (B, T, D) pooled.
    def __init__(self, config: AceStepConfig):
        super().__init__()
        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.special_token = mx.random.normal((1, 1, config.hidden_size)) * 0.02
        self.layers = [
            EncoderLayer(config, i)
            for i in range(config.num_attention_pooler_hidden_layers)
        ]
        self.config = config

    def __call__(self, x: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        B, T, P, D = x.shape
        x = self.embed_tokens(x)
        sp = mx.broadcast_to(self.special_token, (B, T, 1, D))
        x = mx.concatenate([sp, x], axis=2)  # (B, T, P+1, D)
        x = mx.reshape(x, (B * T, P + 1, D))
        # mask: (B, T, P) -> (B*T, P+1) with special token always valid
        if attention_mask is not None:
            am = attention_mask.reshape((B * T, P))
            sp_mask = mx.ones((B * T, 1), dtype=am.dtype)
            am = mx.concatenate([sp_mask, am], axis=1)
        else:
            am = mx.ones((B * T, P + 1), dtype=x.dtype)
        x = _run_encoder_layers(self.layers, x, am, self.config, use_sliding=False)
        # pooled = special token output (position 0)
        pooled = x[:, 0, :]  # (B*T, D)
        pooled = mx.reshape(pooled, (B, T, D))
        return self.norm(pooled)


class TimbreEncoder(nn.Module):
    # 4-layer encoder over reference audio acoustic features + special token.
    # Extracts timbre embedding (pooled special token) per reference clip.
    def __init__(self, config: AceStepConfig):
        super().__init__()
        self.embed_tokens = nn.Linear(config.timbre_hidden_dim, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.special_token = mx.random.normal((1, 1, config.hidden_size))
        self.layers = [
            EncoderLayer(config, i)
            for i in range(config.num_timbre_encoder_hidden_layers)
        ]
        self.config = config

    def __call__(
        self, packed: mx.array, order_mask: mx.array
    ) -> tuple[mx.array, mx.array]:
        # packed: (N, timbre_hidden_dim) reference clips. order_mask: (N,) batch ids.
        N = packed.shape[0]
        h = self.embed_tokens(packed)  # (N, D)
        sp = mx.broadcast_to(self.special_token, (N, 1, h.shape[-1]))
        h = mx.concatenate([sp, mx.expand_dims(h, 1)], axis=1)  # (N, 2, D)
        am = mx.ones((N, 2), dtype=h.dtype)
        h = _run_encoder_layers(self.layers, h, am, self.config, use_sliding=False)
        timb = h[:, 0, :]  # (N, D) pooled special token
        # unpack to (B, max_count, D)
        timb_unpacked, mask = self._unpack(timb, order_mask)
        return timb_unpacked, mask

    def _unpack(
        self, timb: mx.array, order_mask: mx.array
    ) -> tuple[mx.array, mx.array]:
        N, d = timb.shape
        B = int(mx.max(order_mask).item()) + 1
        # count per batch
        counts = mx.zeros((B,), dtype=mx.int32)
        for i in range(N):
            b = int(order_mask[i].item())
            counts = mx.where(mx.arange(B) == b, counts + 1, counts)
        max_count = int(mx.max(counts).item())
        # scatter into (B, max_count, d)
        out = mx.zeros((B, max_count, d), dtype=timb.dtype)
        mask = mx.zeros((B, max_count), dtype=mx.int32)
        pos = mx.zeros((B,), dtype=mx.int32)
        for i in range(N):
            b = int(order_mask[i].item())
            p = int(pos[b].item())
            out[b, p, :] = timb[i]
            mask[b, p] = 1
            pos = mx.where(mx.arange(B) == b, pos + 1, pos)
        return out, mask


def pack_sequences(h1, h2, m1, m2):
    # Concatenate, sort valid-first, return (packed_hidden, packed_mask).
    hidden = mx.concatenate([h1, h2], axis=1)  # (B, L1+L2, D)
    mask = mx.concatenate([m1, m2], axis=1).astype(mx.float32)  # (B, L1+L2)
    B, L = mask.shape
    sort_idx = mx.argsort(-mask, axis=1)  # descending: valid(1) first
    gathered = mx.take_along_axis(
        hidden, mx.broadcast_to(sort_idx[:, :, None], (B, L, hidden.shape[-1])), axis=1
    )
    lengths = mask.sum(axis=1).astype(mx.int32)
    new_mask = (mx.arange(L)[None, :] < lengths[:, None]).astype(mask.dtype)
    return gathered, new_mask


class ConditionEncoder(nn.Module):
    # Packs text + lyric + timbre into a single cross-attn conditioning sequence.
    def __init__(self, config: AceStepConfig):
        super().__init__()
        self.text_projector = nn.Linear(
            config.text_hidden_dim, config.hidden_size, bias=False
        )
        self.lyric_encoder = LyricEncoder(config)
        self.timbre_encoder = TimbreEncoder(config)
        self.config = config

    def __call__(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: mx.array | None,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: mx.array,
        refer_audio_packed: mx.array | None,
        refer_audio_order_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        text_hidden_states = self.text_projector(text_hidden_states)
        lyric_h = self.lyric_encoder(lyric_hidden_states, lyric_attention_mask)
        if refer_audio_packed is not None and refer_audio_packed.shape[0] > 0:
            timb_h, timb_m = self.timbre_encoder(
                refer_audio_packed, refer_audio_order_mask
            )
            enc_h, enc_m = pack_sequences(lyric_h, timb_h, lyric_attention_mask, timb_m)
        else:
            enc_h, enc_m = lyric_h, lyric_attention_mask
        if text_hidden_states is not None:
            tm = (
                text_attention_mask
                if text_attention_mask is not None
                else mx.ones(
                    (text_hidden_states.shape[0], text_hidden_states.shape[1]),
                    dtype=enc_m.dtype,
                )
            )
            enc_h, enc_m = pack_sequences(enc_h, text_hidden_states, enc_m, tm)
        return enc_h, enc_m


__all__ = [
    "EncoderLayer",
    "LyricEncoder",
    "AttentionPooler",
    "TimbreEncoder",
    "ConditionEncoder",
    "pack_sequences",
]
