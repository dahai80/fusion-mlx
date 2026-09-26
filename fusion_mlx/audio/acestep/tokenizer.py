# SPDX-License-Identifier: Apache-2.0
# ACE-Step MLX audio tokenizer (issue #988): ResidualFSQ quantizer +
# attention pooler tokenize, detokenizer expands tokens back to acoustic.
# Ported from ACE-Step/Ace-Step1.5 modeling_acestep_v15_turbo.py (MIT).
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from .blocks import ResidualFSQ
from .condition import AttentionPooler, EncoderLayer, _run_encoder_layers
from .config import AceStepConfig

logger = logging.getLogger(__name__)


class AudioTokenDetokenizer(nn.Module):
    # Expands quantized tokens (B, T, D) back to acoustic features (B, T*P, acoustic_dim).
    # Each token -> pool_window_size patches via learnable special tokens, encoder layers, proj_out.
    def __init__(self, config: AceStepConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.special_tokens = (
            mx.random.normal((1, config.pool_window_size, config.hidden_size)) * 0.02
        )
        self.layers = [
            EncoderLayer(config, i)
            for i in range(config.num_attention_pooler_hidden_layers)
        ]
        self.proj_out = nn.Linear(config.hidden_size, config.audio_acoustic_hidden_dim)

    def __call__(self, x: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        B, T, D = x.shape
        P = self.config.pool_window_size
        x = self.embed_tokens(x)  # (B, T, D)
        x = mx.repeat(x[:, :, None, :], P, axis=2)  # (B, T, P, D)
        sp = mx.broadcast_to(self.special_tokens, (B, T, P, D))
        x = x + sp
        x = mx.reshape(x, (B * T, P, D))
        am = mx.ones((B * T, P), dtype=x.dtype)
        x = _run_encoder_layers(self.layers, x, am, self.config, use_sliding=False)
        x = self.norm(x)
        x = self.proj_out(x)  # (B*T, P, acoustic_dim)
        x = mx.reshape(x, (B, T * P, -1))
        return x


class AceStepAudioTokenizer(nn.Module):
    # Continuous acoustic -> discrete tokens. proj -> attention pooler -> ResidualFSQ.
    def __init__(self, config: AceStepConfig):
        super().__init__()
        self.audio_acoustic_proj = nn.Linear(
            config.audio_acoustic_hidden_dim, config.hidden_size
        )
        self.attention_pooler = AttentionPooler(config)
        self.quantizer = ResidualFSQ(
            dim=config.fsq_dim,
            levels=config.fsq_input_levels,
            num_quantizers=config.fsq_input_num_quantizers,
        )
        self.pool_window_size = config.pool_window_size
        self.config = config

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # x: (B, T, P, acoustic_dim) -> pool to (B, T, hidden) -> quantize
        x = self.audio_acoustic_proj(x)  # (B, T, P, hidden)
        x = self.attention_pooler(x)  # (B, T, hidden)
        quantized, indices = self.quantizer(x)
        return quantized, indices

    def tokenize(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # x: (B, T*P, acoustic_dim) -> reshape (B, T, P, acoustic_dim)
        P = self.pool_window_size
        T = x.shape[1] // P
        x = mx.reshape(x, (x.shape[0], T, P, x.shape[-1]))
        return self.__call__(x)


__all__ = ["AudioTokenDetokenizer", "AceStepAudioTokenizer"]
