# SPDX-License-Identifier: Apache-2.0
"""Shared builders for qwen3_5 LSA attention tests (not collected by pytest)."""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.qwen3_next import Qwen3NextAttention

HEAD_DIM = 32
HIDDEN = 64
KV_HEADS = 2
Q_HEADS = 4
LATENT = 48
ROTARY = 8


class LsaTestArgs:
    def __init__(self):
        self.hidden_size = HIDDEN
        self.num_attention_heads = Q_HEADS
        self.num_key_value_heads = KV_HEADS
        self.head_dim = HEAD_DIM
        self.attention_bias = False
        self.rms_norm_eps = 1e-6
        self.partial_rotary_factor = ROTARY / HEAD_DIM
        self.rope_theta = 10000.0
        self.rope_scaling = None
        self.max_position_embeddings = 4096


def make_attn():
    mx.random.seed(11)
    return Qwen3NextAttention(LsaTestArgs())


def write_map(tmp_path, layers=(0,)):
    import numpy as np
    from safetensors.numpy import save_file

    tensors = {}
    for layer in layers:
        tensors[f"layer{layer}.query_projection"] = np.zeros(
            (LATENT, 8), dtype=np.float32
        )
        tensors[f"layer{layer}.key_projection"] = np.zeros(
            (LATENT, 8), dtype=np.float32
        )
    path = str(tmp_path / "map.safetensors")
    save_file(
        tensors,
        path,
        metadata={
            "model_arch": "qwen38_27b",
            "indexer_layers": ",".join(str(l) for l in layers),
            "latent_dim": str(LATENT),
            "index_dim": "8",
        },
    )
    return path
