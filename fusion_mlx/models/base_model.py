# SPDX-License-Identifier: Apache-2.0
"""Base model utilities for custom model implementations."""

from dataclasses import dataclass

import mlx.core as mx

logger = __import__("logging").getLogger(__name__)


@dataclass
class BaseModelArgs:
    pass


@dataclass
class BaseModelOutput:
    last_hidden_state: mx.array
    text_embeds: mx.array | None = None
    pooler_output: mx.array | None = None
    hidden_states: tuple | None = None


def mean_pooling(hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    mask_expanded = attention_mask[:, :, None].astype(hidden_states.dtype)
    sum_embeddings = mx.sum(hidden_states * mask_expanded, axis=1)
    sum_mask = mx.clip(mx.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def last_token_pool(
    hidden_states: mx.array, attention_mask: mx.array | None = None
) -> mx.array:
    if attention_mask is None:
        return hidden_states[:, -1]
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(axis=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[mx.arange(batch_size), sequence_lengths]


def normalize_embeddings(embeddings: mx.array) -> mx.array:
    return embeddings / mx.linalg.norm(embeddings, axis=-1, keepdims=True)
