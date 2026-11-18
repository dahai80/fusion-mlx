# SPDX-License-Identifier: Apache-2.0
"""Chunked LM-head cross-entropy loss for long-sequence SFT.

mlx-lm's default_loss materializes the full [B, T, V] logits tensor
plus fp32 CE intermediates and a same-size gradient; measured peak on
this class of models is ~2.6GB per 1k tokens (V=151936, bf16 head
matmul) — 41.6GB at T=32768 for the head loss alone
(docs/logs/cce-spike-1118.md in fusion-research).

This loss computes the head CE in token chunks:
  - forward transient drops from T*V to chunk*V elements
  - CE arithmetic stays in bf16 (matches default fp32 loss within
    bf16 resolution; sum accumulation is fp32)
  - under mlx-lm's whole-graph value_and_grad training loop the
    backward still retains per-chunk logits (grad of CE needs them),
    so the training-step win is ~34% at T=8192 and grows with T.
    The -88% head-only number from the spike requires per-chunk
    backward, which needs a chunked-body trainer (StreamBP scope).
  - for eval (forward-only, no backward) there is no retention at
    all: peak drops to chunk*V elements.

Models without a separate lm_head attribute fall back to the default
whole-model forward with default-style fp32 CE.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_TOKENS = 1024


def make_chunked_ce_loss(chunk_tokens: int = DEFAULT_CHUNK_TOKENS):
    def chunked_ce_loss(model, batch, lengths):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        body = getattr(model, "model", None)
        head = getattr(model, "lm_head", None)
        if body is None or head is None:
            # fallback: default-style whole-model fp32 CE
            logits = model(inputs)
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            ce = nn.losses.cross_entropy(logits, targets) * mask
            ntoks = mask.sum()
            return ce.astype(mx.float32).sum() / ntoks, ntoks

        h = body(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

        ce_sum = mx.array(0.0, dtype=mx.float32)
        ntoks = mask.sum()
        for i in range(0, targets.shape[1], chunk_tokens):
            j = min(i + chunk_tokens, targets.shape[1])
            logits_c = head(h[:, i:j])
            ce_c = nn.losses.cross_entropy(logits_c, targets[:, i:j])
            ce_sum = ce_sum + (ce_c * mask[:, i:j]).astype(mx.float32).sum()
        return ce_sum / ntoks, ntoks

    return chunked_ce_loss
