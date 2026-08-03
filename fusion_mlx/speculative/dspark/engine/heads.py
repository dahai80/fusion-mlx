"""DSpark auxiliary heads: vanilla Markov head and confidence head.

Mirrors the PyTorch reference:
- refs/deepspec/deepspec/modeling/dspark/markov_head.py (VanillaMarkov)
- refs/deepspec/deepspec/modeling/dspark/common.py (AcceptRatePredictor)

Module/parameter names match the converted checkpoint keys exactly so
``load_weights`` maps directly:
- ``markov_head.markov_w1`` (Embedding [vocab_size, markov_rank])
- ``markov_head.markov_w2`` (Linear [vocab_size, markov_rank], no bias)
- ``confidence_head.proj`` (Linear [1, input_dim], with bias)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MarkovHead(nn.Module):
    """Vanilla Markov head: logits_k = base_logits_k + W2(W1[x_{k-1}]).

    The bias for position k depends only on the previous token, so for a
    *given* token sequence the whole block bias is computable in parallel
    (``bias``/``apply_logits``). During sampling the reference iterates
    serially because x_{k-1} is only known after sampling position k-1;
    the serial loop lives in the runtime (U4), built on ``step_bias``.
    """

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        assert (
            markov_rank > 0
        ), f"MarkovHead requires markov_rank > 0, got {markov_rank}"
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def prev_embeddings(self, token_ids: mx.array) -> mx.array:
        """W1[x] — also the markov feature consumed by the confidence head."""
        return self.markov_w1(token_ids)

    def bias(self, prev_token_ids: mx.array) -> mx.array:
        """Markov bias W2(W1[x_prev]) for any shape of prev token ids.

        prev_token_ids: [..., L] int → returns [..., L, vocab_size].
        """
        return self.markov_w2(self.markov_w1(prev_token_ids))

    def step_bias(self, prev_token_ids: mx.array) -> mx.array:
        """Single-step bias, prev_token_ids [B] → [B, vocab_size]."""
        return self.bias(prev_token_ids)

    def apply_logits(self, base_logits: mx.array, prev_token_ids: mx.array) -> mx.array:
        """base_logits [..., L, V] + bias from prev ids [..., L] (parallel form)."""
        return base_logits + self.bias(prev_token_ids)


class ConfidenceHead(nn.Module):
    """Acceptance-rate predictor: a single linear layer producing a *logit*.

    Matches ``AcceptRatePredictor`` in the reference: the head itself returns
    raw logits (``proj(features).squeeze(-1)``); the sigmoid is applied by the
    caller at threshold time (see ``_confident_prefix_length`` in
    refs/deepspec/deepspec/eval/dspark/draft_ops.py).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.proj = nn.Linear(self.input_dim, 1, bias=True)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1)
