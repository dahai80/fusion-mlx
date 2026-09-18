# SPDX-License-Identifier: Apache-2.0
"""PR-N: MoE route dispatch + gather combine (v2 doc §2.1/§5.7).

llama.cpp serves MoE via MUL_MAT_ID + GET_ROWS: tokens are reordered by
expert id so each expert's rows are contiguous, the grouped matmul runs
once per expert over contiguous memory, and results are gathered back to
token order and weighted-combined.

fusion-mlx patches (glm_moe_dsa/switch_layers.py) already sort tokens for
gather_mm when indices.size >= 64. What is missing is the deterministic
dispatch bookkeeping: per-expert counts, segment offsets (MUL_MAT_ID
group boundaries), alignment padding for closed shape sets, and a
standalone combine. This module provides those as pure mx ops.

Tier-2 conditional — only meaningful for MoE models; no default wiring,
no behavior change when unused. Callers opt in explicitly.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)


def is_moe_dispatch_enabled() -> bool:
    """FUSION_SHIM_MOE=1 enables the shim MoE path where wired."""
    return os.environ.get("FUSION_SHIM_MOE", "0") == "1"


class DispatchPlan:
    """Deterministic token reorder by expert id (MUL_MAT_ID prep).

    Attributes:
        order: (P,) permutation over flattened (N, k) pairs, sorted by
            (expert_id, pair_rank) — stable by construction.
        sorted_expert_ids: (P,) expert id per sorted pair.
        sorted_token_ids: (P,) source token index (0..N-1) per sorted pair.
        counts: (E,) pairs per expert.
        offsets: (E,) start offset of each expert's segment in sorted order.
        inverse: (P,) sorted position of each original pair (restore map).
    """

    __slots__ = (
        "order",
        "sorted_expert_ids",
        "sorted_token_ids",
        "counts",
        "offsets",
        "inverse",
        "num_experts",
        "num_tokens",
        "top_k",
    )

    def __init__(
        self,
        order,
        sorted_expert_ids,
        sorted_token_ids,
        counts,
        offsets,
        inverse,
        num_experts,
        num_tokens,
        top_k,
    ):
        self.order = order
        self.sorted_expert_ids = sorted_expert_ids
        self.sorted_token_ids = sorted_token_ids
        self.counts = counts
        self.offsets = offsets
        self.inverse = inverse
        self.num_experts = num_experts
        self.num_tokens = num_tokens
        self.top_k = top_k


def route_dispatch(inds, num_experts: int) -> DispatchPlan:
    """Build the dispatch plan for inds of shape (..., k).

    Stability comes from a composite sort key expert * P + pair_rank, so
    equal experts keep original pair order (deterministic under Rule 5).
    """
    if inds.ndim < 1:
        raise ValueError(f"route_dispatch: inds must be >=1D, got shape {inds.shape}")
    top_k = int(inds.shape[-1])
    if top_k < 1:
        raise ValueError(f"route_dispatch: top_k must be >= 1, got {top_k}")
    num_tokens = int(inds.size // top_k)
    if num_tokens < 1:
        raise ValueError(f"route_dispatch: empty inds {inds.shape}")

    flat = inds.reshape(-1).astype(mx.int32)
    num_pairs = flat.shape[0]
    pair_rank = mx.arange(num_pairs, dtype=mx.int32)
    key = flat * num_pairs + pair_rank
    order = mx.argsort(key).astype(mx.int32)

    sorted_expert_ids = flat[order]
    sorted_token_ids = (order // top_k).astype(mx.int32)
    inverse = mx.argsort(order).astype(mx.int32)
    # No scatter_add/bincount in this MLX build — count via one-hot sum
    # over (P, E). Fine for decode-sized P and typical expert counts.
    onehot = (
        sorted_expert_ids[:, None] == mx.arange(num_experts, dtype=mx.int32)[None, :]
    ).astype(mx.int32)
    counts = onehot.sum(axis=0).astype(mx.int32)
    offsets = mx.cumsum(counts) - counts
    logger.debug(
        "route_dispatch: %d tokens x top_%d -> %d pairs, %d experts",
        num_tokens,
        top_k,
        num_pairs,
        num_experts,
    )
    return DispatchPlan(
        order=order,
        sorted_expert_ids=sorted_expert_ids,
        sorted_token_ids=sorted_token_ids,
        counts=counts,
        offsets=offsets,
        inverse=inverse,
        num_experts=num_experts,
        num_tokens=num_tokens,
        top_k=top_k,
    )


def gather_rows(x, plan: DispatchPlan):
    """GET_ROWS analog: gather token rows in sorted-expert order.

    x: (N, D) token activations. Returns (P, D) rows grouped by expert.
    """
    if x.ndim != 2:
        raise ValueError(f"gather_rows: x must be 2D (N, D), got {x.shape}")
    if x.shape[0] != plan.num_tokens:
        raise ValueError(
            f"gather_rows: x has {x.shape[0]} tokens, plan has {plan.num_tokens}"
        )
    return x[plan.sorted_token_ids]


def mul_mat_id(x_sorted, weights, plan: DispatchPlan):
    """MUL_MAT_ID analog: grouped matmul over sorted pairs.

    x_sorted: (P, D_in) from gather_rows. weights: (E, D_out, D_in).
    Returns (P, D_out) — each row times its expert's weight matrix.
    """
    if x_sorted.ndim != 2:
        raise ValueError(
            f"mul_mat_id: x_sorted must be 2D (P, D), got {x_sorted.shape}"
        )
    if weights.ndim != 3:
        raise ValueError(
            f"mul_mat_id: weights must be 3D (E, D_out, D_in), got {weights.shape}"
        )
    # Match SwitchLinear's gather_mm calling convention: lhs (..., 1, D)
    # with 1D rhs_indices broadcasts to (P, 1, D_out); squeeze back.
    lhs = mx.expand_dims(x_sorted, -2)
    out = mx.gather_mm(
        lhs, weights.swapaxes(-1, -2), rhs_indices=plan.sorted_expert_ids
    )
    return mx.squeeze(out, -2)


def aligned_offsets(plan: DispatchPlan, alignment: int):
    """Segment boundaries padded up to `alignment` multiples.

    Returns (starts, ends) so expert e's aligned segment is
    [starts[e], ends[e]) in the padded-sorted layout. Padding keeps the
    shape set closed (same idea as PR-M bucket_pad): only (E,) segment
    sizes exist, not per-token lengths.
    """
    if alignment < 1:
        raise ValueError(f"aligned_offsets: alignment must be >= 1, got {alignment}")
    c = plan.counts.astype(mx.int32)
    padded = ((c + alignment - 1) // alignment) * alignment
    starts = mx.cumsum(padded) - padded
    ends = mx.cumsum(padded)
    return starts, ends


def gather_combine(sorted_y, scores, plan: DispatchPlan) -> mx.array:
    """MoEGatherCombine: restore token order + weighted sum over top_k.

    sorted_y: (P, D) expert outputs in sorted order.
    scores: (..., k) routing weights matching the original inds layout.
    Returns (N, D) combined token outputs.
    """
    if sorted_y.ndim != 2:
        raise ValueError(
            f"gather_combine: sorted_y must be 2D (P, D), got {sorted_y.shape}"
        )
    if sorted_y.shape[0] != plan.num_tokens * plan.top_k:
        raise ValueError(
            f"gather_combine: sorted_y has {sorted_y.shape[0]} rows, "
            f"expected {plan.num_tokens * plan.top_k}"
        )
    flat_scores = scores.reshape(-1).astype(sorted_y.dtype)
    if flat_scores.shape[0] != plan.num_tokens * plan.top_k:
        raise ValueError(
            f"gather_combine: scores has {flat_scores.shape[0]} entries, "
            f"expected {plan.num_tokens * plan.top_k}"
        )
    # Restore original pair order, weight, then sum the k contributions.
    y = sorted_y[plan.inverse]
    weighted = y * flat_scores[:, None]
    return weighted.reshape(plan.num_tokens, plan.top_k, -1).sum(axis=1)


def expert_load_stats(plan: DispatchPlan) -> dict:
    """Deterministic load summary for logging/metrics."""
    counts = [int(v) for v in plan.counts]
    total = sum(counts)
    active = sum(1 for c in counts if c > 0)
    peak = max(counts) if counts else 0
    return {
        "pairs": total,
        "experts_total": plan.num_experts,
        "experts_active": active,
        "max_pairs_per_expert": peak,
        "imbalance": (peak * plan.num_experts / total) if total else 0.0,
    }
