# SPDX-License-Identifier: Apache-2.0
"""PR-M: bucket padding for JIT shape stability (v2 doc §5.2).

mx.compile specializes per input shape. Variable-length sequences hit a
new graph per distinct length — compile time explosion under mixed
prompt sizes (v2 doc §5.2 Bucket-Padding). The fix is deterministic:
snap every sequence up to the next bucket edge, pad, run the compiled
graph, then trim the outputs. Bucket edges are powers-of-two multiples
of a base granularity so the set of shapes a serving lifetime can hit
stays small and closed.

Pure arithmetic — no model decisions (Rule 5). Callers opt in by using
these helpers around their prefill paths; with no callers there is no
behavior change, so no degrade switch is needed (nothing is wired by
default).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)


def next_bucket(length: int, buckets: tuple[int, ...] = DEFAULT_BUCKETS) -> int:
    """Smallest bucket >= length; ValueError if length exceeds all buckets."""
    if length <= 0:
        raise ValueError(f"bucket_pad: length must be positive, got {length}")
    for b in buckets:
        if b >= length:
            return b
    raise ValueError(
        f"bucket_pad: length {length} exceeds largest bucket {buckets[-1]}"
    )


def pad_ids(
    ids: list[int] | np.ndarray,
    buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    pad_id: int = 0,
) -> tuple[np.ndarray, int]:
    """Pad token ids up to the next bucket edge. Returns (padded, original_len)."""
    orig = np.asarray(ids, dtype=np.int64).reshape(-1)
    target = next_bucket(int(orig.size), buckets)
    if target == orig.size:
        return orig, int(orig.size)
    padded = np.full(target, pad_id, dtype=np.int64)
    padded[: orig.size] = orig
    logger.debug("bucket_pad: %d -> %d tokens", orig.size, target)
    return padded, int(orig.size)


def trim_to_length(padded, length: int, axis: int = -1):
    """Trim padded activation output back to the original length."""
    import mlx.core as mx

    arr = mx.asarray(padded)
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(0, length)
    return arr[tuple(slices)]


def bucket_distribution(lengths: list[int], buckets: tuple[int, ...] = DEFAULT_BUCKETS):
    """Deterministic summary of the shapes a batch of lengths maps to.

    Returns {bucket: count} plus the padding waste ratio — the number to
    watch: waste is the cost paid for a closed shape set.
    """
    if not lengths:
        return {"counts": {}, "padded_tokens": 0, "raw_tokens": 0, "waste_ratio": 0.0}
    counts: dict[int, int] = {}
    for n in lengths:
        b = next_bucket(int(n), buckets)
        counts[b] = counts.get(b, 0) + 1
    padded = sum(b * c for b, c in counts.items())
    raw = sum(int(n) for n in lengths)
    return {
        "counts": counts,
        "padded_tokens": padded,
        "raw_tokens": raw,
        "waste_ratio": (padded - raw) / raw if raw else 0.0,
    }
