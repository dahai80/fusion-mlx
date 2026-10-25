# SPDX-License-Identifier: Apache-2.0
# #1400: sequence packing for SFT. mlx-lm 0.31.3 has no packing, so this
# module supplies the three pieces its trainer lets a caller override:
#
#   * iterate_packed_batches — packs short samples into rows of at most
#     max_seq_length and yields (tokens, segment_ids, loss_mask) batches
#     instead of the (tokens, lengths) pairs iterate_batches produces.
#   * packed_loss — cross-entropy masked per token, with a block-diagonal
#     attention mask so one packed sample never attends to its neighbour.
#   * install_mask_injection — mlx-lm models do not accept a mask argument at
#     the Model level (only the per-block Attention does), so the block
#     __call__ is patched to thread the current packed mask through. Same
#     class-level patching mlx-lm's own grad_checkpoint() uses.
#
# Position ids are not reset per segment: MLX 0.32 RoPE offsets must be scalar
# (mx.fast.rope rejects array offsets), and RoPE attention is relative anyway,
# so intra-segment scores are identical to unpacked training as long as the
# segments cannot attend across their boundary. That is what the block mask
# enforces; the only difference is larger absolute positions, which matters
# only near max_position_embeddings (max_seq_length bounds that).

from __future__ import annotations

import logging
from contextvars import ContextVar

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

_PAD_TO = 32
_PAD_TOKEN_ID = 0

_CURRENT_MASK: ContextVar = ContextVar("fusion_packed_attention_mask", default=None)

_INSTALLED_CLASSES: set[type] = set()


def pack_samples(samples, max_seq_length: int):
    samples_iter = list(samples)
    rows: list[dict] = []
    truncated = 0
    skipped = 0
    for tokens, offset in sorted(samples_iter, key=lambda s: len(s[0])):
        tokens = list(tokens)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            truncated += 1
        offset = max(0, min(int(offset or 0), len(tokens)))
        if not tokens:
            skipped += 1
            continue
        row = (
            rows[-1]
            if rows and len(rows[-1]["tokens"]) + len(tokens) <= max_seq_length
            else None
        )
        if row is None:
            row = {"tokens": [], "segments": []}
            rows.append(row)
        start = len(row["tokens"])
        row["segments"].append((start, start + len(tokens), offset))
        row["tokens"].extend(tokens)
    if truncated:
        logger.warning(
            "packing: truncated %d sample(s) longer than max_seq_length=%d",
            truncated,
            max_seq_length,
        )
    if skipped:
        logger.warning("packing: skipped %d empty sample(s)", skipped)
    logger.info(
        "packing: %d samples -> %d rows (avg %.2f samples/row, fill %.1f%%)",
        len(samples_iter),
        len(rows),
        (len(samples_iter) / len(rows)) if rows else 0.0,
        (
            100.0 * (sum(len(r["tokens"]) for r in rows) / (len(rows) * max_seq_length))
            if rows
            else 0.0
        ),
    )
    return rows


def _materialize(dataset):
    out = []
    for i in range(len(dataset)):
        item = dataset[i]
        if isinstance(item, dict):
            item = dataset.process(item)
        out.append(item)
    return out


def _build_batch(rows, max_seq_length: int):
    longest = max(len(r["tokens"]) for r in rows)
    width = 1 + _PAD_TO * ((longest + _PAD_TO - 1) // _PAD_TO)
    width = min(width, max_seq_length)
    batch = np.full((len(rows), width), _PAD_TOKEN_ID, dtype=np.int32)
    segments = np.full((len(rows), width), -1, dtype=np.int32)
    loss_mask = np.zeros((len(rows), width), dtype=bool)
    for i, row in enumerate(rows):
        n = min(len(row["tokens"]), width)
        batch[i, :n] = row["tokens"][:n]
        for seg_id, (start, end, offset) in enumerate(row["segments"]):
            if start >= n:
                break
            stop = min(end, n)
            segments[i, start:stop] = seg_id
            train_from = start + min(offset, stop - start)
            loss_mask[i, train_from:stop] = True
    return mx.array(batch), mx.array(segments), mx.array(loss_mask)


_ROWS_CACHE: dict[tuple, list] = {}


def _packed_rows(dataset, max_seq_length: int):
    # train() builds one iterator for the epoch loop and evaluate() builds a
    # fresh one per validation pass; packing the same dataset again every time
    # re-runs the greedy layout (and re-logs the fill line).
    key = (id(dataset), len(dataset), max_seq_length)
    rows = _ROWS_CACHE.get(key)
    if rows is None:
        rows = pack_samples(_materialize(dataset), max_seq_length)
        if len(_ROWS_CACHE) > 8:
            _ROWS_CACHE.clear()
        _ROWS_CACHE[key] = rows
    return rows


def iterate_packed_batches(
    dataset,
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    seed=None,
    comm_group=None,
):
    if comm_group is not None and comm_group.size() > 1:
        raise ValueError(
            "pack_sequences is not supported in distributed training "
            f"(world size {comm_group.size()}); run single-node or disable packing"
        )
    rows = _packed_rows(dataset, max_seq_length)
    if len(rows) < batch_size:
        raise ValueError(
            f"Dataset must pack into at least batch_size={batch_size} rows "
            f"but only has {len(rows)}; lower batch_size or max_seq_length"
        )
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(rows))
        starts = range(0, len(order) - batch_size + 1, batch_size)
        for i in starts:
            yield _build_batch(
                [rows[j] for j in order[i : i + batch_size]], max_seq_length
            )
        if not loop:
            break


def block_diagonal_mask(segments):
    seg = segments[:, None, :, None]
    return seg == segments[:, None, None, :]


def packed_loss(model, batch, segments, loss_mask):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    mask = loss_mask[:, 1:].astype(mx.float32)
    attn = block_diagonal_mask(segments[:, :-1])
    token = _CURRENT_MASK.set(attn)
    try:
        logits = model(inputs)
    finally:
        _CURRENT_MASK.reset(token)
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / mx.maximum(ntoks, 1), ntoks


def _block_classes(model):
    layers = getattr(model, "layers", None)
    if layers is None:
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None)
    if not layers:
        raise RuntimeError(
            "pack_sequences: cannot find model.layers on the loaded model; "
            "sequence packing is unsupported for this architecture"
        )
    return [type(layer) for layer in layers]


def install_mask_injection(model):
    import inspect

    patchable = 0
    unsupported = []
    for cls in _block_classes(model):
        if cls in _INSTALLED_CLASSES:
            patchable += 1
            continue
        original = cls.__call__
        params = list(inspect.signature(original).parameters)
        if params[:4] != ["self", "x", "mask", "cache"]:
            unsupported.append(f"{cls.__module__}.{cls.__name__}{tuple(params)}")
            continue

        def patched(self, x, mask=None, cache=None, _original=original):
            packed = _CURRENT_MASK.get()
            if packed is not None:
                mask = packed
            return _original(self, x, mask, cache)

        cls.__call__ = patched
        _INSTALLED_CLASSES.add(cls)
        patchable += 1
    if unsupported:
        raise RuntimeError(
            "pack_sequences: these transformer blocks do not accept "
            "(x, mask, cache) so a packed attention mask cannot be injected: "
            + ", ".join(sorted(unsupported))
        )
    logger.info(
        "pack_sequences: attention-mask injection installed on %d block class(es)",
        patchable,
    )
