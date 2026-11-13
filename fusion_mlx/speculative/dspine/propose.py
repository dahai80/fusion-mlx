# SPDX-License-Identifier: Apache-2.0
# Ported from tencent DSpine reference MLX implementation
# (github.com/tencent/DSpine, dspine_mlx, Apache-2.0).
from __future__ import annotations

import math
import os

import mlx.core as mx

from .model import DSpineDraftModel
from .target import HeadAccess

_TOPK_MODE = os.environ.get("DSPINE_TOPK", "two")


def _topk_ids_full(logits: mx.array, k: int) -> mx.array:
    order = mx.argsort(-logits.astype(mx.float32), axis=-1)[:, :k]
    return mx.sort(order, axis=-1).astype(mx.int32)


def topk_ids(logits: mx.array, k: int) -> mx.array:
    """Top-k token ids per row, ascending by id (ties: the lower id wins).

    Default is the reference's two-stage chunked stable top-k: the max of each
    64-wide chunk, the K chunks with the largest maxima (stable, lower chunk
    id), then the stable top-K of those chunks. It matches the full stable
    argsort's selected ids while touching 1/64 of the vocabulary for the sort.
    Set DSPINE_TOPK=full to use the old full-vocabulary argsort.
    """
    if logits.ndim != 2:
        raise ValueError("topk_ids expects [n, V] logits")
    n, V = logits.shape
    if n == 0:
        return mx.zeros((0, k), dtype=mx.int32)
    ch = 64
    nch = (V + ch - 1) // ch
    if _TOPK_MODE == "full" or nch < k:
        return _topk_ids_full(logits, k)
    pad = nch * ch - V
    padded = (
        mx.concatenate(
            [logits, mx.full((n, pad), -mx.inf, dtype=logits.dtype)], axis=-1
        )
        if pad
        else logits
    )
    chunks = padded.reshape(n, nch, ch)
    chunk_max = mx.max(chunks, axis=-1)
    chunk_ids = mx.sort(
        mx.argsort(-chunk_max.astype(mx.float32), axis=-1)[:, :k], axis=-1
    )
    gathered = mx.take_along_axis(
        chunks,
        mx.broadcast_to(chunk_ids[:, :, None], (n, k, ch)),
        axis=1,
    ).reshape(n, k * ch)
    local = mx.argsort(-gathered.astype(mx.float32), axis=-1)[:, :k]
    selected_chunks = mx.take_along_axis(chunk_ids, local // ch, axis=1)
    ids = selected_chunks * ch + (local % ch)
    return mx.sort(ids, axis=-1).astype(mx.int32)


def filter_probs(
    p: mx.array, top_k: int, top_p: float, min_p: float
) -> tuple[mx.array, mx.array]:
    """Joint top-k / top-p + min-p on one chain row; returns (q, keep)."""
    K = p.shape[0]
    lane = mx.arange(K)
    p_c = p[:, None]
    p_d = p[None, :]
    ahead = (p_d > p_c) | ((p_d == p_c) & (lane[None, :] < lane[:, None]))
    rank = mx.sum(ahead.astype(mx.int32), axis=1)
    before = mx.sum(mx.where(ahead, p_d, 0.0), axis=1)
    keep = (rank < int(top_k)) & (before <= float(top_p))
    if min_p > 0:
        keep = keep & (p >= float(min_p) * mx.max(p))
    w = mx.where(keep, p, 0.0)
    q = w / mx.sum(w)
    return q, keep


def filter_target_probs(
    probs: mx.array, top_k: int, top_p: float, min_p: float
) -> mx.array:
    """Vectorized joint top-k / top-p + min-p over [rows, V] target probs."""
    rows, V = probs.shape
    if V == 0:
        return probs
    order = mx.argsort(-probs, axis=-1)  # [rows, V]
    rank = mx.argsort(order, axis=-1)  # inverse permutation: rank of each id
    sorted_p = mx.take_along_axis(probs, order, axis=-1)
    cum = mx.cumsum(sorted_p, axis=-1)
    before_sorted = cum - sorted_p
    before = mx.take_along_axis(before_sorted, rank, axis=-1)
    keep = (rank < int(top_k)) & (before <= float(top_p))
    if min_p > 0:
        keep = keep & (probs >= float(min_p) * mx.max(probs, axis=-1, keepdims=True))
    w = mx.where(keep, probs, 0.0)
    return w / mx.sum(w, axis=-1, keepdims=True)


def _chain(idx: mx.array, cand: mx.array) -> mx.array:
    """Walk the K-state greedy table; returns the selected candidate per row."""
    n = cand.shape[0]
    j = idx[0, 0]
    out = []
    for i in range(n):
        out.append(mx.take(cand[i], j))
        if i + 1 < n:
            j = mx.take(idx[i + 1], j)
    return mx.stack(out).astype(mx.int32)


def serial_sample_chain(
    table: mx.array,
    cand: mx.array,
    *,
    inv_temp: float,
    top_k: int,
    top_p: float,
    min_p: float,
    uniforms: mx.array,
) -> tuple[mx.array, mx.array]:
    """Sample each row's chain under the request's filters."""
    n, K, _ = table.shape
    lane = mx.arange(K)
    sender = mx.array(0, dtype=mx.int32)
    toks = []
    qs = []
    for i in range(n):
        row = mx.take(table[i], sender, axis=0)
        p = mx.softmax(row * float(inv_temp), axis=-1)
        q, keep = filter_probs(p, top_k, top_p, min_p)
        cdf = mx.cumsum(q, axis=-1)
        threshold = uniforms[i] * cdf[-1]
        choice = mx.sum((cdf <= threshold).astype(mx.int32))
        last = mx.max(mx.where(keep, lane, 0))
        choice = mx.minimum(choice, last)
        toks.append(mx.take(cand[i], choice))
        qs.append(q)
        sender = choice
    return mx.stack(toks).astype(mx.int32), mx.stack(qs)


class RefinementTail:
    """Serial last-write refinement over the proposal rows."""

    def __init__(
        self,
        draft: DSpineDraftModel,
        head: HeadAccess,
        tg: mx.array,
        tsv: mx.array,
    ):
        self.draft = draft
        self.head = head
        self.tg = tg
        self.tsv = tsv
        self.k = int(draft.head_topk)
        bridge = draft.selected_prefix_bridge
        self.rank = int(bridge.rank)
        self.code_size = int(bridge.code_size)
        self.sqrt_c = math.sqrt(self.code_size)
        self.norm_w = draft.norm.weight.astype(mx.float32)
        self.eps = float(draft.norm.eps)
        self.gate_b = bridge.gate.bias.astype(mx.float32)
        self.codebook = bridge.codebook
        self.pred_gate_w = bridge.predecessor_gate.weight.astype(mx.float32)
        self.prev_w = bridge.prev.weight.astype(mx.float32)
        self.write_w = bridge.feature_write_weights[-1].astype(mx.float32)

    def _split_stash(self, stash: dict, n: int, start: int = 1):
        # ``start`` is the stash row index of the first REFINED position's
        # sender: the row at ``start - 1`` is dropped (it only feeds the
        # serial chain via tsv token lookups). Default 1 keeps the pure
        # path's contract (row 0 = anchor). N11 hybrid passes start=p so
        # rows p+1..n are refined after a p-token given prefix.
        raw = stash["raw"]
        rms = stash["rms"]
        packed = stash["packed"]
        T = n + start
        if raw.shape[0] != T:
            raise ValueError(
                f"refinement stash has {raw.shape[0]} rows, expected T={T}"
            )
        return (
            raw[start:].astype(mx.float32),  # [n,H]
            rms[start:].astype(mx.float32),  # [n]
            packed[start:, : self.rank].astype(mx.float32),  # [n,R]
        )

    def candidates(self, rows: mx.array) -> mx.array:
        logits = self.head.matmul(rows)
        return topk_ids(logits, self.k)

    def greedy(self, rows: mx.array, stash: dict, anchor: int):
        """Returns (draft token ids [n], candidates [n,K])."""
        n = rows.shape[0]
        raw, rms, q = self._split_stash(stash, n)
        cand = self.candidates(rows)

        head_rows = self.head.rows(cand).astype(mx.float32)  # [n,K,H]
        base = mx.sum((raw[:, None, :] * self.norm_w) * head_rows, axis=-1)  # [n,K]

        anchor_arr = mx.full((1, self.k), int(anchor), dtype=mx.int32)
        sid = (
            mx.concatenate([anchor_arr, cand[:-1]], axis=0) if n > 1 else anchor_arr
        )  # [n,K]
        tsv_sid = self.tsv[sid].astype(mx.float32)  # [n,K,2R]
        tvg = tsv_sid[..., : self.rank]
        tvv = tsv_sid[..., self.rank :]
        m = mx.sigmoid(q[:, None, :] + tvg + self.gate_b) * tvv  # [n,K,R]
        tg_cand = self.tg[cand].astype(mx.float32)  # [n,K,R]
        s = mx.matmul(m, tg_cand.swapaxes(1, 2))  # [n,K,K]
        score = base[:, None, :] + rms[:, None, None] * s
        idx = mx.argmax(score, axis=-1).astype(mx.int32)  # [n,K]
        return _chain(idx, cand), cand

    def sampled_table(
        self, rows: mx.array, stash: dict, anchor: int
    ) -> tuple[mx.array, mx.array]:
        """Returns (table [n,K,K] fp32, candidates [n,K])."""
        raw, rms, q = self._split_stash(stash, rows.shape[0])
        return self._serial_table(rows, raw, rms, q, anchor)

    def _serial_table(
        self,
        rows: mx.array,
        raw: mx.array,
        rms: mx.array,
        q: mx.array,
        anchor: int,
    ) -> tuple[mx.array, mx.array]:
        """Serial-table math over pre-split stash parts (N11 hybrid seam)."""
        n = rows.shape[0]
        cand = self.candidates(rows)

        anchor_arr = mx.full((1, self.k), int(anchor), dtype=mx.int32)
        sid = (
            mx.concatenate([anchor_arr, cand[:-1]], axis=0) if n > 1 else anchor_arr
        )  # [n,K]
        e = self.sqrt_c * self.codebook[sid].astype(mx.float32)  # [n,K,C]
        gate = mx.sigmoid(
            q[:, None, :] + mx.matmul(e, self.pred_gate_w.T) + self.gate_b
        )  # [n,K,R]
        value = mx.matmul(e, self.prev_w.T)  # [n,K,R]
        refined = raw[:, None, :] + rms[:, None, None] * mx.matmul(
            gate * value, self.write_w.T
        )  # [n,K,H]
        norm = mx.sqrt(mx.mean(refined * refined, axis=-1, keepdims=True) + self.eps)
        normalized = refined / norm * self.norm_w
        head_rows = self.head.rows(cand).astype(mx.float32)  # [n,K,H]
        table = mx.matmul(normalized, head_rows.swapaxes(1, 2))  # [n,K,K]
        return table, cand

    def sampled(
        self,
        rows: mx.array,
        stash: dict,
        anchor: int,
        *,
        inv_temp: float,
        top_k: int,
        top_p: float,
        min_p: float,
        uniforms: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Returns (draft token ids [n], q rows [n,K], candidates [n,K])."""
        table, cand = self.sampled_table(rows, stash, anchor)
        tokens, q_rows = serial_sample_chain(
            table,
            cand,
            inv_temp=inv_temp,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            uniforms=uniforms,
        )
        return tokens, q_rows, cand

    def sampled_hybrid(
        self,
        rows: mx.array,
        stash: dict,
        anchor: int,
        prefix_tokens: list[int],
        *,
        inv_temp: float,
        top_k: int,
        top_p: float,
        min_p: float,
        uniforms: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Sampled refinement of the rows AFTER a given (non-mask) prefix.

        N11 hybrid drafting: block positions 1..len(prefix_tokens) carried
        real n-gram tokens instead of masks, so their hidden rows are
        already conditioned on the prefix through the block forward. Only
        the suffix rows are refined here, and the serial chain's first
        sender is the LAST prefix token (exact_sender semantics: the tail
        reads sender codes from the serial tables, exactly as it does for
        the anchor in the pure path).
        """
        p = len(prefix_tokens)
        n_total = int(rows.shape[0])
        if not 0 < p < n_total:
            raise ValueError(
                f"sampled_hybrid needs 0 < prefix {p} < draft rows {n_total}"
            )
        # Stash rows are [anchor, pos1..posN]; the sender row for the first
        # refined position is pos p (stash index p), so refined rows start
        # at stash index p + 1.
        raw, rms, q = self._split_stash(stash, n_total - p, start=p + 1)
        table, cand = self._serial_table(rows[p:], raw, rms, q, int(prefix_tokens[-1]))
        tokens, q_rows = serial_sample_chain(
            table,
            cand,
            inv_temp=inv_temp,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            uniforms=uniforms,
        )
        return tokens, q_rows, cand
