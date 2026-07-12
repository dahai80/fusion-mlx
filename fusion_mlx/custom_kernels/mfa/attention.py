# SPDX-License-Identifier: Apache-2.0
# Core Flash Attention implementation - wraps mlx_mfa or falls back to MLX SDPA.

from __future__ import annotations

import logging

import mlx.core as mx

from ..mfa import _HAS_MFA_EXT, _MFA_EXT
from .masks import make_causal_mask, make_sliding_window_mask

logger = logging.getLogger(__name__)


def _normalize_qkv_layout(q, k, v):
    # Ensure Q/K/V are in (B, H, N, D) layout.
    # If inputs are (B, N, H, D), transpose to (B, H, N, D).
    # Returns (q, k, v, was_transposed).
    if q.ndim != 4:
        return q, k, v, False
    d1, d2 = q.shape[1], q.shape[2]
    is_head = lambda x: 1 <= x <= 32
    if is_head(d2) and not is_head(d1):
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        return q, k, v, True
    if is_head(d1) and not is_head(d2):
        return q, k, v, False
    # Ambiguous: both dims in head range. Transpose only if d2 is a plausible
    # head count (>=2) and smaller than d1. d2==1 is a single-token decode query
    # in BHDN (real models never use 1 head), so keep it as-is.
    if d2 >= 2 and d2 < d1:
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        return q, k, v, True
    return q, k, v, False


def _resolve_scale(q: mx.array, scale: float | None) -> float:
    return scale if scale is not None else q.shape[-1] ** -0.5


def flash_attention_impl(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float | None = None,
    mask: mx.array | None = None,
    causal: bool = False,
    window_size: int = 0,
    softcap: float = 0.0,
    return_lse: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    q, k, v, transposed = _normalize_qkv_layout(q, k, v)
    scale = _resolve_scale(q, scale)

    if _HAS_MFA_EXT:
        try:
            result = _MFA_EXT.flash_attention(
                q,
                k,
                v,
                scale=scale,
                mask=mask,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
            )
            if return_lse:
                out = result
                scores = mx.matmul(q, mx.swapaxes(k, -2, -1)) * scale
                if mask is not None:
                    scores = scores + mask
                lse = mx.logsumexp(scores, axis=-1, keepdims=True)
                return out, lse.squeeze(-1)
            if transposed:
                result = mx.transpose(result, (0, 2, 1, 3))
            return result
        except Exception as exc:
            logger.debug("MFA forward failed (%s), falling back to MLX SDPA", exc)

    return _mlx_sdpa_fallback(
        q, k, v, scale, mask, causal, window_size, return_lse, transposed
    )


def _mlx_sdpa_fallback(
    q,
    k,
    v,
    scale,
    mask,
    causal,
    window_size,
    return_lse,
    transposed,
):
    mask = _resolve_mask(mask, causal, window_size, q, k)

    if q.ndim != 4:
        raise ValueError(f"Expected 4D input, got {q.ndim}D")

    logger.debug("using mx.fast.sdpa fallback (transposed=%s)", transposed)
    out = mx.fast.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        mask=mask,
    )

    if return_lse:
        scores = mx.matmul(q, mx.swapaxes(k, -2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        lse = mx.logsumexp(scores, axis=-1, keepdims=True)
        if transposed:
            out = mx.transpose(out, (0, 2, 1, 3))
        return out, lse.squeeze(-1)

    if transposed:
        out = mx.transpose(out, (0, 2, 1, 3))
    return out


def _resolve_mask(mask, causal, window_size, q, k):
    if mask is not None:
        return mask
    if causal:
        return make_causal_mask(q.shape[-2], k.shape[-2], dtype=q.dtype)
    if window_size > 0:
        return make_sliding_window_mask(
            q.shape[-2], k.shape[-2], window_size, causal=False, dtype=q.dtype
        )
    return None


def flash_attention_kvcache_impl(
    q: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    cache: object,
    *,
    scale: float | None = None,
    mask: mx.array | None = None,
    causal: bool = True,
) -> mx.array:
    from .kv_cache import KVCacheProtocol

    if not isinstance(cache, KVCacheProtocol):
        raise TypeError(f"Expected KVCacheProtocol, got {type(cache)}")

    cache.append(k_new, v_new)

    k_full = cache.k_for_attention()
    v_full = cache.v_for_attention()

    return flash_attention_impl(
        q, k_full, v_full, scale=scale, mask=mask, causal=causal
    )


def flash_attention_paged_impl(
    q: mx.array,
    k_pages: mx.array,
    v_pages: mx.array,
    block_table: mx.array,
    *,
    scale: float | None = None,
    causal: bool = False,
) -> mx.array:
    scale = _resolve_scale(q, scale)

    if _HAS_MFA_EXT and hasattr(_MFA_EXT, "flash_attention_paged"):
        try:
            return _MFA_EXT.flash_attention_paged(
                q,
                k_pages,
                v_pages,
                block_table,
                scale=scale,
                causal=causal,
            )
        except Exception as exc:
            logger.debug("MFA paged failed (%s), falling back to gather+SDPA", exc)

    k = _gather_pages(k_pages, block_table)
    v = _gather_pages(v_pages, block_table)
    return flash_attention_impl(q, k, v, scale=scale, causal=causal)


def _gather_pages(pages: mx.array, block_table: mx.array) -> mx.array:
    num_blocks = block_table.shape[0]
    blocks = []
    for i in range(num_blocks):
        page_idx = int(block_table[i])
        blk = pages[page_idx : page_idx + 1, :, :, :]
        blocks.append(blk)
    return mx.concatenate(blocks, axis=-2)


def flash_attention_varlen_impl(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
    max_seq_len: int,
    *,
    scale: float | None = None,
) -> mx.array:
    scale = _resolve_scale(q, scale)

    if _HAS_MFA_EXT and hasattr(_MFA_EXT, "flash_attention_varlen"):
        try:
            return _MFA_EXT.flash_attention_varlen(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seq_len,
                scale=scale,
            )
        except Exception as exc:
            logger.debug("MFA varlen failed (%s), falling back to per-seq", exc)

    batch_size = cu_seqlens_q.shape[0] - 1
    outputs = []
    for i in range(batch_size):
        q_start = int(cu_seqlens_q[i])
        q_end = int(cu_seqlens_q[i + 1])
        k_start = int(cu_seqlens_k[i])
        k_end = int(cu_seqlens_k[i + 1])
        qi = q[q_start:q_end]
        ki = k[k_start:k_end]
        vi = v[k_start:k_end]
        qi = qi[None]
        ki = ki[None]
        vi = vi[None]
        out = mx.fast.scaled_dot_product_attention(qi, ki, vi, scale=scale)
        outputs.append(out[0])
    return mx.concatenate(outputs, axis=0)


def sage_attention_impl(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    block_mask: mx.array,
    *,
    scale: float | None = None,
) -> mx.array:
    scale = _resolve_scale(q, scale)

    if _HAS_MFA_EXT and hasattr(_MFA_EXT, "sage_attention"):
        try:
            return _MFA_EXT.sage_attention(q, k, v, block_mask, scale=scale)
        except Exception as exc:
            logger.debug("MFA SAGE failed (%s), falling back to dense SDPA", exc)

    block_size = 64
    n_blocks_q = block_mask.shape[0]
    n_blocks_k = block_mask.shape[1]
    seq_len_q = q.shape[-2]
    seq_len_k = k.shape[-2]

    dense = mx.full((seq_len_q, seq_len_k), -float("inf"), dtype=q.dtype)
    for bq in range(n_blocks_q):
        for bk in range(n_blocks_k):
            if block_mask[bq, bk] > 0:
                i0 = bq * block_size
                i1 = min(i0 + block_size, seq_len_q)
                j0 = bk * block_size
                j1 = min(j0 + block_size, seq_len_k)
                dense[i0:i1, j0:j1] = 0.0

    dense = dense.reshape(1, 1, seq_len_q, seq_len_k)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=dense)
