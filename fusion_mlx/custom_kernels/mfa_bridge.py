# SPDX-License-Identifier: Apache-2.0
# MFA bridge - unified entry point for Flash Attention on Apple Silicon.
# Auto-dispatches to mlx_mfa (Metal kernel) or mx.fast.scaled_dot_product_attention (fallback).

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from .mfa import is_mfa_available
from .mfa.attention import _normalize_qkv_layout
from .mfa.dispatch_policy import (
    AttentionBackend,
    DispatchDecision,
    get_device_info,
    get_supported_head_dims,
    select_backend,
    supports_backward,
    warmup_kernels,
)
from .mfa.masks import (
    make_causal_mask,
    make_sliding_window_mask,
    mask_from_attn_mask,
)

logger = logging.getLogger(__name__)

__all__ = [
    "flash_attention",
    "flash_attention_kvcache",
    "flash_attention_paged",
    "flash_attention_varlen",
    "sage_attention",
    "is_available",
    "get_device_info",
    "select_backend",
    "supports_backward",
    "get_supported_head_dims",
    "warmup_kernels",
    "make_causal_mask",
    "make_sliding_window_mask",
]


def is_available() -> bool:
    return is_mfa_available()


def _resolve_scale(q: mx.array, scale: float | None) -> float:
    if scale is not None:
        return scale
    return q.shape[-1] ** -0.5


def flash_attention(
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
    scale = _resolve_scale(q, scale)
    head_dim = q.shape[-1]
    _, _, seq_len_q, _ = q.shape if q.ndim == 4 else (1, 1, q.shape[-2], q.shape[-1])
    is_decode = seq_len_q == 1
    batch_size = q.shape[0] if q.ndim == 4 else 1

    decision: DispatchDecision = select_backend(
        head_dim=head_dim,
        dtype=q.dtype,
        causal=causal,
        window_size=window_size,
        seq_len_q=seq_len_q,
        seq_len_kv=k.shape[-2] if k.ndim == 4 else k.shape[-3],
        batch_size=batch_size,
        is_decode=is_decode,
        has_mask=mask is not None,
    )

    logger.debug("selected backend: %s (%s)", decision.backend.name, decision.reason)

    if decision.backend in (
        AttentionBackend.STEEL,
        AttentionBackend.STEEL_DSPLIT,
        AttentionBackend.NAX,
    ):
        return _mfa_dispatch(
            q, k, v, scale, mask, causal, window_size, softcap, return_lse
        )
    else:
        return _mlx_sdpa_fallback(
            q, k, v, scale, mask, causal, window_size, softcap, return_lse
        )


def _mfa_dispatch(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    mask: mx.array | None,
    causal: bool,
    window_size: int,
    softcap: float,
    return_lse: bool,
) -> mx.array | tuple[mx.array, mx.array]:
    if not is_mfa_available():
        logger.debug("MFA ext not available, falling back to MLX SDPA")
        return _mlx_sdpa_fallback(
            q, k, v, scale, mask, causal, window_size, softcap, return_lse
        )

    from .mfa.attention import flash_attention_impl

    return flash_attention_impl(
        q,
        k,
        v,
        scale=scale,
        mask=mask,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        return_lse=return_lse,
    )


def _mlx_sdpa_fallback(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    mask: mx.array | None,
    causal: bool,
    window_size: int,
    softcap: float,
    return_lse: bool,
) -> mx.array | tuple[mx.array, mx.array]:
    q, k, v, transposed = _normalize_qkv_layout(q, k, v)
    logger.debug("using mx.fast.sdpa fallback (transposed=%s)", transposed)

    if mask is None and causal:
        mask = make_causal_mask(q.shape[-2], k.shape[-2], dtype=q.dtype)
    elif mask is not None:
        mask = mask_from_attn_mask(
            mask,
            q.shape[-2],
            k.shape[-2],
            q.shape[-3] if q.ndim == 4 else 1,
            q.shape[0] if q.ndim == 4 else 1,
        )

    if window_size > 0 and mask is None and not causal:
        mask = make_sliding_window_mask(
            q.shape[-2], k.shape[-2], window_size, causal=False, dtype=q.dtype
        )

    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

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


def flash_attention_kvcache(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cache: Any,
    *,
    scale: float | None = None,
    mask: mx.array | None = None,
    causal: bool = True,
) -> mx.array:
    from .mfa.kv_cache import KVCacheProtocol

    if not isinstance(cache, KVCacheProtocol):
        raise TypeError(f"Expected KVCacheProtocol, got {type(cache)}")

    scale = _resolve_scale(q, scale)
    cache.append(k, v)
    k_full = cache.k_for_attention()
    v_full = cache.v_for_attention()

    return flash_attention(q, k_full, v_full, scale=scale, mask=mask, causal=causal)


def flash_attention_paged(
    q: mx.array,
    k_pages: mx.array,
    v_pages: mx.array,
    block_table: mx.array,
    *,
    scale: float | None = None,
    causal: bool = False,
) -> mx.array:
    from .mfa.attention import flash_attention_paged_impl

    return flash_attention_paged_impl(
        q, k_pages, v_pages, block_table, scale=scale, causal=causal
    )


def flash_attention_varlen(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
    max_seq_len: int,
    *,
    scale: float | None = None,
) -> mx.array:
    from .mfa.attention import flash_attention_varlen_impl

    return flash_attention_varlen_impl(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seq_len, scale=scale
    )


def sage_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    block_mask: mx.array,
    *,
    scale: float | None = None,
) -> mx.array:
    from .mfa.attention import sage_attention_impl

    return sage_attention_impl(q, k, v, block_mask, scale=scale)
