# SPDX-License-Identifier: Apache-2.0
# MFA (Multi-Head Flash Attention) - Metal-accelerated attention for Apple Silicon.
# Bridges mlx_mfa (if installed) into fusion-mlx, or provides pure-MLX fallback.

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

_HAS_MFA_EXT: bool = False
_MFA_EXT: Any = None

try:
    _MFA_EXT = importlib.import_module("mlx_mfa")
    _HAS_MFA_EXT = True
    logger.info(
        "mlx_mfa v%s available - full Metal kernel acceleration enabled",
        getattr(_MFA_EXT, "__version__", "?"),
    )
except ImportError:
    _HAS_MFA_EXT = False
    logger.info(
        "mlx_mfa not installed - using mx.fast.scaled_dot_product_attention fallback"
    )
except Exception as exc:
    _HAS_MFA_EXT = False
    logger.warning("mlx_mfa import failed (%s); using fallback", exc)


def is_mfa_available() -> bool:
    return _HAS_MFA_EXT


def get_mfa_version() -> str | None:
    if _MFA_EXT is not None:
        return getattr(_MFA_EXT, "__version__", None)
    return None


def flash_attention(q, k, v, scale=None, mask=None, causal=False, **kwargs):
    from .attention import flash_attention_impl

    return flash_attention_impl(
        q, k, v, scale=scale, mask=mask, causal=causal, **kwargs
    )


def flash_attention_kvcache(q, k, v, cache, scale=None, mask=None, causal=True):
    from .attention import flash_attention_kvcache_impl

    return flash_attention_kvcache_impl(
        q, k, v, cache, scale=scale, mask=mask, causal=causal
    )


def flash_attention_paged(q, k_pages, v_pages, block_table, scale=None, causal=False):
    from .attention import flash_attention_paged_impl

    return flash_attention_paged_impl(
        q, k_pages, v_pages, block_table, scale=scale, causal=causal
    )


def flash_attention_varlen(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seq_len, scale=None
):
    from .attention import flash_attention_varlen_impl

    return flash_attention_varlen_impl(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seq_len, scale=scale
    )


def sage_attention(q, k, v, block_mask, scale=None):
    from .attention import sage_attention_impl

    return sage_attention_impl(q, k, v, block_mask, scale=scale)


def temporal_attention(*args, **kwargs):
    from .video_attention import TemporalAttention

    return TemporalAttention(*args, **kwargs)


def video_transformer_block(*args, **kwargs):
    from .video_attention import VideoTransformerBlock

    return VideoTransformerBlock(*args, **kwargs)


def fp8_linear(*args, **kwargs):
    from .fp8_linear import FP8Linear

    return FP8Linear(*args, **kwargs)


__all__ = [
    "is_mfa_available",
    "get_mfa_version",
    "flash_attention",
    "flash_attention_kvcache",
    "flash_attention_paged",
    "flash_attention_varlen",
    "sage_attention",
]
