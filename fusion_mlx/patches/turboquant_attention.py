# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to support TurboQuantKVCache.

When TurboQuantKVCache is detected, routes attention to:
  - Decode (L=1): cache.decode_attention() — Metal kernel, no dequant
  - Prefill (L>1): cache.prefill_attention() fast path, fallback to
    dequantize + mx.fast.scaled_dot_product_attention
"""

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_ORIGINAL_SDPA = None
_LONG_PREFILL_QUANTIZED_THRESHOLD = 8192
_LONG_PREFILL_QUERY_BLOCK_SIZE = 256
_LONG_PREFILL_KEY_CHUNK_SIZE = 16384

_TQ_TYPES: tuple | None = None
_TQ_TYPES_TRIED = False


def _tq_cache_types():
    # #1403: both TurboQuant cache classes live behind mlx_vlm, which a slim
    # install does not have. Resolve them once instead of per attention call.
    # Without mlx_vlm no TurboQuant cache can exist at all, so every call
    # takes the plain SDPA path — a text-model DPO/ORPO/reward job (they run
    # in-process, unlike SFT) must never require the [vlm] extra.
    global _TQ_TYPES, _TQ_TYPES_TRIED
    if not _TQ_TYPES_TRIED:
        _TQ_TYPES_TRIED = True
        try:
            from mlx_vlm.turboquant import TurboQuantKVCache

            from ..turboquant_kv import (
                BatchTurboQuantKVCache,
                _state_length,
            )

            _TQ_TYPES = (TurboQuantKVCache, BatchTurboQuantKVCache, _state_length)
        except ImportError:
            _TQ_TYPES = ()
            logger.info(
                "turboquant attention patch: mlx_vlm not installed, "
                "TurboQuant caches unavailable — attention stays on stock SDPA"
            )
    return _TQ_TYPES


def apply_turboquant_attention_patch() -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for TurboQuant."""
    global _PATCHED, _ORIGINAL_SDPA
    if _PATCHED:
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    # _ORIGINAL_SDPA survives re-application after a test resets _PATCHED,
    # so the rebind loop below still sees the true stock function even when
    # mlx_base.scaled_dot_product_attention is already the patched one.
    if _ORIGINAL_SDPA is None:
        _ORIGINAL_SDPA = mlx_base.scaled_dot_product_attention
    original_sdpa = _ORIGINAL_SDPA

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: mx.array | None,
        sinks: mx.array | None = None,
    ) -> mx.array:
        tq = _tq_cache_types()
        if not tq:
            return original_sdpa(queries, keys, values, cache, scale, mask, sinks)
        _TQCache, BatchTurboQuantKVCache, _state_length = tq

        # Detect underlying TQ cache (may be wrapped by proxy objects)
        real_cache = cache
        if hasattr(cache, "_cache") and not isinstance(
            cache, (_TQCache, BatchTurboQuantKVCache)
        ):
            real_cache = cache._cache

        if isinstance(real_cache, (_TQCache, BatchTurboQuantKVCache)):
            if sinks is not None:
                # TurboQuant's quantized kernels do not implement attention
                # sinks. Preserve correctness by falling back to MLX's
                # sink-aware SDPA over dequantized states.
                dequantized_keys, dequantized_values = real_cache.dequantize(
                    keys_state=keys,
                    values_state=values,
                )
                return mx.fast.scaled_dot_product_attention(
                    queries,
                    dequantized_keys.astype(queries.dtype),
                    dequantized_values.astype(queries.dtype),
                    scale=scale,
                    mask=mask,
                    sinks=sinks,
                )
            if queries.shape[-2] == 1:
                # Decode (B=1 and B>1). Continuous-batching decode passes a
                # per-request left-padding array mask; the masked decode_attention
                # path runs the quantized kernels directly (no full-batch
                # dequantize per step). The RHT masked-decode fix landed upstream
                # in mlx-vlm (Blaizzy/mlx-vlm#1244, in the pinned commit).
                return real_cache.decode_attention(
                    queries,
                    keys_state=keys,
                    values_state=values,
                    scale=scale,
                    mask=mask,
                )
            # Prefill: try quantized fast path, fallback to dequantize+SDPA
            result = real_cache.prefill_attention(
                queries,
                keys_state=keys,
                values_state=values,
                scale=scale,
                mask=mask,
            )
            if result is not None:
                return result
            keys_state = getattr(keys, "_state", keys)
            try:
                total_tokens = _state_length(keys_state)
            except Exception:
                total_tokens = 0
            if total_tokens > _LONG_PREFILL_QUANTIZED_THRESHOLD and hasattr(
                real_cache, "quantized_attention"
            ):
                old_query_block_size = getattr(
                    real_cache, "prefill_query_block_size", None
                )
                old_key_chunk_size = getattr(real_cache, "prefill_key_chunk_size", None)
                try:
                    real_cache.prefill_query_block_size = _LONG_PREFILL_QUERY_BLOCK_SIZE
                    real_cache.prefill_key_chunk_size = _LONG_PREFILL_KEY_CHUNK_SIZE
                    return real_cache.quantized_attention(
                        queries,
                        keys_state=keys,
                        values_state=values,
                        scale=scale,
                        mask=mask,
                    )
                except Exception:
                    logger.debug(
                        "TurboQuant quantized prefill attention failed; "
                        "falling back to dequantize+SDPA",
                        exc_info=True,
                    )
                finally:
                    if old_query_block_size is not None:
                        real_cache.prefill_query_block_size = old_query_block_size
                    if old_key_chunk_size is not None:
                        real_cache.prefill_key_chunk_size = old_key_chunk_size
            dequantized_keys, dequantized_values = real_cache.dequantize()
            return mx.fast.scaled_dot_product_attention(
                queries,
                dequantized_keys.astype(queries.dtype),
                dequantized_values.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )

        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Patch the module attribute
    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Also patch any model modules that already imported it locally
    # Covers both mlx_lm (LLM) and mlx_vlm (VLM) model modules
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is mlx_base:
            continue
        if not (
            mod_name.startswith("mlx_lm.models.")
            or mod_name.startswith("mlx_vlm.models.")
        ):
            # Remote-code architectures (trust_remote_code, e.g. k2_horizon)
            # live outside the mlx_*.models.* namespace but bind the stock
            # SDPA by value at import time — during model load, which happens
            # before this patch in BatchedEngine.__init__. Without a rebind
            # the stock fn sees cache.bits on a TurboQuant cache and routes
            # to quantized_scaled_dot_product_attention, which subscripts the
            # _QuantizedStateProxy state → TypeError 500 on every request.
            # Only exact stock references are rebound; custom model SDPA
            # implementations are left alone.
            try:
                func = getattr(mod, "scaled_dot_product_attention", None)
            except Exception:
                continue
            if func is original_sdpa:
                mod.scaled_dot_product_attention = patched_sdpa
                logger.debug(
                    "TurboQuant attention patch: rebound stock SDPA in %s",
                    mod_name,
                )
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            func = mod.scaled_dot_product_attention
            if func is original_sdpa or func is not patched_sdpa:
                mod.scaled_dot_product_attention = patched_sdpa

    # Also patch mlx_vlm.models.base if loaded
    try:
        from mlx_vlm.models import base as vlm_base

        if hasattr(vlm_base, "scaled_dot_product_attention"):
            vlm_base.scaled_dot_product_attention = patched_sdpa
    except ImportError:
        pass

    _PATCHED = True
    logger.info("TurboQuant attention patch applied")
    return True


from .registry import register as _reg

_reg(
    "turboquant_long_prefill_chunking",
    "global",
    "Long-prefill quantized attention chunking (fusion-mlx native, not upstream)",
    lambda model, config: apply_turboquant_attention_patch(),
    is_global=True,
)
