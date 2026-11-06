# SPDX-License-Identifier: Apache-2.0
"""LSA+TQ serving memory budget hard gate (migration plan P3 item 4).

Budget contract: weights + KV + selection workspace must stay under
unified memory x 0.85 (120 GB reference machine -> ~102 GB). The existing
prefill guard (PrefillMemoryExceededError via MemoryMonitor) provides the
enforcement; what LSA serving lacked was (a) a KV bytes-per-token estimate
matching the packed TurboQuant k8v4 state - the monitor's fp16 formula
overestimates ~12x and would falsely reject long contexts - and (b) an
operator turning the guard on at all. arm_lsa_memory_budget does both.
"""

from __future__ import annotations

import logging
import os
import platform

logger = logging.getLogger(__name__)

_BUDGET_RAM_FRACTION = 0.85


def default_budget_bytes() -> int:
    # unified memory x 0.85; leave the rest to the OS and GPU driver
    if platform.system() == "Darwin":
        name = os.sysconf_names.get("HW_MEMSIZE")
        total = os.sysconf(name) if name is not None else 0
        if total <= 0:
            logger.warning("HW_MEMSIZE unavailable; falling back to 128 GB budget")
            total = 128 * 1024**3
    else:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return int(total * _BUDGET_RAM_FRACTION)


def tq_kv_bytes_per_token(num_kv_heads: int, head_dim: int, kv_layers: int) -> int:
    # packed k8v4 state per token per layer: K indices 1 B/elem (8-bit) +
    # V indices 0.5 B/elem (4-bit), plus one fp16 norm per head for each of
    # K and V. Linear-attention (ArraysCache) layers carry fixed-size
    # recurrent state, not per-token KV - excluded by construction because
    # kv_layers counts only full-attention layers.
    per_layer = num_kv_heads * head_dim * 1.5 + num_kv_heads * 2 * 2
    return int(per_layer * kv_layers)


def _cfg_get(config, key):
    # model.args carries the HF config.json as plain dicts (text_config is
    # a dict of dicts), while model.config exposes attributes - support both
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _extract_dims(model) -> tuple[int, int, int, int] | None:
    # Same config probing as sched_misc._set_model_info_for_monitor, minus
    # the monitor: nested VLM text_config, head_dim derivation, and the
    # KVCache layer count via make_cache (hybrid models carry fewer
    # full-attention layers than total layers).
    config = getattr(model, "config", None) or getattr(model, "args", None)
    if config is None:
        return None
    for sub_attr in ("text_config", "language_config", "llm_config"):
        sub = _cfg_get(config, sub_attr)
        if sub is not None and (
            _cfg_get(sub, "num_hidden_layers") or _cfg_get(sub, "n_layer")
        ):
            config = sub
            break
    num_layers = _cfg_get(config, "num_hidden_layers") or _cfg_get(config, "n_layer")
    num_kv_heads = (
        _cfg_get(config, "num_key_value_heads")
        or _cfg_get(config, "num_attention_heads")
        or _cfg_get(config, "n_head")
    )
    head_dim = _cfg_get(config, "head_dim")
    if head_dim is None:
        hidden = _cfg_get(config, "hidden_size")
        if hidden and num_kv_heads:
            head_dim = hidden // num_kv_heads
    if not (num_layers and num_kv_heads and head_dim):
        return None
    kv_layers = num_layers
    make_cache = getattr(model, "make_cache", None)
    if make_cache is not None:
        try:
            from mlx_lm.models.cache import CacheList, KVCache

            def _count(c) -> int:
                if type(c) is KVCache:
                    return 1
                if isinstance(c, CacheList):
                    return sum(_count(inner) for inner in c.caches)
                return 0

            n = sum(_count(c) for c in make_cache())
            if n > 0:
                kv_layers = n
        except Exception as e:
            logger.debug("make_cache KV layer count failed (%s); using num_layers", e)
    return int(num_layers), int(num_kv_heads), int(head_dim), int(kv_layers)


def arm_lsa_memory_budget(scheduler, weights_bytes: int = 0) -> dict | None:
    # Feed the TQ-accurate KV estimate to the monitor, arm the prefill
    # guard, and set the doc ceiling unless a memory enforcer already did.
    # Returns a summary dict for tests/logs, None when there is nothing to
    # arm (no monitor and no extractable model dims).
    monitor = getattr(scheduler, "memory_monitor", None)
    if monitor is None:
        logger.warning("LSA memory budget not armed: memory monitor missing")
        return None
    if not monitor.has_model_info():
        dims = _extract_dims(getattr(scheduler, "model", None))
        if dims is None:
            _model = getattr(scheduler, "model", None)
            _args = getattr(_model, "args", None)
            logger.warning(
                "LSA budget dim probe: model=%r has_config=%s has_args=%s "
                "args_keys=%s",
                type(_model).__name__,
                hasattr(_model, "config"),
                hasattr(_model, "args"),
                sorted(vars(_args).keys())[:12]
                if _args is not None
                else None,
            )
            logger.warning(
                "LSA memory budget not armed: no monitor model info and no "
                "extractable model config"
            )
            return None
        monitor.set_model_info(
            num_layers=dims[0],
            num_kv_heads=dims[1],
            head_dim=dims[2],
            num_kv_cache_layers=dims[3],
        )

    bpt = tq_kv_bytes_per_token(
        monitor._num_kv_heads, monitor._head_dim, monitor._num_kv_cache_layers
    )
    monitor.set_model_info(
        num_layers=monitor._num_layers,
        head_dim=monitor._head_dim,
        num_kv_heads=monitor._num_kv_heads,
        num_attention_heads=monitor._num_attention_heads,
        dtype_size=monitor._dtype_size,
        num_kv_cache_layers=monitor._num_kv_cache_layers,
        compute_dtype_size=monitor._score_dtype_size,
        kv_bytes_per_token=bpt,
    )

    ceiling = int(getattr(scheduler, "_memory_hard_limit_bytes", 0) or 0)
    if ceiling <= 0:
        ceiling = default_budget_bytes()
        scheduler._memory_hard_limit_bytes = ceiling
    scheduler._prefill_memory_guard = True

    max_ctx = max(0, int((ceiling - max(0, weights_bytes)) // bpt))
    logger.info(
        "LSA memory budget armed: ceiling %.1f GB, KV ~%d B/token, "
        "weights %.1f GB, max context ~%d tokens",
        ceiling / 1024**3,
        bpt,
        max(0, weights_bytes) / 1024**3,
        max_ctx,
    )
    return {
        "ceiling_bytes": ceiling,
        "kv_bytes_per_token": bpt,
        "weights_bytes": max(0, weights_bytes),
        "max_context_tokens": max_ctx,
    }
