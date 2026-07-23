# SPDX-License-Identifier: Apache-2.0
# UMA Radix Latent cache - image/video latent reuse for diffusion I2V/V2V.
#
# Extends the #178 DiffusionRadixCache from text-KV to image latents: on
# Apple Silicon UMA a VAE-encoded first-frame latent already lives on GPU,
# so a repeat I2V request with the same image+resolution reuses the cached
# mx.array via zero-copy pointer share and skips the full VAE load+forward.
#
# Milestone #2 (UMA Radix Latent cache), Phase-1 (image latent cache) +
# Phase-2 (session tail-frame latent cache for multi-shot reuse).

import hashlib
import logging
import os

from fusion_mlx.cache.radix_diffusion_cache import DiffusionRadixCache

logger = logging.getLogger(__name__)

_DEFAULT_MAX_MB = 2048

_IMAGE_LATENT_CACHES: "dict[str, DiffusionRadixCache]" = {}
_SESSION_TAIL_CACHE: DiffusionRadixCache | None = None


def latent_cache_enabled() -> bool:
    return os.getenv("FUSION_LATENT_CACHE", "1") == "1"


def session_tail_cache_enabled() -> bool:
    return latent_cache_enabled() and os.getenv("FUSION_SESSION_TAIL_CACHE", "1") == "1"


def latent_cache_max_mb() -> int:
    raw = os.getenv("FUSION_LATENT_CACHE_MAX_MB", str(_DEFAULT_MAX_MB))
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "invalid FUSION_LATENT_CACHE_MAX_MB=%r, using %d", raw, _DEFAULT_MAX_MB
        )
        return _DEFAULT_MAX_MB
    if val <= 0:
        return _DEFAULT_MAX_MB
    return val


def image_latent_key(model_id, image_source, height, width, dtype) -> str:
    src = image_source if isinstance(image_source, str) else str(image_source)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return f"latent:{model_id}:{height}x{width}:{dtype}:{digest}"


def get_image_latent_cache(model_id, max_mb=None):
    if not latent_cache_enabled():
        logger.debug("latent cache disabled (FUSION_LATENT_CACHE=0)")
        return None
    cached = _IMAGE_LATENT_CACHES.get(model_id)
    if cached is not None:
        return cached
    mb = max_mb if max_mb is not None else latent_cache_max_mb()
    cache = DiffusionRadixCache(max_mb=mb, name=f"latent:{model_id}")
    _IMAGE_LATENT_CACHES[model_id] = cache
    logger.info("latent cache created: model=%s max_mb=%d", model_id, mb)
    return cache


def session_tail_key(session_id: str, model_id: str) -> str:
    return f"session_tail:{model_id}:{session_id}"


def get_session_tail_cache() -> DiffusionRadixCache | None:
    global _SESSION_TAIL_CACHE
    if not session_tail_cache_enabled():
        logger.debug("session tail cache disabled (FUSION_SESSION_TAIL_CACHE=0)")
        return None
    if _SESSION_TAIL_CACHE is not None:
        return _SESSION_TAIL_CACHE
    _SESSION_TAIL_CACHE = DiffusionRadixCache(
        max_mb=latent_cache_max_mb(), name="session_tail"
    )
    logger.info("session tail cache created: max_mb=%d", latent_cache_max_mb())
    return _SESSION_TAIL_CACHE


def put_session_tail(session_id: str, model_id: str, latents) -> bool:
    cache = get_session_tail_cache()
    if cache is None:
        return False
    key = session_tail_key(session_id, model_id)
    cache.put(key, latents)
    logger.info("session tail put: %s", key)
    return True


def get_session_tail(session_id: str, model_id: str):
    cache = get_session_tail_cache()
    if cache is None:
        return None
    key = session_tail_key(session_id, model_id)
    result = cache.get(key)
    if result is not None:
        logger.info("session tail hit: %s", key)
    else:
        logger.debug("session tail miss: %s", key)
    return result
