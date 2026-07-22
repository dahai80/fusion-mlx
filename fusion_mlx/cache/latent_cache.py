# SPDX-License-Identifier: Apache-2.0
# UMA Radix Latent cache - image/video latent reuse for diffusion I2V/V2V.
#
# Extends the #178 DiffusionRadixCache from text-KV to image latents: on
# Apple Silicon UMA a VAE-encoded first-frame latent already lives on GPU,
# so a repeat I2V request with the same image+resolution reuses the cached
# mx.array via zero-copy pointer share and skips the full VAE load+forward.
#
# Milestone #2 (UMA Radix Latent cache), Phase-1.

import hashlib
import logging
import os

from fusion_mlx.cache.radix_diffusion_cache import DiffusionRadixCache

logger = logging.getLogger(__name__)

_DEFAULT_MAX_MB = 2048

_IMAGE_LATENT_CACHES: "dict[str, DiffusionRadixCache]" = {}


def latent_cache_enabled() -> bool:
    return os.getenv("FUSION_LATENT_CACHE", "1") == "1"


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
