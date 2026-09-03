from __future__ import annotations

import logging
from typing import Any

import mlx.nn as nn

from .paged_kv_cache import FusionPagedKVCache

logger = logging.getLogger(__name__)

_INSTALLED_ATTR = "_fusion_paged_kv_installed"
_CACHE_REGISTRY_ATTR = "_fusion_paged_kv_registry"

_GLOBAL_CACHE_REGISTRY: dict = {}


def is_paged_kv_available() -> bool:
    try:
        from .glm_moe_dsa.fast import is_native_available

        return is_native_available()
    except Exception:
        return False


def install_paged_kv(model: nn.Module, config: Any) -> None:
    block_size = int(getattr(config, "paged_kv_block_size", 16))
    num_blocks = int(getattr(config, "paged_kv_num_blocks", 256))
    logger.info(
        "install_paged_kv: block_size=%s num_blocks=%s native=%s model=%s",
        block_size,
        num_blocks,
        is_paged_kv_available(),
        type(model).__name__,
    )
    try:
        setattr(model, _INSTALLED_ATTR, True)
        setattr(model, _CACHE_REGISTRY_ATTR, {})
        def _fusion_make_cache():
            num_layers = _detect_num_layers(model)
            return [
                FusionPagedKVCache(
                    block_size=block_size,
                    num_blocks=num_blocks,
                )
                for _ in range(num_layers)
            ]

        model.make_cache = _fusion_make_cache
        logger.info("install_paged_kv: make_cache bound (%d layers)", _detect_num_layers(model))
    except Exception as e:
        logger.warning("install_paged_kv: bind failed (%s), passthrough", e)


def _detect_num_layers(model: nn.Module) -> int:
    for attr in ("layers", "model"):
        layers = getattr(model, attr, None)
        if layers is not None and hasattr(layers, "__len__"):
            return len(layers)
        inner = getattr(layers, "layers", None)
        if inner is not None and hasattr(inner, "__len__"):
            return len(inner)
    return 0


def uninstall_paged_kv(model: nn.Module) -> None:
    try:
        if getattr(model, _INSTALLED_ATTR, False):
            setattr(model, _INSTALLED_ATTR, False)
            if hasattr(model, "make_cache"):
                try:
                    delattr(model, "make_cache")
                except Exception:
                    pass
            logger.info("uninstall_paged_kv: cleared on %s", type(model).__name__)
    except Exception:
        pass


def register_cache(model: nn.Module, request_id: str, caches: list) -> None:
    reg = getattr(model, _CACHE_REGISTRY_ATTR, None)
    if reg is None:
        return
    reg[request_id] = caches
    _GLOBAL_CACHE_REGISTRY[request_id] = caches


def evict_request(model: nn.Module, request_id: str) -> int:
    reg = getattr(model, _CACHE_REGISTRY_ATTR, None)
    caches = None
    if reg is not None:
        caches = reg.pop(request_id, None)
    if caches is None:
        caches = _GLOBAL_CACHE_REGISTRY.pop(request_id, None)
    if not caches:
        return 0
    freed = 0
    for c in caches:
        try:
            freed += c.free_all()
        except Exception:
            pass
    logger.info("evict_request: freed %d blocks for %s", freed, request_id)
    return freed


def evict_request_by_id(request_id: str) -> int:
    caches = _GLOBAL_CACHE_REGISTRY.pop(request_id, None)
    if not caches:
        logger.debug("evict_request_by_id: no caches for %s", request_id)
        return 0
    freed = 0
    for c in caches:
        try:
            freed += c.free_all()
        except Exception:
            pass
    logger.info("evict_request_by_id: freed %d blocks for %s", freed, request_id)
    return freed


__all__ = [
    "is_paged_kv_available",
    "install_paged_kv",
    "uninstall_paged_kv",
    "FusionPagedKVCache",
    "register_cache",
    "evict_request",
    "evict_request_by_id",
]
