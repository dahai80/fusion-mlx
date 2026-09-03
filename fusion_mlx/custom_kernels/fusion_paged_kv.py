from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .paged_kv_cache import FusionPagedKVCache
from .paged_kv_pool import FusionPagedKVPool, FusionPagedRequestCache

logger = logging.getLogger(__name__)

_INSTALLED_ATTR = "_fusion_paged_kv_installed"
_CACHE_REGISTRY_ATTR = "_fusion_paged_kv_registry"
_POOL_ATTR = "_fusion_paged_pool"
_POOL_SEQ_ATTR = "_fusion_paged_pool_seq"

_GLOBAL_CACHE_REGISTRY: dict = {}


def is_paged_kv_available() -> bool:
    try:
        from .glm_moe_dsa.fast import is_native_available

        return is_native_available()
    except Exception:
        return False


def _infer_kv_geometry(model: nn.Module) -> tuple[int, int]:
    args = getattr(model, "args", None)
    if args is None:
        logger.warning(
            "install_paged_kv: model.args missing, using placeholder "
            "n_kv_heads=1 head_dim=1 for pool geometry"
        )
        return 1, 1
    n_kv_heads = int(
        getattr(args, "n_kv_heads", None)
        or getattr(args, "num_key_value_heads", None)
        or 1
    )
    head_dim = getattr(args, "head_dim", None)
    if head_dim is None:
        n_heads = int(
            getattr(args, "n_heads", None)
            or getattr(args, "num_attention_heads", None)
            or 1
        )
        dim = int(getattr(args, "dim", None) or getattr(args, "hidden_size", None) or 1)
        head_dim = dim // n_heads if n_heads else 1
    return n_kv_heads, int(head_dim)


def _infer_model_dtype(model: nn.Module) -> Any:
    try:
        leaves = nn.utils.tree_flatten(model.parameters())
        for leaf in leaves:
            dt = getattr(leaf, "dtype", None)
            if dt is not None:
                return dt
    except Exception as e:
        logger.debug("install_paged_kv: dtype inference failed (%s)", e)
    return mx.float32


def install_paged_kv(model: nn.Module, config: Any) -> None:
    block_size = int(getattr(config, "paged_kv_block_size", 16))
    num_blocks = int(getattr(config, "paged_kv_num_blocks", 256))
    pool_enabled = bool(getattr(config, "pool_enabled", False))
    pool_num_blocks = int(getattr(config, "pool_num_blocks", 256))
    logger.info(
        "install_paged_kv: block_size=%s num_blocks=%s pool_enabled=%s "
        "pool_num_blocks=%s native=%s model=%s",
        block_size,
        num_blocks,
        pool_enabled,
        pool_num_blocks,
        is_paged_kv_available(),
        type(model).__name__,
    )
    try:
        setattr(model, _INSTALLED_ATTR, True)
        setattr(model, _CACHE_REGISTRY_ATTR, {})
        if pool_enabled:
            n_kv_heads, head_dim = _infer_kv_geometry(model)
            model_dtype = _infer_model_dtype(model)
            pool = FusionPagedKVPool(
                block_size=block_size,
                num_blocks=pool_num_blocks,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                dtype=model_dtype,
            )
            setattr(model, _POOL_ATTR, pool)
            setattr(model, _POOL_SEQ_ATTR, 0)

            def _fusion_make_cache_pool():
                pool_obj = getattr(model, _POOL_ATTR, None)
                if pool_obj is None:
                    n_kv_heads_l, head_dim_l = _infer_kv_geometry(model)
                    model_dtype_l = _infer_model_dtype(model)
                    pool_obj = FusionPagedKVPool(
                        block_size=block_size,
                        num_blocks=pool_num_blocks,
                        n_kv_heads=n_kv_heads_l,
                        head_dim=head_dim_l,
                        dtype=model_dtype_l,
                    )
                    setattr(model, _POOL_ATTR, pool_obj)
                    logger.info(
                        "install_paged_kv: pool lazily constructed cap=%d",
                        pool_num_blocks,
                    )
                seq = getattr(model, _POOL_SEQ_ATTR, 0)
                request_id = f"pool_{seq}"
                setattr(model, _POOL_SEQ_ATTR, seq + 1)
                num_layers = _detect_num_layers(model)
                handles = [
                    FusionPagedRequestCache(pool_obj, request_id)
                    for _ in range(num_layers)
                ]
                _GLOBAL_CACHE_REGISTRY[request_id] = handles
                logger.info(
                    "install_paged_kv: pool make_cache request_id=%s layers=%d",
                    request_id,
                    num_layers,
                )
                return handles

            model.make_cache = _fusion_make_cache_pool
            logger.info(
                "install_paged_kv: pool mode installed cap=%d block_size=%d",
                pool_num_blocks,
                block_size,
            )
        else:

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
            logger.info(
                "install_paged_kv: make_cache bound (%d layers)",
                _detect_num_layers(model),
            )
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
