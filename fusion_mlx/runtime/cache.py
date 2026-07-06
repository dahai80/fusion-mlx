# SPDX-License-Identifier: Apache-2.0
import hashlib
import logging
import os
import time

from ..config import get_config

logger = logging.getLogger(__name__)

_DEFAULT_SHUTDOWN_BUDGET_SEC = 3.5
_COMMIT_HEADROOM_SEC = 0.4


def _shutdown_budget_sec() -> float:
    raw = os.environ.get("FUSION_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET")
    if raw is None:
        return _DEFAULT_SHUTDOWN_BUDGET_SEC
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "[lifespan] invalid FUSION_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET=%r, "
            "falling back to default %ss",
            raw,
            _DEFAULT_SHUTDOWN_BUDGET_SEC,
        )
        return _DEFAULT_SHUTDOWN_BUDGET_SEC


def load_prefix_cache_from_disk() -> None:
    cfg = get_config()
    engine = _get_engine(cfg)
    if engine is None:
        return
    try:
        d = get_cache_dir()
        logger.info("[lifespan] Loading prefix cache from %s", d)
        loaded = engine.load_cache_from_disk(d)
        if loaded > 0:
            logger.info("[lifespan] Loaded %d prefix cache entries", loaded)
        else:
            logger.info("[lifespan] No prefix cache entries found on disk")
        _load_radix_index_after_cache(engine, d)
    except Exception as e:
        logger.warning("[lifespan] Failed to load cache from disk: %s", e, exc_info=True)


def _load_radix_index_after_cache(engine, cache_dir: str) -> None:
    cache = _resolve_memory_aware_cache(engine)
    if cache is None:
        return
    radix = getattr(cache, "_radix_index", None)
    if radix is None:
        return
    radix_path = os.path.join(cache_dir, "radix.index")
    if radix.load(radix_path):
        return
    try:
        with cache._lock:  # noqa: SLF001
            keys = list(cache._entries.keys())  # noqa: SLF001
        if keys:
            radix.rebuild_from_keys(keys)
            logger.info("[radix] rebuilt index from %d loaded cache entries", len(keys))
    except Exception as e:
        logger.warning("[radix] rebuild_from_keys failed: %s", e, exc_info=True)


def _resolve_memory_aware_cache(engine):
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is None:
        return None
    return getattr(scheduler, "memory_aware_cache", None)


def save_prefix_cache_to_disk(budget_sec: float | None = None) -> None:
    cfg = get_config()
    engine = _get_engine(cfg)
    if engine is None:
        return
    if budget_sec is None:
        budget_sec = _shutdown_budget_sec()
    should_abort = _make_should_abort(budget_sec) if budget_sec > 0 else None
    try:
        d = get_cache_dir()
        if should_abort is not None:
            logger.info(
                "[lifespan] Saving prefix cache to %s "
                "(shutdown budget %.1fs, commit headroom %.1fs)",
                d,
                budget_sec,
                _COMMIT_HEADROOM_SEC,
            )
        else:
            logger.info("[lifespan] Saving prefix cache to %s (no shutdown budget)", d)
        saved = _call_save_cache_to_disk(engine, d, should_abort)
        if saved:
            logger.info("[lifespan] Saved prefix cache to %s", d)
        else:
            logger.info("[lifespan] No cache to save")
        _save_radix_index_after_cache(engine, d)
    except Exception as e:
        logger.warning("[lifespan] Failed to save cache to disk: %s", e, exc_info=True)


def _save_radix_index_after_cache(engine, cache_dir: str) -> None:
    cache = _resolve_memory_aware_cache(engine)
    if cache is None:
        return
    radix = getattr(cache, "_radix_index", None)
    if radix is None:
        return
    try:
        radix.save(os.path.join(cache_dir, "radix.index"))
    except Exception as e:
        logger.warning("[radix] save failed: %s", e, exc_info=True)


def _make_should_abort(budget_sec: float):
    deadline = time.monotonic() + budget_sec
    safe_deadline = deadline - _COMMIT_HEADROOM_SEC

    def predicate(predicted_sec: float = 0.0) -> bool:
        return time.monotonic() + predicted_sec >= safe_deadline

    return predicate


def _call_save_cache_to_disk(engine, cache_dir: str, should_abort):
    import inspect

    try:
        sig = inspect.signature(engine.save_cache_to_disk)
    except (TypeError, ValueError):
        return engine.save_cache_to_disk(cache_dir, should_abort=should_abort)

    accepts_should_abort = "should_abort" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_should_abort:
        return engine.save_cache_to_disk(cache_dir, should_abort=should_abort)

    logger.warning(
        "[lifespan] engine.save_cache_to_disk does not accept "
        "should_abort kwarg — calling legacy signature "
        "(no deadline awareness for this engine)"
    )
    return engine.save_cache_to_disk(cache_dir)


def _get_engine(cfg):
    """Resolve the engine instance from config or server state.

    Rapid-MLX stores cfg.engine directly. Fusion-mlx uses an EnginePool
    accessed via _server_state. Try both paths for robustness.
    """
    engine = getattr(cfg, "engine", None)
    if engine is not None:
        return engine
    try:
        from ..service.helpers import _server_state

        pool = _server_state.get("engine_pool")
        if pool is not None:
            engines = pool.get_loaded_model_ids()
            if engines:
                return pool.get_engine(engines[0])
    except Exception:
        pass
    return None


def get_cache_dir() -> str:
    cfg = get_config()
    model_name = cfg.scheduler.model_name or "default"
    raw = str(model_name)
    safe_name = (
        raw.replace("/", "--").replace("\\", "--").replace("..", "--").lstrip(".")
    ) or "default"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    leaf = f"{safe_name}--{digest}"
    return os.path.join(
        os.path.expanduser("~"), ".cache", "fusion-mlx", "prefix_cache", leaf
    )
