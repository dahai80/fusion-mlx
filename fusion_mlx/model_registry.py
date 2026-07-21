# SPDX-License-Identifier: Apache-2.0
"""Model Registry — tracks model ownership to prevent BatchKVCache conflicts."""

import logging
import os
import threading
import weakref
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelOwnershipError(Exception):
    pass


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._owners: dict[int, tuple[weakref.ref, str]] = {}
        self._registry_lock = threading.Lock()

    def acquire(
        self, model: Any, engine: Any, engine_id: str, force: bool = False
    ) -> bool:
        model_id = id(model)
        with self._registry_lock:
            if model_id in self._owners:
                weak_ref, owner_id = self._owners[model_id]
                owner = weak_ref()
                if owner is not None and owner_id != engine_id:
                    if force:
                        logger.warning(
                            f"Model ownership transfer: {owner_id} -> {engine_id}"
                        )
                        self._reset_owner(owner)
                    else:
                        raise ModelOwnershipError(
                            f"Model already owned by engine {owner_id}"
                        )
            if engine is not None:
                self._owners[model_id] = (weakref.ref(engine), engine_id)
            else:
                self._owners[model_id] = (None, engine_id)
            return True

    def release(self, model: Any, engine_id: str) -> bool:
        model_id = id(model)
        with self._registry_lock:
            if model_id in self._owners:
                _, owner_id = self._owners[model_id]
                if owner_id == engine_id:
                    del self._owners[model_id]
                    return True
        return False

    def is_owned(self, model: Any) -> tuple[bool, str | None]:
        model_id = id(model)
        with self._registry_lock:
            if model_id in self._owners:
                weak_ref, owner_id = self._owners[model_id]
                if weak_ref is None or weak_ref() is not None:
                    return (True, owner_id)
                else:
                    del self._owners[model_id]
        return (False, None)

    def _reset_owner(self, owner: Any) -> None:
        try:
            if hasattr(owner, "scheduler"):
                owner.scheduler.deep_reset()
        except Exception as e:
            logger.warning(f"Failed to reset previous owner: {e}")

    def cleanup(self) -> int:
        cleaned = 0
        with self._registry_lock:
            stale = [
                k for k, (r, _) in self._owners.items() if r is not None and r() is None
            ]
            for k in stale:
                del self._owners[k]
                cleaned += 1
        return cleaned

    def get_stats(self) -> dict[str, Any]:
        with self._registry_lock:
            active = sum(
                1 for _, (r, _) in self._owners.items() if r is None or r() is not None
            )
            return {"total_entries": len(self._owners), "active_owners": active}


_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _registry


# ---------------------------------------------------------------------------
# Model discovery registry (issue #172)
# Programmatic discovery of available image/video models + capabilities.
# Powers Fusion-ComfyUI /object_info + CLI list-models. Additive over
# pool.model_discovery (raw DiscoveredModel) - enriches with backend name +
# per-backend constraints. Coexists with the ModelRegistry ownership tracker
# above (different concern; both live here per issue #172 import path).
# ---------------------------------------------------------------------------

_IMAGE_BACKEND = "mflux"
_IMAGE_QUANTIZE_OPTIONS: list[Any] = [None, 4, 8]

# Flux2Klein rounds the image edge to the nearest 16px (see
# engines/image_gen._to_latent_size: ((h+15)//16)*16), so 16 is the alignment
# unit. 2048 is a conservative documented ceiling; the model accepts larger
# but quality/memory degrade - consumers treat this as a safe default.
_IMAGE_DIM_DIVISIBILITY = 16
_IMAGE_MAX_RESOLUTION = 2048

_VALID_MODEL_TYPES = ("image", "video")


def _registry_default_model_dirs() -> list[Path]:
    # Respect an explicit override (os.pathsep-separated for multiple dirs)
    # before falling back to the app default (~/.fusion-mlx/models, config.py).
    env = os.environ.get("FUSION_MLX_MODELS", "").strip()
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p.strip()]
    return [Path.home() / ".fusion-mlx" / "models"]


def _registry_resolve_model_dirs(
    model_dir: str | Path | list[str | Path] | None,
) -> list[Path]:
    if model_dir is None:
        return _registry_default_model_dirs()
    if isinstance(model_dir, (str, Path)):
        return [Path(model_dir)]
    return [Path(p) for p in model_dir]


def _registry_image_entry(model_id: str, info: Any) -> dict[str, Any]:
    return {
        "name": model_id,
        "type": "image",
        "backend": _IMAGE_BACKEND,
        "quantize": list(_IMAGE_QUANTIZE_OPTIONS),
        "constraints": {
            "dim_divisibility": _IMAGE_DIM_DIVISIBILITY,
            "max_resolution": _IMAGE_MAX_RESOLUTION,
        },
        "path": info.model_path,
        "size_bytes": info.estimated_size,
    }


def _registry_video_capabilities(model_name: str) -> dict[str, Any]:
    # resolve_backend builds a throwaway backend (no weights loaded);
    # constraints_for reads its static VideoConstraints. Both are cheap.
    from .engines.video_backends import constraints_for, resolve_backend

    try:
        backend = resolve_backend(model_name)
        constraints = constraints_for(model_name)
    except Exception as exc:
        logger.warning(
            "model_registry: backend resolve failed for %s: %s",
            model_name,
            exc,
        )
        return {"backend": "unknown", "supports_i2v": False, "constraints": {}}

    constraints_dict: dict[str, Any] = {
        "dim_divisibility": constraints.dim_divisibility,
        "max_n": constraints.max_n,
    }
    # num_frames_hint is a human-readable frame rule (e.g. "1 + 8k"); surface
    # it as frame_pattern so ComfyUI can validate frame counts up front.
    if constraints.num_frames_hint:
        constraints_dict["frame_pattern"] = constraints.num_frames_hint
    if constraints.dim_hint:
        constraints_dict["dim_hint"] = constraints.dim_hint
    return {
        "backend": backend.name,
        "supports_i2v": bool(constraints.supports_i2v),
        "constraints": constraints_dict,
    }


def _registry_video_entry(model_id: str, info: Any) -> dict[str, Any]:
    caps = _registry_video_capabilities(model_id)
    return {
        "name": model_id,
        "type": "video",
        "backend": caps.get("backend", "unknown"),
        "supports_i2v": caps.get("supports_i2v", False),
        "constraints": caps.get("constraints", {}),
        "path": info.model_path,
        "size_bytes": info.estimated_size,
    }


def _registry_other_entry(model_id: str, info: Any) -> dict[str, Any]:
    # Non image/video models (llm, vlm, audio, embedding, ...). Included only
    # when no type filter is set, so ComfyUI gets the full inventory.
    return {
        "name": model_id,
        "type": (info.model_type or "unknown"),
        "backend": (info.engine_type or "unknown"),
        "path": info.model_path,
        "size_bytes": info.estimated_size,
    }


def list_available_models(
    model_type: str | None = None,
    *,
    model_dir: str | Path | list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    # Scan model directories (respecting FUSION_MLX_MODELS / explicit model_dir)
    # and return structured capability info. model_type filters to "image" or
    # "video"; None returns every discovered model. Additive + read-only: no
    # model is loaded and no state is mutated.
    if model_type is not None:
        model_type = model_type.lower()
        if model_type not in _VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {_VALID_MODEL_TYPES} or None; "
                f"got {model_type!r}"
            )

    dirs = _registry_resolve_model_dirs(model_dir)
    logger.info("model_registry: scanning dirs=%s type=%s", dirs, model_type)

    # Lazy import: pool.model_discovery is heavier than the ownership tracker;
    # keep it out of the engine_core import path.
    from .pool.model_discovery import discover_models_from_dirs

    discovered = discover_models_from_dirs(dirs)

    results: list[dict[str, Any]] = []
    for model_id, info in sorted(discovered.items()):
        mt = (info.model_type or "").lower()
        if model_type is not None and mt != model_type:
            continue
        if mt == "image":
            results.append(_registry_image_entry(model_id, info))
        elif mt == "video":
            results.append(_registry_video_entry(model_id, info))
        elif model_type is None:
            results.append(_registry_other_entry(model_id, info))

    logger.info(
        "model_registry: found %d models (type=%s, scanned=%d)",
        len(results),
        model_type,
        len(discovered),
    )
    return results
