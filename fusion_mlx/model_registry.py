# SPDX-License-Identifier: Apache-2.0
"""Model Registry — tracks model ownership to prevent BatchKVCache conflicts."""

import logging
import threading
import weakref
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

    def acquire(self, model: Any, engine: Any, engine_id: str, force: bool = False) -> bool:
        model_id = id(model)
        with self._registry_lock:
            if model_id in self._owners:
                weak_ref, owner_id = self._owners[model_id]
                owner = weak_ref()
                if owner is not None and owner_id != engine_id:
                    if force:
                        logger.warning(f"Model ownership transfer: {owner_id} -> {engine_id}")
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
            if hasattr(owner, 'scheduler'):
                owner.scheduler.deep_reset()
        except Exception as e:
            logger.warning(f"Failed to reset previous owner: {e}")

    def cleanup(self) -> int:
        cleaned = 0
        with self._registry_lock:
            stale = [k for k, (r, _) in self._owners.items() if r is not None and r() is None]
            for k in stale:
                del self._owners[k]
                cleaned += 1
        return cleaned

    def get_stats(self) -> dict[str, Any]:
        with self._registry_lock:
            active = sum(1 for _, (r, _) in self._owners.items() if r is None or r() is not None)
            return {"total_entries": len(self._owners), "active_owners": active}


_registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    return _registry
