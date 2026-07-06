# SPDX-License-Identifier: Apache-2.0
import logging
import threading

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY: dict[str, str] = {
    "qwen3.5-27b-8bit": "z-lab/Qwen3.5-27B-DFlash",
}

_registry: dict[str, str] = dict(_DEFAULT_REGISTRY)
_registry_lock = threading.Lock()


def get_dflash_drafter_path(alias: str | None) -> str | None:
    if not alias:
        return None
    return _registry.get(alias)


def register_dflash_drafter(alias: str, drafter_hf_path: str) -> None:
    if not alias:
        raise ValueError("alias must be a non-empty string")
    if not drafter_hf_path:
        raise ValueError("drafter_hf_path must be a non-empty string")
    with _registry_lock:
        existing = _registry.get(alias)
        if existing == drafter_hf_path:
            return
        if existing is not None and existing != drafter_hf_path:
            logger.warning(
                "[dflash.registry] Overwriting DFlash drafter binding for "
                "alias %r: %r -> %r",
                alias,
                existing,
                drafter_hf_path,
            )
        _registry[alias] = drafter_hf_path


def list_registered_aliases() -> list[str]:
    return sorted(_registry.keys())


def clear_drafter_registry_for_tests() -> None:
    global _registry
    with _registry_lock:
        _registry = dict(_DEFAULT_REGISTRY)
