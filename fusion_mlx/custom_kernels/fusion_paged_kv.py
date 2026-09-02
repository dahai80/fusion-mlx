from __future__ import annotations

import logging
from typing import Any

import mlx.nn as nn

logger = logging.getLogger(__name__)

_INSTALLED_ATTR = "_fusion_paged_kv_installed"


def is_paged_kv_available() -> bool:
    try:
        from .glm_moe_dsa.fast import is_native_available

        return is_native_available()
    except Exception:
        return False


def install_paged_kv(model: nn.Module, config: Any) -> None:
    logger.info(
        "install_paged_kv: stub install on %s (block_size=%s num_blocks=%s native=%s)",
        type(model).__name__,
        getattr(config, "paged_kv_block_size", 16),
        getattr(config, "paged_kv_num_blocks", 256),
        is_paged_kv_available(),
    )
    try:
        setattr(model, _INSTALLED_ATTR, True)
    except Exception:
        pass


def uninstall_paged_kv(model: nn.Module) -> None:
    try:
        if getattr(model, _INSTALLED_ATTR, False):
            setattr(model, _INSTALLED_ATTR, False)
            logger.info("uninstall_paged_kv: cleared on %s", type(model).__name__)
    except Exception:
        pass


__all__ = [
    "is_paged_kv_available",
    "install_paged_kv",
    "uninstall_paged_kv",
]
