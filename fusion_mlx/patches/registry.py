# SPDX-License-Identifier: Apache-2.0
"""Central patch registry — single source of truth for all model patches.

Replaces the no-op apply_all_for_model with a real registry. Each patch
registers its id, target, reason, and apply function. apply_all_for_model
dispatches to all registered patches matching the model type.

Audit coverage: grep call sites no longer needed — registry lists everything.
Idempotency: each patch tracks is_applied state, double-apply is a no-op.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


@dataclass
class PatchEntry:
    """One registered patch."""

    patch_id: str
    target: str  # model_type or "global"
    reason: str
    apply_fn: Callable[[Any, Any], None]
    is_applied: bool = False
    is_global: bool = False  # True = unconditional (applied to all models)
    upstream_issue: str = ""  # GitHub issue URL if filed upstream


_REGISTRY: list[PatchEntry] = []


def register(
    patch_id: str,
    target: str,
    reason: str,
    apply_fn: Callable[[Any, Any], None],
    *,
    is_global: bool = False,
    upstream_issue: str = "",
) -> None:
    """Register a patch in the central registry.

    patch_id: unique identifier (e.g. "mtp_vlm_qwen35")
    target: model type this patch applies to (e.g. "qwen3_vlm", or "global")
    reason: why this patch exists (one line)
    apply_fn: callable(model, config) -> None
    is_global: True if unconditional (applies to all models)
    upstream_issue: URL of upstream issue if filed
    """
    with _LOCK:
        for e in _REGISTRY:
            if e.patch_id == patch_id:
                logger.debug("Patch %s already registered", patch_id)
                return
        entry = PatchEntry(
            patch_id=patch_id,
            target=target,
            reason=reason,
            apply_fn=apply_fn,
            is_global=is_global,
            upstream_issue=upstream_issue,
        )
        _REGISTRY.append(entry)
        logger.debug("Registered patch: %s (target=%s)", patch_id, target)


def apply_all_for_model(model_type: str, config: Any = None) -> None:
    """Apply all registered patches matching model_type.

    Global patches (is_global=True) apply to every model.
    Targeted patches apply only when model_type matches entry.target.
    Idempotent: already-applied patches are skipped.
    """
    applied = 0
    with _LOCK:
        entries = list(_REGISTRY)
    for entry in entries:
        if entry.is_applied:
            continue
        if entry.is_global or entry.target == model_type:
            try:
                entry.apply_fn(None, config)
                entry.is_applied = True
                applied += 1
                logger.info(
                    "Applied patch: %s (target=%s)", entry.patch_id, entry.target
                )
            except Exception as e:
                logger.error("Patch %s failed: %s", entry.patch_id, e, exc_info=True)
    if applied:
        logger.info("apply_all_for_model(%s): %d patches applied", model_type, applied)


def list_patches() -> list[dict[str, Any]]:
    """Return registry contents for audit/inspection."""
    with _LOCK:
        return [
            {
                "patch_id": e.patch_id,
                "target": e.target,
                "reason": e.reason,
                "is_applied": e.is_applied,
                "is_global": e.is_global,
                "upstream_issue": e.upstream_issue,
            }
            for e in _REGISTRY
        ]


def reset_for_tests() -> None:
    """Clear registry (tests only)."""
    with _LOCK:
        _REGISTRY.clear()
