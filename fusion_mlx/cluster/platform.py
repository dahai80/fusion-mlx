"""Cluster node platform identification (#365).

The fusion-gateway routes requests by platform: lightweight inference -> ``mac``
(fusion-mlx MLX), heavy LLM / diffusion -> ``windows-cuda`` (vLLM CUDA node).
This module gives every node a stable ``platform`` string surfaced through the
mDNS TXT records so the gateway's ``HealthyNodesByPlatform`` can select nodes.

Detection order:
1. ``FUSION_PLATFORM`` env var (explicit override, e.g. ``windows-cuda``).
2. ``sys.platform`` heuristics (win32 + CUDA -> ``windows-cuda``, darwin -> ``mac``).
3. Fallback ``mac`` (this codebase's native platform).
"""

from __future__ import annotations

import logging
import os
import sys
from enum import Enum

logger = logging.getLogger(__name__)


class Platform(str, Enum):
    """Cluster node platform tag, matched verbatim by fusion-gateway routing."""

    MAC = "mac"
    WINDOWS_CUDA = "windows-cuda"

    def __str__(self) -> str:
        return self.value


def _cuda_available() -> bool:
    """Best-effort CUDA presence check (Windows CUDA nodes)."""
    try:
        import torch  # type: ignore

        return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        logger.debug("platform: torch.cuda probe failed", exc_info=True)
        return False


def detect_platform() -> Platform:
    """Detect this node's platform.

    Honors ``FUSION_PLATFORM`` first so a CUDA node can self-declare without
    relying on the torch probe (vLLM imports torch lazily on some setups).
    """
    explicit = os.environ.get("FUSION_PLATFORM", "").strip().lower()
    if explicit:
        try:
            return Platform(explicit)
        except ValueError:
            logger.warning(
                "platform: ignoring unknown FUSION_PLATFORM=%r (expected %s)",
                explicit,
                ", ".join(p.value for p in Platform),
            )
    if sys.platform.startswith("win") and _cuda_available():
        return Platform.WINDOWS_CUDA
    if sys.platform == "darwin":
        return Platform.MAC
    if sys.platform.startswith("win"):
        logger.info(
            "platform: Windows host without detected CUDA, reporting %s",
            Platform.MAC.value,
        )
    return Platform.MAC
