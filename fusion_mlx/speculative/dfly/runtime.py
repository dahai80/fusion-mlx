# SPDX-License-Identifier: Apache-2.0
"""DFly runtime — lazy import and lifecycle hooks for the DFly drafter."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_runtime(drafter_path: str | None = None) -> Any:
    """Lazy-load the DFly drafter runtime.

    Returns the DFlyDrafter class (not an instance) so the caller
    controls instantiation.
    """
    from .drafter import DFlyDrafter

    logger.info("DFly runtime loaded (drafter_path=%s)", drafter_path)
    return DFlyDrafter
