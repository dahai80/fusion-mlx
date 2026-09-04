# SPDX-License-Identifier: Apache-2.0
"""Community benchmark submission — local-first no-op.

fusion-mlx is 100% local (no cloud telemetry submission). This surface
is kept importable so callers don't crash, but it never transmits data.
Returns a dict describing the no-op so callers can log it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def submit_benchmark(*args: Any, **kwargs: Any) -> dict[str, Any]:
    logger.info(
        "community_bench: submission skipped — local-first build, no cloud upload"
    )
    return {
        "submitted": False,
        "reason": "local-first build; community submission disabled",
    }


def submit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return submit_benchmark(*args, **kwargs)
