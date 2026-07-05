# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache compression stub — not available in this build."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def resolve_turboquant_mode_default(
    mode, model_name: str = "", model_family: str = ""
) -> Any:
    logger.debug(
        "TurboQuant not available, returning None (mode=%s, model=%s)",
        mode,
        model_name or model_family,
    )
    return None
