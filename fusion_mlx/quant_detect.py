# SPDX-License-Identifier: Apache-2.0
"""Shared quantization detection helpers.

R-P1-7 (#0908 audit): _looks_like_4bit was duplicated 4x across
speculative strategy eligibility modules (dflash/dflash2/dspark/dfly).
"""

import logging

logger = logging.getLogger(__name__)


def looks_like_4bit(hf_path: str) -> bool:
    lowered = hf_path.lower()
    if "-4bit" in lowered:
        return True
    if "mxfp4" in lowered or "nvfp4" in lowered:
        return True
    return False
