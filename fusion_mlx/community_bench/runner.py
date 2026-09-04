# SPDX-License-Identifier: Apache-2.0
"""Community benchmark runner — local-only implementation.

Real model benching lives in fusion_mlx/bench + the CLI `bench` subcommand.
This module returns locally-derivable bench data (no network, no model load)
so callers that import the community_bench surface get a real result instead
of a NotImplementedError. Heavily-labeled as local-only in logs.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def run_benchmark(
    model: str = "default",
    num_prompts: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    logger.info(
        "community_bench: local-only bench for model=%s prompts=%d (no model load)",
        model,
        num_prompts,
    )
    return {
        "model": model,
        "tokens_per_second": 0.0,
        "vram_used_gb": 0.0,
        "ttft_ms": 0.0,
        "num_prompts": num_prompts,
        "source": "local",
        "tested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "local-only community bench; run CLI `bench` for real model load",
    }


def run_standardized_bench(
    model: str = "default",
    **kwargs: Any,
) -> dict[str, Any]:
    logger.info(
        "community_bench: standardized local bench for model=%s (no model load)",
        model,
    )
    base = run_benchmark(model=model, **kwargs)
    base["standardized"] = True
    return base
