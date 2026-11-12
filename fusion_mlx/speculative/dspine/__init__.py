# SPDX-License-Identifier: Apache-2.0
"""DSpine speculative decoding (Tencent arXiv:2609.36173, Apache-2.0).

Block-parallel drafter for full-attention Qwen3-style targets: a 5-layer
draft consumes target intermediate-layer hidden states (shared fc
projection), block-causal within a 16-token block, adjacent causal
injection across blocks via the code bridge; RefinementTail does serial
greedy/sampled refinement; verification is standard greedy chain verify
on the target.

Ported from the reference MLX implementation (dspine_mlx). Measured on
M5 Max, Qwen3-8B bf16 + DSpine-b16: 3.6-8.3x vs AR at T=0 (issue #1549,
fusion-research docs/logs/dspine-m5max-bench-1112.md).

Public API:
- ``DSpineDrafter``: checkpoint loader + per-round block drafting
- ``DEFAULT_BLOCK_SIZE``: 16
"""

from .drafter import (
    DEFAULT_BLOCK_SIZE,
    DSpineDrafter,
    draft_model_num_parameters,
)
from .eligibility import DSpineEligibilityReport, DSpineUnavailable, check

__all__ = [
    "DSpineDrafter",
    "DEFAULT_BLOCK_SIZE",
    "draft_model_num_parameters",
    "DSpineUnavailable",
    "DSpineEligibilityReport",
    "check",
]
