# SPDX-License-Identifier: Apache-2.0
"""DFly speculative-decoding integration (AngelSpec Hy3 migration).

DFly = DFlash + hidden-correction, a block-parallel drafter native to
Hunyuan (Hy3) models.  AngelSlim publishes released drafter checkpoints:
  - AngelSlim/Hy3-DFly-Block8 (no-think)
  - AngelSlim/Hy3-DFly-Block8-Think-High (high-think)

The verify loop reuses the same block-diffusion engine as DFlash; DFly
adds a HiddenStatesCorrection module:
    h'_{t+i} = h_{t+i} + SwiGLU(norm(h_{t+i}) :: norm(e_{t+i-1}))

Public API:
- ``DFlyUnavailable``: raised by ``eligibility.check`` on gate failure
- ``check(profile)``: AliasProfile-based eligibility gate
- ``DFlyDrafter``: MLX drafter loading AngelSlim weights
- ``DFlyAcceptCounter`` / ``DFlyAcceptSnapshot``: block-level stats
- ``DEFAULT_BLOCK_SIZE``: default block size (16)
"""

from .eligibility import DFlyUnavailable, check, eligible_aliases, report
from .drafter import DFlyDrafter
from .accept_counter import (
    DFlyAcceptCounter,
    DFlyAcceptSnapshot,
    get_global_counter,
    reset_global_counter_for_tests,
)
from .runtime import load_runtime

DEFAULT_BLOCK_SIZE = 16

__all__ = [
    "DFlyUnavailable",
    "check",
    "report",
    "eligible_aliases",
    "DFlyDrafter",
    "DFlyAcceptCounter",
    "DFlyAcceptSnapshot",
    "get_global_counter",
    "reset_global_counter_for_tests",
    "load_runtime",
    "DEFAULT_BLOCK_SIZE",
]
