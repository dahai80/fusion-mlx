# SPDX-License-Identifier: Apache-2.0
"""DFlash2 speculative decoding — z-lab block-diffusion (official dflash pkg).

Bridges the official ``dflash`` PyPI package (0.1.0, MLX-native,
DFlash2DraftModel + CandidateSelector + GroupedDynamicCausalConv) into
fusion-mlx. In-target pattern: ``DFlash2InTargetDrafter`` loads ONLY the
draft model and binds to the scheduler's already-loaded target (via
``draft.bind`` + ``_patch_model`` hooks). The propose->verify->rollback
loop runs in ``dflash2_spec_step`` (spec_decode.py), reusing the
scheduler's model + prompt_cache — no duplicate 27B load, no prefill
replay. Mirrors the DFlash-v1 in-target pattern (dflash/drafter.py).

Public API:
- ``DFlash2Unavailable``: raised by ``eligibility.check`` on gate failure
- ``check``: AliasProfile-based eligibility gate
- ``have_runtime``: probe whether the ``dflash`` pkg is importable
- ``DFlash2Runtime``: handle owning the drafter + telemetry
- ``load_runtime``: lazy build of DFlash2InTargetDrafter
"""

from .eligibility import DFlash2Unavailable, check, have_runtime
from .runtime import DFlash2Runtime, load_runtime

__all__ = [
    "DFlash2Unavailable",
    "check",
    "have_runtime",
    "DFlash2Runtime",
    "load_runtime",
]
