# SPDX-License-Identifier: Apache-2.0
"""DFlash2 speculative decoding — z-lab block-diffusion (official dflash pkg).

Bridges the official ``dflash`` PyPI package (0.1.0, MLX-native,
DFlash2DraftModel + CandidateSelector + GroupedDynamicCausalConv) into
fusion-mlx. Mirrors the DSpark self-contained-generator pattern: a
``DFlash2Generator`` loads its own target + draft and runs the full
propose->verify->rollback loop via ``dflash.stream_generate`` (hidden-state
capture/trim handled internally, lossless). The scheduler step pulls
tokens from the generator session and emits RequestOutputs — it does NOT
reuse ``dflash_spec_step`` (whose ``draft_block`` contract passes no
target hidden states, incompatible with DFlash2's propose). See
architecture/fusion-mlx-dflash2.md §5.3 for the design rationale.

Public API:
- ``DFlash2Unavailable``: raised by ``eligibility.check`` on gate failure
- ``check``: AliasProfile-based eligibility gate
- ``have_runtime``: probe whether the ``dflash`` pkg is importable
- ``DFlash2Runtime``: handle owning the generator + telemetry
- ``load_runtime``: lazy build of DFlash2Generator
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
