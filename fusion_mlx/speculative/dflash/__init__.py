# SPDX-License-Identifier: Apache-2.0
"""DFlash speculative-decoding integration (issue #264).

DFlash is a block-diffusion drafter (z-lab) integrated into mlx-vlm's
``generate_step``. Rapid-MLX wires it into ``BatchedEngine`` for B=1
generation; B>1 transparently falls back to AR until phase-2 batched
support lands.

Two eligibility paths coexist:
- ``eligibility.check`` — fusion-mlx AliasProfile-based gating (primary)
- ``detect.detect_dflash_eligibility`` — Rapid-MLX config-based detection

Public API:
- ``DFlashUnavailable``: raised by ``eligibility.check`` on gate failure
- ``check(profile)``: AliasProfile-based eligibility gate
- ``load_runtime(drafter_repo)``: lazy import of mlx-vlm drafter loader
- ``DFlashEligibility``: enum (NONE / READY) for config-based detection
- ``detect_dflash_eligibility``: config-based eligibility check
- ``DFlashAcceptCounter`` / ``DFlashAcceptSnapshot``: block-level stats
- ``get_global_counter``: module-level accept counter singleton
- ``get_dflash_drafter_path`` / ``register_dflash_drafter``: drafter registry
- ``BlockDiffusionDrafter``: protocol for drafters
- ``MlxVlmDFlashDriver``: production driver wrapping mlx-vlm
- ``dflash_generate_step``: standalone generate loop
- ``verify_block`` / ``VerifyResult``: single-block verification
- ``DEFAULT_BLOCK_SIZE``: default block size (16)
"""

# Rapid-MLX migrated additions
from .accept_counter import (
    DFlashAcceptCounter,
    DFlashAcceptSnapshot,
    get_global_counter,
    reset_global_counter_for_tests,
)
from .detect import DFlashEligibility, detect_dflash_eligibility
from .drafter import (
    BlockDiffusionDrafter,
    MlxVlmDFlashDriver,
    StubBlockDiffusionDrafter,
)
from .drafter_registry import (
    clear_drafter_registry_for_tests,
    get_dflash_drafter_path,
    list_registered_aliases,
    register_dflash_drafter,
)
from .eligibility import DFlashUnavailable, check
from .generator import dflash_generate_step
from .runtime import load_runtime
from .verifier import VerifyResult, verify_block

DEFAULT_BLOCK_SIZE = 16

__all__ = [
    # Original fusion-mlx
    "DFlashUnavailable",
    "check",
    "load_runtime",
    # Rapid-MLX migrated
    "DFlashAcceptCounter",
    "DFlashAcceptSnapshot",
    "DFlashEligibility",
    "BlockDiffusionDrafter",
    "MlxVlmDFlashDriver",
    "StubBlockDiffusionDrafter",
    "VerifyResult",
    "DEFAULT_BLOCK_SIZE",
    "detect_dflash_eligibility",
    "dflash_generate_step",
    "get_dflash_drafter_path",
    "get_global_counter",
    "list_registered_aliases",
    "register_dflash_drafter",
    "reset_global_counter_for_tests",
    "verify_block",
    "clear_drafter_registry_for_tests",
]
