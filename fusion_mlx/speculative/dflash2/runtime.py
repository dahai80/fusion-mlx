# SPDX-License-Identifier: Apache-2.0
# DFlash2 runtime wrapper — bridges the official dflash pkg.
#
# dflash (PyPI 0.1.0, z-lab) is MLX-native: DFlash2DraftModel.propose +
# CandidateSelector + GroupedDynamicCausalConv. We do NOT vendor it (unlike
# DSpark) — it is a declared pip dependency. The import stays local to
# load_runtime so the heavy mlx stack only loads on demand and unit tests
# can mock the pkg without installing it.
#
# In-target pattern: DFlash2Runtime holds ONLY the drafter (no target
# model). The target is bound at engine startup via drafter.bind(model).

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DFlash2Runtime:
    drafter: Any = None
    draft_repo: str = ""
    block_size: int = 5
    _accept_lens: list = field(default_factory=list)

    def record_accept(self, avg_accept: float | None) -> None:
        if avg_accept and avg_accept > 0:
            self._accept_lens.append(float(avg_accept))

    def accept_lens_snapshot(self) -> list:
        return list(self._accept_lens)

    def reset_accept_lens(self) -> None:
        self._accept_lens.clear()


def load_runtime(
    draft_repo: str,
    block_size: int = 5,
    draft_bits: int | None = 4,
) -> DFlash2Runtime:
    if not draft_repo:
        raise ValueError("draft_repo must be a non-empty string")
    if block_size <= 0 or block_size > 8:
        raise ValueError(f"block_size must be in [1, 8]; got {block_size}")
    if draft_bits is not None and draft_bits not in (4, 8):
        raise ValueError(f"draft_bits must be 4 or 8; got {draft_bits}")
    from .engine import DFlash2InTargetDrafter

    logger.info(
        "loading DFlash2InTargetDrafter draft=%s block_size=%d draft_bits=%s",
        draft_repo,
        block_size,
        draft_bits,
    )
    drafter = DFlash2InTargetDrafter(
        draft_repo=draft_repo,
        block_size=block_size,
        draft_bits=draft_bits,
    )
    logger.info(
        "DFlash2InTargetDrafter ready draft=%s block_size=%d draft_bits=%s",
        draft_repo,
        block_size,
        draft_bits,
    )
    return DFlash2Runtime(
        drafter=drafter,
        draft_repo=draft_repo,
        block_size=block_size,
    )
