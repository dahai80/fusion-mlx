# SPDX-License-Identifier: Apache-2.0
# DFlash2 runtime wrapper — bridges the official dflash pkg.
#
# dflash (PyPI 0.1.0, z-lab) is MLX-native: DFlash2DraftModel.propose +
# CandidateSelector + GroupedDynamicCausalConv, with stream_generate
# running the full propose->verify->rollback loop (hidden-state capture
# via _patch_model + _GDNStateCapture, trimmed per accept count). We do
# NOT vendor it (unlike DSpark) — it is a declared pip dependency. The
# import stays local to load_runtime so the heavy mlx stack only loads
# on demand and unit tests can mock the pkg without installing it.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DFlash2Runtime:
    generator: Any = None
    target_repo: str = ""
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
    target_repo: str,
    draft_repo: str,
    block_size: int = 5,
) -> DFlash2Runtime:
    if not target_repo:
        raise ValueError("target_repo must be a non-empty string")
    if not draft_repo:
        raise ValueError("draft_repo must be a non-empty string")
    if block_size <= 0 or block_size > 5:
        raise ValueError(
            f"block_size must be in [1, 5] for MLX quantized targets; got {block_size}"
        )
    from .engine import DFlash2Generator

    logger.info(
        "loading DFlash2Generator target=%s draft=%s block_size=%d",
        target_repo,
        draft_repo,
        block_size,
    )
    gen = DFlash2Generator(
        target_repo=target_repo,
        draft_repo=draft_repo,
        block_size=block_size,
    )
    logger.info(
        "DFlash2Generator ready target=%s draft=%s block_size=%d",
        target_repo,
        draft_repo,
        block_size,
    )
    return DFlash2Runtime(
        generator=gen,
        target_repo=target_repo,
        draft_repo=draft_repo,
        block_size=block_size,
    )
