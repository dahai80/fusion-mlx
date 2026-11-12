# SPDX-License-Identifier: Apache-2.0
"""DSpine runtime — bundles drafter + hidden capture for the scheduler.

In-target pattern (same as DFlash2Runtime): the runtime holds ONLY the
drafter side. The target model is bound at engine startup via
DSpineDrafter.load(target_model); hidden states from the target's
intermediate layers are captured by a permanently-installed
HiddenStateCapture on the engine model (layer replacement, see
hidden_capture.py — install happens at engine load, between batches,
under _capture_lock).

Issue #1549.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DSpineRuntime:
    drafter: Any = None
    capture: Any = None
    draft_repo: str = ""
    block_size: int = 16
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
    target_model: Any,
    block_size: int = 16,
    target_layer_ids: list[int] | None = None,
) -> DSpineRuntime:
    """Load the DSpine drafter, bind it to the loaded target, and install
    the hidden-state capture on the target model.

    Must run on the same single-worker MLX executor that runs scheduler
    steps (#411 pattern — MLX binds arrays to the loading thread's stream).
    """
    if not draft_repo:
        raise ValueError("draft_repo must be a non-empty string")
    if block_size <= 1:
        raise ValueError(f"block_size must be >= 2; got {block_size}")

    from ..hidden_capture import HiddenStateCapture
    from .drafter import DSpineDrafter

    logger.info(
        "loading DSpineDrafter draft=%s block_size=%d target_layers=%s",
        draft_repo,
        block_size,
        target_layer_ids,
    )
    drafter = DSpineDrafter(
        draft_repo,
        target_model=target_model,
        target_layer_ids=target_layer_ids,
        block_size=block_size,
    )
    drafter.load(target_model)
    capture = HiddenStateCapture(target_model, layer_ids=drafter.target_layer_ids)
    capture.install()
    logger.info(
        "DSpine capture installed on layers %s (concat_size=%d)",
        drafter.target_layer_ids,
        drafter.config.concat_size,
    )
    return DSpineRuntime(
        drafter=drafter,
        capture=capture,
        draft_repo=draft_repo,
        block_size=block_size,
    )
