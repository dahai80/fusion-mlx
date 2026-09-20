# SPDX-License-Identifier: Apache-2.0
"""Abstract upsampler base (PRD v1 §5.1/§5.2).

LTX multi-stage超分 (1080P) and H3 two-stage 384P→768P精炼 both
implement this. The unified router picks the backend's upsampler
without touching model-specific conv stacks.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import mlx.core as mx

logger = logging.getLogger(__name__)


class UpsampleBase(ABC):
    scale: float = 2.0
    name: str = "base"

    @abstractmethod
    def upsample(self, latent: mx.array) -> mx.array:
        pass

    def release(self) -> None:
        logger.debug("upsampler %s release (no-op base)", self.name)
