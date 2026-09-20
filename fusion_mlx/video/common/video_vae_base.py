# SPDX-License-Identifier: Apache-2.0
"""Abstract Video-VAE base (PRD v1 §3.5 / §4.2).

Every video backend's VAE (LTX high-compression 1:192, H3 standard)
implements this interface so the unified scheduler can decode latents
without knowing which model produced them. Tiled decode is the
memory-safe path (PRD §3.2: per-step tensor immediate release).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import mlx.core as mx

logger = logging.getLogger(__name__)


class VideoVAEBase(ABC):
    name: str = "base"

    @abstractmethod
    def encode(self, pixels: mx.array) -> mx.array:
        pass

    @abstractmethod
    def decode(self, latent: mx.array) -> mx.array:
        pass

    @abstractmethod
    def decode_tiled(self, latent: mx.array, tile_size: int = 256) -> mx.array:
        pass

    def release(self) -> None:
        logger.debug("VAE %s release (no-op base)", self.name)

    def stats(self) -> dict:
        return {"vae": self.name}
