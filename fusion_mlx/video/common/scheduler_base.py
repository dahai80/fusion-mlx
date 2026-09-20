# SPDX-License-Identifier: Apache-2.0
"""Abstract noise-scheduler base (PRD v1 §5.1/§5.2).

LTX cosine/distilled scheduler and H3 short-drama scheduler both
implement this. The unified scheduler calls step()/add_noise() without
caring about the model's sigma table shape.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import mlx.core as mx

logger = logging.getLogger(__name__)


class NoiseSchedulerBase(ABC):
    name: str = "base"

    @abstractmethod
    def sigmas(self, num_steps: int) -> mx.array:
        pass

    @abstractmethod
    def add_noise(self, latent: mx.array, sigma: mx.array, seed: int) -> mx.array:
        pass

    @abstractmethod
    def step(self, latent: mx.array, model_out: mx.array, sigma: mx.array) -> mx.array:
        pass
