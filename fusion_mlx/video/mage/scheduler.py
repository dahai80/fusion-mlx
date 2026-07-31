# SPDX-License-Identifier: Apache-2.0
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


class FlowMatchScheduler:
    def __init__(self, num_steps: int = 30, shift: float = 6.0):
        self.num_steps = num_steps
        self.shift = shift
        self.sigmas = self._compute_sigmas()

    def _compute_sigmas(self) -> mx.array:
        sigmas = []
        for i in range(self.num_steps + 1):
            t = 1.0 - i / self.num_steps
            sigma = t / (1.0 - t) if t < 1.0 else 1.0
            sigma = sigma / (sigma + self.shift)
            sigmas.append(sigma)
        return mx.array(sigmas, dtype=mx.float32)

    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
    ) -> mx.array:
        sigma = self.sigmas[timestep]
        sigma_next = self.sigmas[timestep + 1]
        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output
        return prev_sample

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timestep: int,
    ) -> mx.array:
        sigma = self.sigmas[timestep]
        return (1.0 - sigma) * original_samples + sigma * noise
