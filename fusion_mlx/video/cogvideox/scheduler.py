# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx
import numpy as np

logger = __import__("logging").getLogger(__name__)


class FlowMatchScheduler:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 3.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.timesteps = mx.array([])
        self.sigmas = mx.array([])

    def _sigma_to_t(self, sigma: float) -> float:
        return sigma * self.num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, shift: float | None = None):
        if shift is not None:
            self.shift = shift
        sigmas = np.linspace(
            1.0, 1.0 / self.num_train_timesteps, num_inference_steps + 1
        )
        sigmas = self._shift_sigmas(sigmas)
        self.sigmas = mx.array(sigmas[:-1])
        self.timesteps = self.sigmas * self.num_train_timesteps

    def _shift_sigmas(self, sigmas: np.ndarray) -> np.ndarray:
        if self.shift == 1.0:
            return sigmas
        t = sigmas
        shifted = t / (t + self.shift * (1.0 - t))
        return shifted

    def step(
        self,
        model_output: mx.array,
        timestep: float,
        sample: mx.array,
    ) -> mx.array:
        sigma = timestep / self.num_train_timesteps
        sigma_next = sigma - 1.0 / self.num_train_timesteps
        sigma_next = max(sigma_next, 0.0)
        dt = sigma_next - sigma
        pred = sample + model_output * dt
        return pred

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        sigmas = timesteps / self.num_train_timesteps
        while sigmas.ndim < original_samples.ndim:
            sigmas = sigmas[..., None]
        return (1.0 - sigmas) * original_samples + sigmas * noise
