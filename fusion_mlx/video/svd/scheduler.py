# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of the SVD Euler Discrete scheduler with v-prediction
# and Karras sigmas. Based on diffusers EulerDiscreteScheduler.

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def _linspace(start, stop, num, dtype=mx.float32):
    return mx.linspace(start, stop, num, dtype=dtype)


def _karras_sigmas(sigma_min, sigma_max, rho, num_steps):
    ramp = _linspace(0, 1, num_steps + 1)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


class SVDEulerScheduler:
    def __init__(
        self,
        sigma_min=0.002,
        sigma_max=700.0,
        sigma_data=1.0,
        rho=7.0,
        prediction_type="v_prediction",
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.prediction_type = prediction_type
        self.init_noise_sigma = 1.0
        self.timesteps = None
        self.sigmas = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps):
        n = num_inference_steps
        sigmas = _karras_sigmas(self.sigma_min, self.sigma_max, self.rho, n)
        timesteps = self._sigma_to_t(sigmas)
        self.timesteps = timesteps
        self.sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=mx.float32)])
        self.num_inference_steps = n
        logger.info(
            "svd scheduler: steps=%d sigma_range=[%.4f, %.4f]",
            n,
            float(sigmas[0]),
            float(sigmas[-1]),
        )
        return timesteps

    def _sigma_to_t(self, sigma):
        return sigma

    def scale_model_input(self, sample, timestep=None):
        if timestep is not None:
            sigma = (
                timestep
                if isinstance(timestep, mx.array)
                else mx.array(timestep, dtype=mx.float32)
            )
        else:
            sigma = mx.array(1.0, dtype=mx.float32)
        sigma = sigma.reshape((-1,) + (1,) * (sample.ndim - 1))
        return sample / (sigma**2 + self.sigma_data**2) ** 0.5

    def step(self, model_output, timestep, sample):
        sigma = (
            timestep
            if isinstance(timestep, mx.array)
            else mx.array(timestep, dtype=mx.float32)
        )
        sigma_next = self._get_next_sigma(sigma)
        sigma = sigma.reshape((-1,) + (1,) * (sample.ndim - 1))
        sigma_next = sigma_next.reshape((-1,) + (1,) * (sample.ndim - 1))
        if self.prediction_type == "v_prediction":
            denoised = (
                sample * sigma / (sigma**2 + self.sigma_data**2) ** 0.5
                + model_output
                * self.sigma_data
                / (sigma**2 + self.sigma_data**2) ** 0.5
            )
        else:
            denoised = sample - model_output * sigma
        dt = sigma_next - sigma
        derivative = (sample - denoised) / sigma
        prev_sample = sample + dt * derivative
        return prev_sample

    def _get_next_sigma(self, sigma):
        sigmas_list = self.sigmas.tolist()
        sigma_val = (
            float(sigma.flatten()[0]) if isinstance(sigma, mx.array) else float(sigma)
        )
        for i, s in enumerate(sigmas_list):
            if abs(s - sigma_val) < 1e-6 and i + 1 < len(sigmas_list):
                return mx.array(sigmas_list[i + 1], dtype=mx.float32)
        return mx.zeros((1,), dtype=mx.float32)

    def add_noise(self, original, noise, timesteps):
        sigma = timesteps
        if isinstance(sigma, (int, float)):
            sigma = mx.array(sigma, dtype=mx.float32)
        sigma = sigma.reshape((-1,) + (1,) * (original.ndim - 1))
        return original + sigma * noise
