# SPDX-License-Identifier: Apache-2.0
# Flow-matching scheduler for HunyuanVideo.

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


class HunyuanVideoScheduler:
    def __init__(
        self,
        sigma_min=0.002,
        sigma_max=700.0,
        num_train_timesteps=1000,
        prediction_type="flow_prediction",
        shift=3.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.shift = shift
        self.init_noise_sigma = 1.0
        self.timesteps = None
        self.sigmas = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps):
        n = num_inference_steps
        sigmas = mx.linspace(self.sigma_max, self.sigma_min, n + 1, dtype=mx.float32)
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        timesteps = sigmas[:-1]
        self.timesteps = timesteps
        self.sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=mx.float32)])
        self.num_inference_steps = n
        logger.info(
            "hunyuan scheduler: steps=%d sigma_range=[%.4f, %.4f] shift=%.2f",
            n,
            float(timesteps[0]),
            float(timesteps[-1]),
            self.shift,
        )
        return timesteps

    def scale_model_input(self, sample, timestep=None):
        return sample

    def step(self, model_output, timestep, sample):
        sigma = (
            timestep
            if isinstance(timestep, mx.array)
            else mx.array(timestep, dtype=mx.float32)
        )
        sigma = sigma.reshape((-1,) + (1,) * (sample.ndim - 1))
        sigma_next = self._get_next_sigma(sigma)
        sigma_next = sigma_next.reshape((-1,) + (1,) * (sample.ndim - 1))
        if self.prediction_type == "flow_prediction":
            dt = sigma_next - sigma
            prev_sample = sample + dt * model_output
        else:
            denoised = sample - model_output * sigma
            dt = sigma_next - sigma
            d = (sample - denoised) / sigma
            prev_sample = sample + dt * d
        return prev_sample

    def _get_next_sigma(self, sigma):
        sigmas_list = self.sigmas.tolist()
        sigma_val = (
            float(sigma.flatten()[0]) if isinstance(sigma, mx.array) else float(sigma)
        )
        for i, s in enumerate(sigmas_list):
            if abs(s - sigma_val) < 1e-4 and i + 1 < len(sigmas_list):
                return mx.array(sigmas_list[i + 1], dtype=mx.float32)
        return mx.zeros((1,), dtype=mx.float32)

    def add_noise(self, original, noise, timesteps):
        sigma = timesteps
        if isinstance(sigma, (int, float)):
            sigma = mx.array(sigma, dtype=mx.float32)
        sigma = sigma.reshape((-1,) + (1,) * (original.ndim - 1))
        return original + sigma * noise
