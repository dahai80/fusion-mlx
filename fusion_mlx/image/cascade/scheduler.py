import logging
import math

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class DDPMWuerstchenScheduler:
    # MLX port of diffusers DDPMWuerstchenScheduler. Operates on
    # timestep_ratio t in [0,1] (NOT integer timesteps). alpha_cumprod uses
    # cosine schedule with offset s=0.008, normalized by initial alpha_cumprod
    # so t=1 -> 1.0. step() uses DDPM posterior mean + stochastic noise when
    # prev_t != 0. init_noise_sigma = 1.0.

    def __init__(self, s: float = 0.008, scaler: float = 1.0):
        self.s = s
        self.scaler = scaler
        self._init_alpha_cumprod = math.cos(self.s / (1 + self.s) * math.pi * 0.5) ** 2
        self.init_noise_sigma = 1.0
        self.timesteps = None

    def _alpha_cumprod(self, t: mx.array) -> mx.array:
        t = t.astype(mx.float32)
        if self.scaler > 1:
            t = 1 - (1 - t) ** self.scaler
        elif self.scaler < 1:
            t = t**self.scaler
        s = self.s
        acp = mx.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2 / self._init_alpha_cumprod
        return mx.clip(acp, 0.0001, 0.9999)

    def scale_model_input(self, sample: mx.array, timestep=None) -> mx.array:
        return sample

    def set_timesteps(self, num_inference_steps: int) -> None:
        ts = np.linspace(1.0, 0.0, num_inference_steps + 1, dtype=np.float32)
        self.timesteps = mx.array(ts)
        logger.info(
            "DDPMWuerstchen set_timesteps n=%d range=[%.4f..%.4f]",
            num_inference_steps,
            float(ts[0]),
            float(ts[-1]),
        )

    def previous_timestep(self, timestep: mx.array) -> mx.array:
        ts_np = np.array(self.timesteps, dtype=np.float32)
        t0 = float(timestep.reshape(-1)[0])
        index = int(np.abs(ts_np - t0).argmin())
        prev_index = index + 1
        if prev_index >= len(ts_np):
            prev_index = len(ts_np) - 1
        prev_val = ts_np[prev_index]
        return mx.full((timestep.shape[0],), float(prev_val), dtype=mx.float32)

    def _view(self, acp: mx.array, sample: mx.array) -> mx.array:
        shape = [acp.shape[0]] + [1] * (sample.ndim - 1)
        return acp.reshape(shape)

    def step(
        self,
        model_output: mx.array,
        timestep: mx.array,
        sample: mx.array,
        generator=None,
    ) -> mx.array:
        t = timestep.astype(mx.float32)
        prev_t = self.previous_timestep(t)
        alpha_cumprod = self._view(self._alpha_cumprod(t), sample)
        alpha_cumprod_prev = self._view(self._alpha_cumprod(prev_t), sample)
        alpha = alpha_cumprod / alpha_cumprod_prev
        mu = (1.0 / alpha).sqrt() * (
            sample - (1 - alpha) * model_output / (1 - alpha_cumprod).sqrt()
        )
        noise = mx.random.normal(mu.shape, dtype=mu.dtype)
        std = (
            (1 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        ).sqrt() * noise
        prev_t_expanded = self._view(prev_t, sample)
        mask = (prev_t_expanded != 0).astype(mu.dtype)
        pred = mu + std * mask
        return pred

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        alpha_cumprod = self._view(self._alpha_cumprod(timesteps), original_samples)
        return (
            alpha_cumprod.sqrt() * original_samples + (1 - alpha_cumprod).sqrt() * noise
        )
