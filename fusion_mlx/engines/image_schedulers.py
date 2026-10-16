# SPDX-License-Identifier: Apache-2.0
"""#1202: DPM-Solver++(2M) scheduler for mflux flow-matching models.

Registered as a contrib scheduler ("dpmpp_2m" / "dpmpp_2m_sde") so image
requests can pass scheduler="dpmpp_2m" through the existing scheduler
request param. Dev-tier quality: a 2nd-order multistep integrator reaches
comparable quality in fewer steps than plain Euler (28 -> 12-20).

Update rule (h = sigma_cur - sigma_next >= 0, downward sigma direction):
  step 0:   x1 = x0 - h0 * v0                     (Euler fallback)
  step i>0: x_{i+1} = x_i - h_i * v_i - (h_i^2 / (2*h_{i-1})) * (v_i - v_{i-1})
Both terms SUBTRACT: mflux's Euler is latents + (sigma_next - sigma_cur)*v
= latents - h*v, and the variable-step AB2 correction carries the same
sign. Getting either sign wrong produces pure-noise output (verified
during development: PSNR ~8dB, ~2-3x Laplacian variance = noise
characteristics).

The sigma schedule mirrors mflux's FlowMatchEulerDiscreteScheduler
offline path exactly (linear ladder + exponential time-shift + terminal
stretch), so sigma ladders match the Euler baseline and paired
comparisons are apples to apples.
"""

import logging
import math

import mlx.core as mx
from mflux.models.common.schedulers import register_contrib
from mflux.models.common.schedulers.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)

REGISTRY_KEYS = ("dpmpp_2m", "dpmpp_2m_sde")


class DPMPP2MSdeScheduler(BaseScheduler):
    def __init__(self, config):
        self.config = config
        self.model_config = config.model_config
        self.num_train_timesteps = 1000
        self.shift_terminal = 0.02
        self._prev_v = None
        self._prev_h = 0.0
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas()

    @property
    def sigmas(self):
        return self._sigmas

    @property
    def timesteps(self):
        return self._timesteps

    def set_image_seq_len(self, image_seq_len: int) -> None:
        from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
            FlowMatchEulerDiscreteScheduler,
        )

        self._timesteps, self._sigmas = (
            FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas(
                image_seq_len=image_seq_len,
                num_inference_steps=self.config.num_inference_steps,
            )
        )
        self._reset_step_state()
        logger.debug(
            "dpmpp_2m schedule set: seq_len=%d steps=%d sigma0=%.4f",
            image_seq_len,
            self.config.num_inference_steps,
            float(self._sigmas[0]),
        )

    def set_mu(self, mu: float) -> None:
        from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
            FlowMatchEulerDiscreteScheduler,
        )

        num_steps = self.config.num_inference_steps
        sigmas = mx.linspace(1.0, 1.0 / num_steps, num_steps, dtype=mx.float32)
        sigmas = FlowMatchEulerDiscreteScheduler._time_shift_exponential_array(
            mu, 1.0, sigmas
        )
        self._timesteps = sigmas * self.num_train_timesteps
        self._sigmas = mx.concatenate(
            [sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0
        )
        self._reset_step_state()

    def _stretch_to_terminal(self, sigmas: list) -> list:
        one_minus_sigmas = [1.0 - s for s in sigmas]
        scale_factor = one_minus_sigmas[-1] / (1.0 - self.shift_terminal)
        stretched = [1.0 - (oms / scale_factor) for oms in one_minus_sigmas]
        return stretched

    def _compute_timesteps_and_sigmas(self):
        num_steps = self.config.num_inference_steps
        sigma_min = 1.0 / self.num_train_timesteps
        sigma_max = 1.0
        timesteps_linear = [
            sigma_max * self.num_train_timesteps
            - i * (sigma_max - sigma_min) * self.num_train_timesteps / (num_steps - 1)
            for i in range(num_steps)
        ]
        sigmas_linear = [t / self.num_train_timesteps for t in timesteps_linear]
        sigmas_shifted = [
            math.exp(1.0) / (math.exp(1.0) + (1.0 / s - 1.0)) for s in sigmas_linear
        ]
        sigmas_final = self._stretch_to_terminal(sigmas_shifted)
        timesteps = [s * self.num_train_timesteps for s in sigmas_final]
        sigmas_with_zero = sigmas_final + [0.0]
        sigmas_arr = mx.array(sigmas_with_zero, dtype=mx.float32)
        timesteps_arr = mx.array(timesteps, dtype=mx.float32)
        return sigmas_arr, timesteps_arr

    def _reset_step_state(self):
        self._prev_v = None
        self._prev_h = 0.0

    def step(self, noise, timestep, latents, **kwargs):
        sigmas = kwargs.get("sigmas", self._sigmas)
        t = int(timestep)
        if t == 0:
            self._reset_step_state()
        v = noise.astype(latents.dtype)
        s_cur = sigmas[t]
        s_next = sigmas[t + 1]
        h = (s_cur - s_next).astype(latents.dtype)
        if self._prev_v is None or float(self._prev_h) <= 0.0:
            out = latents - h * v
        else:
            prev_h = mx.array(self._prev_h, dtype=latents.dtype)
            corr = (h * h / (2.0 * prev_h)) * (v - self._prev_v)
            out = latents - h * v - corr
        self._prev_v = v
        self._prev_h = float(h)
        return out

    def scale_model_input(self, latents, t):
        return latents


def register_image_schedulers():
    # Idempotent: safe to call from both the parent server process and the
    # image subprocess (registration does not cross process boundaries).
    for key in REGISTRY_KEYS:
        register_contrib(DPMPP2MSdeScheduler, key)
    logger.info("registered contrib image schedulers: %s", ", ".join(REGISTRY_KEYS))
