# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of the LTX-Video 0.9.x RectifiedFlowScheduler
# (ltx_video/schedulers/rf.py, MIT licensed). No torch / diffusers dependency.
#
# Flow-matching sampler. set_timesteps builds a uniform linspace, applies the
# SD3 resolution-dependent shift (token-count based) then stretches the schedule
# to a 0.1 terminal. step() is a plain Euler update z_{t-1} = z_t - dt * v where
# dt is the gap to the next lower sigma (0 for the final step).

import logging
import math

import mlx.core as mx

logger = logging.getLogger(__name__)

_T_EPS = 1e-6
_MIN_TOKENS = 1024
_MAX_TOKENS = 4096
_MIN_SHIFT = 0.95
_MAX_SHIFT = 2.05
_TARGET_TERMINAL = 0.1


def _time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def _normal_shift(n_tokens):
    m = (_MAX_SHIFT - _MIN_SHIFT) / (_MAX_TOKENS - _MIN_TOKENS)
    b = _MIN_SHIFT - m * _MIN_TOKENS
    return m * n_tokens + b


def _stretch_to_terminal(shifts, terminal=_TARGET_TERMINAL):
    one_minus_z = 1.0 - shifts
    scale = one_minus_z[-1] / (1.0 - terminal)
    return 1.0 - (one_minus_z / scale)


class RectifiedFlowScheduler:
    def __init__(
        self,
        num_train_timesteps=1000,
        shifting="SD3",
        target_shift_terminal=_TARGET_TERMINAL,
        sampler="Uniform",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shifting = shifting
        self.target_shift_terminal = target_shift_terminal
        self.sampler = sampler
        self.init_noise_sigma = 1.0
        self.timesteps = None
        self.sigmas = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps, samples_shape=None):
        n = min(self.num_train_timesteps, num_inference_steps)
        t = mx.linspace(1.0, 1.0 / n, n, dtype=mx.float32)
        tokens = "-"
        if self.shifting == "SD3" and samples_shape is not None:
            m = 1
            for d in list(samples_shape)[2:]:
                m *= int(d)
            tokens = str(m)
            shift = _normal_shift(m)
            t = mx.array(
                [_time_shift(shift, 1.0, float(tv)) for tv in t.tolist()],
                dtype=mx.float32,
            )
            t = _stretch_to_terminal(t, self.target_shift_terminal)
        self.timesteps = t
        self.sigmas = t
        self.num_inference_steps = n
        logger.info(
            "scheduler: steps=%d tokens=%s t0=%.4f tN=%.4f",
            n,
            tokens,
            float(t[0]),
            float(t[-1]),
        )
        return t

    def scale_model_input(self, sample, timestep=None):
        return sample

    def step(self, model_output, timestep, sample):
        t = (
            mx.array(timestep, dtype=mx.float32)
            if not isinstance(timestep, mx.array)
            else timestep.astype(mx.float32)
        )
        padded = mx.concatenate([self.timesteps, mx.zeros((1,), dtype=mx.float32)])
        cand = mx.where(padded < (t - _T_EPS), padded, mx.zeros_like(padded))
        next_t = mx.max(cand)
        dt = t - next_t
        return sample - dt * model_output

    def add_noise(self, original, noise, timesteps):
        sigma = timesteps
        alphas = 1.0 - sigma
        return alphas * original + sigma * noise
