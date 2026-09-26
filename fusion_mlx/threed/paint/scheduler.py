# SPDX-License-Identifier: Apache-2.0
# DDIM scheduler (v_prediction, rescale_betas_zero_snr, trailing spacing) for
# the Hunyuan3D-2.1 paint pipeline. Pure-MLX/numpy, no diffusers dep.
# Config (paint/config.json scheduler):
#   beta_start=0.00085, beta_end=0.012, scaled_linear schedule,
#   num_train_timesteps=1000, prediction_type=v_prediction,
#   set_alpha_to_one=True, steps_offset=1, timestep_spacing=trailing,
#   rescale_betas_zero_snr=True, clip_sample=False.
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _rescale_zero_snr(alphas_cumprod: np.ndarray) -> np.ndarray:
    # diffusers rescale_betas_zero_snr: normalize so alpha_cumprod[0]=1, then
    # rescale by the max sqrt (a near-noop for scaled_linear that guarantees
    # the SNR floor). Returns rescaled alphas_cumprod; betas recomputed below.
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    sqrt_acp = np.sqrt(alphas_cumprod)
    sqrt_acp = sqrt_acp / sqrt_acp.max()
    return sqrt_acp**2


class DDIMScheduler:
    # Trailing spacing: timesteps = arange(steps, 0, -1) * step_ratio rounded,
    # i.e. the last `steps` train timesteps descending. steps_offset shifts +1.
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        set_alpha_to_one: bool = True,
        steps_offset: int = 1,
        timestep_spacing: str = "trailing",
        rescale_betas_zero_snr: bool = True,
        clip_sample: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.timestep_spacing = timestep_spacing
        self.clip_sample = clip_sample
        self.prediction_type = "v_prediction"

        if beta_schedule == "linear":
            betas = np.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=np.float64
            )
        elif beta_schedule == "scaled_linear":
            betas = (
                np.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=np.float64,
                )
                ** 2
            )
        else:
            raise ValueError(f"unknown beta_schedule {beta_schedule}")

        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        if rescale_betas_zero_snr:
            # Rescale alphas_cumprod to a zero-SNR floor, then rederive betas.
            self.alphas_cumprod = _rescale_zero_snr(self.alphas_cumprod)
            betas = 1.0 - self.alphas_cumprod / np.concatenate(
                [[1.0], self.alphas_cumprod[:-1]]
            )
            self.alphas = 1.0 - betas
            self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.final_alpha_cumprod = (
            np.array(1.0, dtype=np.float64)
            if set_alpha_to_one
            else self.alphas_cumprod[0]
        )
        self.init_noise_sigma = 1.0
        logger.debug(
            "DDIMScheduler: %d train steps, final_alpha_cumprod=%.6f, zero_snr=%s",
            num_train_timesteps,
            float(self.final_alpha_cumprod),
            rescale_betas_zero_snr,
        )

    def set_timesteps(self, steps: int) -> np.ndarray:
        # Trailing: last `steps` train timesteps, descending (high->low for
        # denoise). Clamp to [0, num_train-1] so index into alphas_cumprod is
        # valid (trailing can yield num_train itself, which is OOB by one).
        if self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / steps
            ts = (
                np.arange(self.num_train_timesteps, 0, -step_ratio)
                .round()
                .astype(np.int64)
            )
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps / steps
            ts = (np.arange(0, steps, 1) * step_ratio).round().astype(
                np.int64
            ) + self.steps_offset
        else:
            step_ratio = self.num_train_timesteps / steps
            ts = (
                np.arange(self.num_train_timesteps, 0, -step_ratio)
                .round()
                .astype(np.int64)
            )
            ts -= 1
        ts = np.clip(ts, 0, self.num_train_timesteps - 1)
        self.timesteps = ts
        return ts

    def _alpha_prod_t(self, t: int) -> float:
        if t < 0:
            return float(self.final_alpha_cumprod)
        return float(self.alphas_cumprod[t])

    def step(
        self, model_output: np.ndarray, t: int, sample: np.ndarray, eta: float = 0.0
    ):
        # DDIM deterministic (eta=0). v_prediction: x0 = alpha_sqrt * sample -
        # sqrt_one_minus_alpha * model_output; eps = sqrt_one_minus_alpha *
        # sample + alpha_sqrt * model_output.
        prev_t = t - self.num_train_timesteps // len(self.timesteps)
        alpha_prod_t = self._alpha_prod_t(t)
        alpha_prod_t_prev = self._alpha_prod_t(prev_t)
        sqrt_alpha = alpha_prod_t**0.5
        sqrt_one_minus_alpha = (1.0 - alpha_prod_t) ** 0.5

        if self.prediction_type == "v_prediction":
            pred_x0 = sqrt_alpha * sample - sqrt_one_minus_alpha * model_output
            pred_eps = sqrt_one_minus_alpha * sample + sqrt_alpha * model_output
        elif self.prediction_type == "epsilon":
            pred_eps = model_output
            pred_x0 = (sample - sqrt_one_minus_alpha * pred_eps) / sqrt_alpha
        else:
            raise ValueError(f"unknown prediction_type {self.prediction_type}")

        if self.clip_sample:
            pred_x0 = np.clip(pred_x0, -1.0, 1.0)

        sqrt_alpha_prev = alpha_prod_t_prev**0.5
        sqrt_one_minus_alpha_prev = (1.0 - alpha_prod_t_prev) ** 0.5
        # eta=0 -> deterministic DDIM
        pred_prev = sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * pred_eps
        return pred_prev

    def add_noise(self, original: np.ndarray, noise: np.ndarray, t: int) -> np.ndarray:
        alpha_prod_t = self._alpha_prod_t(t)
        sqrt_alpha = alpha_prod_t**0.5
        sqrt_one_minus = (1.0 - alpha_prod_t) ** 0.5
        return sqrt_alpha * original + sqrt_one_minus * noise


def make_scheduler_from_config(sched_cfg: dict) -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        beta_start=sched_cfg.get("beta_start", 0.00085),
        beta_end=sched_cfg.get("beta_end", 0.012),
        beta_schedule=sched_cfg.get("beta_schedule", "scaled_linear"),
        set_alpha_to_one=sched_cfg.get("set_alpha_to_one", True),
        steps_offset=sched_cfg.get("steps_offset", 1),
        timestep_spacing=sched_cfg.get("timestep_spacing", "trailing"),
        rescale_betas_zero_snr=sched_cfg.get("rescale_betas_zero_snr", True),
        clip_sample=sched_cfg.get("clip_sample", False),
    )
