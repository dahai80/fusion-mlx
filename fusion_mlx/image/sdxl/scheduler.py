import logging

import numpy as _np

import mlx.core as mx

logger = logging.getLogger(__name__)


class SDXLEulerDiscreteScheduler:
    # Port of diffusers EulerDiscreteScheduler for SDXL epsilon prediction.
    # betas = scaled_linear schedule; sigmas derived from cumulative alpha;
    # inference sigmas interpolated across num_inference_steps.

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        timestep_spacing: str = "leading",
        steps_offset: int = 1,
        interpolation_type: str = "linear",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        self.interpolation_type = interpolation_type
        self.sigmas = None
        self.timesteps = None
        self._step_index = 0
        self._init_sigmas()

    def _betas(self) -> mx.array:
        if self.beta_schedule == "scaled_linear":
            return (
                mx.linspace(
                    self.beta_start**0.5,
                    self.beta_end**0.5,
                    self.num_train_timesteps,
                    dtype=mx.float32,
                )
                ** 2
            )
        if self.beta_schedule == "linear":
            return mx.linspace(
                self.beta_start,
                self.beta_end,
                self.num_train_timesteps,
                dtype=mx.float32,
            )
        logger.warning(
            "SDXL unknown beta_schedule=%s, fallback linear", self.beta_schedule
        )
        return mx.linspace(
            self.beta_start, self.beta_end, self.num_train_timesteps, dtype=mx.float32
        )

    def _init_sigmas(self) -> None:
        betas = self._betas()
        alphas = 1.0 - betas
        alphas_cumprod = mx.cumprod(alphas)
        self.sigmas_full = mx.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)

    def set_timesteps(self, num_inference_steps: int) -> None:
        # Mirror diffusers EulerDiscreteScheduler.set_timesteps exactly
        # (numpy compute, then wrap in mx.array). The prior MLX impl derived
        # timesteps with non-integer ratios and STRETCHED sigmas onto the
        # [min,max] range instead of np.interp lookup, giving sigma[0]=14.6
        # vs diffusers 11.0 (33% wrong) for leading spacing. See issue #488.
        self.num_inference_steps = num_inference_steps
        sigmas_full = _np.array(self.sigmas_full, dtype=_np.float32)
        n = self.num_train_timesteps
        if self.timestep_spacing == "linspace":
            timesteps = _np.linspace(
                0, n - 1, num_inference_steps, dtype=_np.float32
            )[::-1].copy()
        elif self.timestep_spacing == "trailing":
            step_ratio = n / num_inference_steps
            timesteps = (
                _np.arange(n, 0, -step_ratio).round().copy().astype(_np.float32)
            )
            timesteps -= 1
        else:
            # leading
            step_ratio = n // num_inference_steps
            timesteps = (
                (_np.arange(0, num_inference_steps) * step_ratio)
                .round()[::-1]
                .copy()
                .astype(_np.float32)
            )
            timesteps += self.steps_offset
        if self.interpolation_type == "linear":
            sigma_interp = _np.interp(
                timesteps, _np.arange(0, len(sigmas_full)), sigmas_full
            )
        else:
            sigma_interp = sigmas_full[timesteps.astype(_np.int32)]
        sigma_interp = _np.concatenate(
            [sigma_interp, _np.zeros(1, dtype=_np.float32)]
        ).astype(_np.float32)
        self.sigmas = mx.array(sigma_interp)
        self.timesteps = mx.array(timesteps)
        self._step_index = 0
        self._scale_called = False
        logger.info(
            "SDXL scheduler steps=%d spacing=%s sigma[0]=%.4f sigma[-2]=%.4f",
            num_inference_steps,
            self.timestep_spacing,
            float(self.sigmas[0]),
            float(self.sigmas[-2]),
        )

    @property
    def init_noise_sigma(self) -> float:
        # diffusers: leading -> sqrt(max_sigma^2 + 1);
        # linspace/trailing -> max_sigma.
        max_sigma = float(self.sigmas.max())
        if self.timestep_spacing in ("linspace", "trailing"):
            return max_sigma
        return (max_sigma**2 + 1) ** 0.5

    def scale_model_input(
        self, sample: mx.array, step_index: int = None
    ) -> mx.array:
        # diffusers scale_model_input: sample / (sigma^2 + 1)^0.5
        if step_index is None:
            step_index = self._step_index
        sigma = float(self.sigmas[step_index])
        self._scale_called = True
        return sample / ((sigma**2 + 1) ** 0.5)

    @property
    def step_index(self) -> int:
        return self._step_index

    def step(self, model_output: mx.array, sample: mx.array) -> mx.array:
        sigma = float(self.sigmas[self._step_index])
        sigma_next = float(self.sigmas[self._step_index + 1])
        if self.prediction_type == "epsilon":
            pred_original = sample - sigma * model_output
            derivative = (sample - pred_original) / max(sigma, 1e-6)
        elif self.prediction_type == "v_prediction":
            pred_original = model_output * (
                -sigma / (sigma**2 + 1) ** 0.5
            ) + sample * (1.0 / (sigma**2 + 1) ** 0.5)
            derivative = (sample - pred_original) / max(sigma, 1e-6)
        else:
            pred_original = model_output
            derivative = model_output
        dt = sigma_next - sigma
        prev_sample = sample.astype(mx.float32) + dt * derivative.astype(mx.float32)
        self._step_index += 1
        return prev_sample
