import logging

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
        self.num_inference_steps = num_inference_steps
        sigmas = self.sigmas_full
        n = self.num_train_timesteps
        if self.timestep_spacing == "linspace":
            timesteps = mx.linspace(0, n - 1, num_inference_steps, dtype=mx.float32)
        elif self.timestep_spacing == "trailing":
            ratio = mx.arange(num_inference_steps, 0, -1, dtype=mx.float32) / (
                num_inference_steps + 1
            )
            timesteps = ratio * n
        else:
            ratio = (
                mx.arange(0, num_inference_steps, dtype=mx.float32)
                / num_inference_steps
            )
            timesteps = (1.0 - ratio) * (n - 1)
            timesteps = timesteps + self.steps_offset
        idx = (timesteps * (len(sigmas) - 1) / n).astype(mx.int32)
        idx = mx.clip(idx, 0, len(sigmas) - 1)
        if self.interpolation_type == "linear":
            t_max = float(timesteps.max())
            t_min = float(timesteps.min())
            sig_idx = sigmas[idx]
            if t_max > t_min:
                sigma_interp = (timesteps - t_min) / (t_max - t_min) * (
                    float(sig_idx.max()) - float(sig_idx.min())
                ) + float(sig_idx.min())
            else:
                sigma_interp = sig_idx
        else:
            sigma_interp = sigmas[idx]
        sigma_interp = mx.concatenate([sigma_interp, mx.zeros(1, dtype=mx.float32)])
        self.sigmas = sigma_interp
        self.timesteps = timesteps
        self._step_index = 0
        logger.info(
            "SDXL scheduler steps=%d spacing=%s sigma[0]=%.4f sigma[-2]=%.4f",
            num_inference_steps,
            self.timestep_spacing,
            float(self.sigmas[0]),
            float(self.sigmas[-2]),
        )

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
