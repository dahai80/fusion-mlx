import logging

import mlx.core as mx
import numpy as _np

logger = logging.getLogger(__name__)


class SD2DDIMScheduler:
    # Port of diffusers DDIMScheduler for SD2.1 v_prediction. The SD2.1
    # checkpoint ships scheduler_config.json with _class_name="DDIMScheduler",
    # prediction_type="v_prediction", beta_schedule="scaled_linear",
    # set_alpha_to_one=False, clip_sample=False, steps_offset=1. Euler was a
    # placeholder; DDIM is the faithful schedule. eta=0 (deterministic DDIM),
    # so no variance noise is added.

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "v_prediction",
        timestep_spacing: str = "leading",
        steps_offset: int = 1,
        set_alpha_to_one: bool = False,
        clip_sample: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        self.set_alpha_to_one = set_alpha_to_one
        self.clip_sample = clip_sample
        self.num_inference_steps = None
        self.timesteps = None
        self.sigmas = None
        self.alphas_cumprod = None
        self.final_alpha_cumprod = None
        self._step_index = 0
        self._init_alphas()

    def _init_alphas(self) -> None:
        if self.beta_schedule == "scaled_linear":
            betas = (
                mx.linspace(
                    self.beta_start**0.5,
                    self.beta_end**0.5,
                    self.num_train_timesteps,
                    dtype=mx.float32,
                )
                ** 2
            )
        elif self.beta_schedule == "linear":
            betas = mx.linspace(
                self.beta_start,
                self.beta_end,
                self.num_train_timesteps,
                dtype=mx.float32,
            )
        else:
            logger.warning(
                "SD2 DDIM unknown beta_schedule=%s, fallback linear", self.beta_schedule
            )
            betas = mx.linspace(
                self.beta_start,
                self.beta_end,
                self.num_train_timesteps,
                dtype=mx.float32,
            )
        alphas = 1.0 - betas
        self.alphas_cumprod = mx.cumprod(alphas)
        # set_alpha_to_one=False (SD2.1) -> final_alpha_cumprod = alphas_cumprod[0].
        if self.set_alpha_to_one:
            self.final_alpha_cumprod = mx.array(1.0, dtype=mx.float32)
        else:
            self.final_alpha_cumprod = self.alphas_cumprod[0]

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.num_inference_steps = num_inference_steps
        n = self.num_train_timesteps
        if self.timestep_spacing == "linspace":
            timesteps = (
                _np.linspace(0, n - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(_np.int64)
            )
        elif self.timestep_spacing == "trailing":
            step_ratio = n / num_inference_steps
            timesteps = _np.round(_np.arange(n, 0, -step_ratio)).astype(_np.int64) - 1
        else:
            # leading (SD2.1 default)
            step_ratio = n // num_inference_steps
            timesteps = (
                (_np.arange(0, num_inference_steps) * step_ratio)
                .round()[::-1]
                .copy()
                .astype(_np.int64)
            )
            timesteps += self.steps_offset
        self.timesteps = mx.array(timesteps)
        # Expose a `sigmas` array for the img2img init path (generate.py slices
        # scheduler.sigmas and reads sigmas[0] as the noise scale added to the
        # init latent). DDIM's own step() uses alphas_cumprod, NOT sigmas, so
        # this is consumed only by img2img. Define sigma = sqrt((1-acp)/acp)
        # per timestep (Karras/Euler epsilon-space convention, same as SD15) so
        # sigma_start matches the SD15 img2img init-noise scaling. Trailing 0
        # keeps len(sigmas) == len(timesteps)+1 so the slice
        # sigmas[len-1-eff_steps:] aligns with timesteps[len-eff_steps:].
        acp_np = _np.array(self.alphas_cumprod, dtype=_np.float32)
        sigma_vals = _np.sqrt((1.0 - acp_np[timesteps]) / acp_np[timesteps])
        sigma_vals = _np.concatenate([sigma_vals, [0.0]]).astype(_np.float32)
        self.sigmas = mx.array(sigma_vals)
        self._step_index = 0
        logger.info(
            "SD2 DDIM steps=%d spacing=%s t[0]=%d t[-1]=%d offset=%d sigma_start=%.4f",
            num_inference_steps,
            self.timestep_spacing,
            int(timesteps[0]),
            int(timesteps[-1]),
            self.steps_offset,
            float(sigma_vals[0]),
        )

    @property
    def init_noise_sigma(self) -> float:
        # diffusers DDIMScheduler: init_noise_sigma = 1.0 (no scaling).
        return 1.0

    def scale_model_input(self, sample: mx.array, step_index: int = None) -> mx.array:
        # DDIM does not scale the model input (identity). Diffusers
        # DDIMScheduler.scale_model_input returns sample unchanged.
        return sample

    @property
    def step_index(self) -> int:
        return self._step_index

    def step(self, model_output: mx.array, sample: mx.array) -> mx.array:
        # Deterministic DDIM (eta=0). Port of diffusers DDIMScheduler.step,
        # v_prediction branch. prev_timestep = t - n//steps.
        t = int(self.timesteps[self._step_index])
        prev_t = t - self.num_train_timesteps // self.num_inference_steps
        acp = self.alphas_cumprod
        alpha_prod_t = float(acp[t])
        alpha_prod_t_prev = (
            float(acp[prev_t]) if prev_t >= 0 else float(self.final_alpha_cumprod)
        )
        beta_prod_t = 1.0 - alpha_prod_t
        sqrt_at = alpha_prod_t**0.5
        sqrt_bt = beta_prod_t**0.5
        sqrt_at_prev = alpha_prod_t_prev**0.5
        sqrt_1_at_prev = (1.0 - alpha_prod_t_prev) ** 0.5

        sample_f = sample.astype(mx.float32)
        mo = model_output.astype(mx.float32)
        if self.prediction_type == "v_prediction":
            pred_original = sqrt_at * sample_f - sqrt_bt * mo
            pred_epsilon = sqrt_at * mo + sqrt_bt * sample_f
        elif self.prediction_type == "epsilon":
            pred_original = (sample_f - sqrt_bt * mo) / max(sqrt_at, 1e-8)
            pred_epsilon = mo
        else:
            pred_original = mo
            pred_epsilon = mo
        if self.clip_sample:
            pred_original = mx.clip(pred_original, -1.0, 1.0)
        # eta=0 -> std_dev_t=0, no variance noise.
        pred_sample_direction = sqrt_1_at_prev * pred_epsilon
        prev_sample = sqrt_at_prev * pred_original + pred_sample_direction
        self._step_index += 1
        return prev_sample
