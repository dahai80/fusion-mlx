import logging
import math

import mlx.core as mx

logger = logging.getLogger(__name__)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class FlowMatchEulerScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        shift: float | None = None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.shift_override = shift
        self.sigmas = None
        self.timesteps = None
        self._step_index = 0
        self._mu = None

    def _time_shift(self, mu: float, sigma: float, t: mx.array) -> mx.array:
        return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int,
        image_seq_len: int,
        shift: float | None = None,
    ) -> None:
        if shift is None:
            shift = self.shift_override
        if shift is None:
            mu = calculate_shift(
                image_seq_len,
                self.base_image_seq_len,
                self.max_image_seq_len,
                self.base_shift,
                self.max_shift,
            )
        else:
            mu = shift
        self._mu = mu
        sigmas = mx.linspace(1.0, 0.0, num_inference_steps + 1, dtype=mx.float32)
        sigmas = self._time_shift(mu, 1.0, sigmas)
        sigmas = mx.concatenate([sigmas, mx.zeros(1, dtype=mx.float32)])
        self.timesteps = sigmas[:-1] * self.num_train_timesteps
        self.sigmas = sigmas
        self._step_index = 0
        logger.info(
            "SD3 scheduler steps=%d seq_len=%d mu=%.4f sigma[0]=%.4f sigma[-2]=%.4f",
            num_inference_steps, image_seq_len, mu,
            float(sigmas[0]), float(sigmas[-2]),
        )

    @property
    def step_index(self) -> int:
        return self._step_index

    def step(self, model_output: mx.array, sample: mx.array) -> mx.array:
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        dt = sigma_next - sigma
        prev_sample = sample.astype(mx.float32) + dt * model_output.astype(mx.float32)
        self._step_index += 1
        return prev_sample
