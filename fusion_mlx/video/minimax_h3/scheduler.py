# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 调度器：rectified-flow Euler（eta=0）+ 指数 sigma shift。
# 源码（权威）：diffusers main src/diffusers/schedulers/scheduling_minimax_h3.py
#
# 与 FlowMatchEulerDiscreteScheduler 不兼容的三点（逐条对齐官方源码注释）：
#   1. 速度符号反转：transformer 预测 data-ward velocity，x0 = x_t + sigma*v（PLUS），
#      与 diffusers 默认 x0 = x_t - sigma*v 相反。早期移植曾误用 MINUS 致 motion 抖动
#      （去噪向错误方向走，在不动点附近振荡），对照官方源码校正为 PLUS。
#   2. timestep t = 1 - sigma，t=1 为干净；scheduler.timesteps = 1 - sigmas[:-1]。
#   3. sigma 网格 linspace(1,0,N)，终点 0 在请求步数内；shift 后 unique_consecutive 折叠重复。
#
# 每请求两个实例：video shift=12.0 / audio shift=3.0。无依赖 PyTorch，纯 MLX。
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


class MiniMaxH3Scheduler:
    def __init__(self, shift: float = 12.0):
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self._shift = float(shift)
        self.num_inference_steps = None
        self.sigmas = None
        self.timesteps = None
        self._step_index = None
        self._begin_index = None

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def set_shift(self, shift: float):
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self._shift = float(shift)

    def set_timesteps(self, num_inference_steps=None, sigmas=None):
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(
                    "set_timesteps requires explicit sigmas or num_inference_steps >= 2, "
                    f"got {num_inference_steps}"
                )
            base = mx.linspace(1.0, 0.0, int(num_inference_steps), dtype=mx.float32)
            sigmas = self._shift * base / (1 + (self._shift - 1) * base)
            sigmas = _unique_consecutive(sigmas)
        else:
            sigmas = mx.asarray(sigmas, dtype=mx.float32).flatten()
            if (
                sigmas.shape[0] < 2
                or not bool(mx.all(sigmas[1:] < sigmas[:-1]))
                or float(sigmas[-1]) != 0.0
            ):
                raise ValueError("sigmas must be strictly decreasing ending at 0.0")

        self.sigmas = sigmas
        self.timesteps = 1.0 - sigmas[:-1]
        self.num_inference_steps = int(self.timesteps.shape[0])
        self._step_index = None
        self._begin_index = None
        logger.info(
            "minimax_h3 scheduler: shift=%.1f steps=%d sigmas[0]=%.4f sigmas[-1]=%.4f",
            self._shift,
            self.num_inference_steps,
            float(sigmas[0]),
            float(sigmas[-1]),
        )

    def index_for_timestep(self, timestep: float) -> int:
        target = mx.array(timestep, dtype=mx.float32)
        eq = self.timesteps == target
        idx = int(mx.argmax(eq))
        if not bool(eq[idx]):
            raise ValueError(
                f"timestep {timestep} not in self.timesteps; use scheduler.timesteps values"
            )
        return idx

    def scale_noise(self, sample, timestep, noise):
        if not isinstance(timestep, mx.array):
            timestep = mx.array(timestep, dtype=sample.dtype)
        timestep = timestep.astype(sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = mx.expand_dims(timestep, -1)
        return timestep * sample + (1.0 - timestep) * noise

    def step(self, model_output, timestep, sample):
        if isinstance(timestep, int):
            raise ValueError(
                "integer timestep not supported; pass one of scheduler.timesteps"
            )

        if self._step_index is None:
            if self._begin_index is None:
                self._step_index = self.index_for_timestep(float(timestep))
            else:
                self._step_index = self._begin_index

        if not isinstance(timestep, mx.array):
            timestep = mx.array(timestep, dtype=sample.dtype)
        timestep = timestep.astype(sample.dtype)
        sigma_from_timestep = 1.0 - timestep
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = mx.expand_dims(sigma_from_timestep, -1)
        denoised = (
            sample + sigma_from_timestep * model_output
        )  # data-ward velocity (官方源码 PLUS)

        compute_dtype = (
            mx.float32 if sample.dtype in (mx.float16, mx.bfloat16) else sample.dtype
        )
        sigma = mx.array(self.sigmas[self._step_index], dtype=compute_dtype)
        sigma_next = mx.array(self.sigmas[self._step_index + 1], dtype=compute_dtype)
        ratio = sigma_next / sigma
        prev_sample = ratio * sample.astype(compute_dtype) + (
            1.0 - ratio
        ) * denoised.astype(compute_dtype)
        prev_sample = prev_sample.astype(sample.dtype)

        self._step_index += 1
        return prev_sample


def _unique_consecutive(x: mx.array) -> mx.array:
    keep = [True] + [bool(x[i] != x[i - 1]) for i in range(1, x.shape[0])]
    idx = [i for i, k in enumerate(keep) if k]
    return x[mx.array(idx, dtype=mx.int32)]
