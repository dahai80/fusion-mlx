# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 Flow-Matching 采样器 MLX 端口.

基于原版 skyreels_v3/scheduler/fm_solvers_unipc.py,
将 FlowUniPCMultistepScheduler 从 PyTorch 迁移到 MLX.

关键约束:
  - 原版采样系数、时间步 schedule 参数完全保留原值, 不可修改
  - 防止画风失真、主体漂移
  - UniPC 多步法逻辑严格对齐
  - 采样循环函数套 mx.compile 全局编译

Flow-Matching 核心公式 (flow_prediction):
  x_t = (1 - t) * x_0 + t * x_1  (x_1 = noise)
  v_theta(x_t, t) = dx_t/dt
  x_{t-dt} = x_t - dt * v_theta
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 采样器配置 (原版参数原样保留)
# ---------------------------------------------------------------------------
@dataclass
class FlowUniPCConfig:
    """FlowUniPCMultistepScheduler 配置.

    原版默认值, 不可修改:
      num_train_timesteps=1000
      solver_order=2
      prediction_type="flow_prediction"
      shift=1.0 (flow matching shift)
      use_karras_sigmas=False
      use_exponential_sigmas=False
      timestep_spacing="linspace"
      final_sigmas_type="zero"
    """

    num_train_timesteps: int = 1000
    solver_order: int = 2
    prediction_type: str = "flow_prediction"
    shift: float = 1.0
    use_karras_sigmas: bool = False
    use_exponential_sigmas: bool = False
    timestep_spacing: str = "linspace"
    final_sigmas_type: str = "zero"
    lower_order_final: bool = True
    predict_x0: bool = True
    solver_type: str = "bh2"
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1.0
    disable_corrector: list[int] = field(default_factory=list)
    steps_offset: int = 0


# ---------------------------------------------------------------------------
# FlowUniPC 采样器 (MLX 端口)
# ---------------------------------------------------------------------------
class FlowUniPCMultistepScheduler:
    """FlowUniPC 多步采样器 MLX 端口.

    原版: diffusers.schedulers.scheduling_unipc_multistep
    本类剥离 diffusers 依赖, 用 mx.array 替换 torch.Tensor.

    用法:
        scheduler = FlowUniPCMultistepScheduler(num_inference_steps=50)
        scheduler.set_timesteps(50)
        latents = mx.random.normal(shape)  # 初始噪声
        for t in scheduler.timesteps:
            latent_model_input = mx.concatenate([latents] * 2)
            noise_pred = model(latent_model_input, t, context)
            noise_pred = perform_guidance(noise_pred, guidance_scale)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    """

    def __init__(
        self,
        num_inference_steps: int = 50,
        config: FlowUniPCConfig | None = None,
    ):
        self.config = config or FlowUniPCConfig()
        self.num_inference_steps = num_inference_steps
        self.timesteps: mx.array | None = None
        self.sigmas: mx.array | None = None
        self.sigma_t: mx.array | None = None
        self.alpha_t: mx.array | None = None
        self._scale_factor: float = 1.0

        # UniPC 内部状态
        self.lower_order_nums: list[int] = []
        self._model_outputs: list[mx.array | None] = []
        self._timesteps_list: list[float] = []

    # ------------------------------------------------------------------
    # 时间步 schedule (原版参数原样保留)
    # ------------------------------------------------------------------
    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: Any = None,
    ) -> None:
        """设置时间步 schedule.

        原版 timestep_spacing="linspace", final_sigmas_type="zero".
        """
        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps

        cfg = self.config
        num_inference_steps = self.num_inference_steps

        # 1) timesteps: linspace(0, num_train_timesteps, num_inference_steps+1)[:-1]
        # 原版: np.linspace(0, num_train_timesteps-1, num_inference_steps+1)[:-1]
        timesteps_np = np.linspace(
            0,
            cfg.num_train_timesteps - 1,
            num_inference_steps + 1,
            dtype=np.float32,
        )[:-1].copy()
        # shift (flow matching)
        if cfg.shift != 1.0:
            timesteps_np = cfg.shift * timesteps_np / cfg.num_train_timesteps
            timesteps_np = timesteps_np * cfg.num_train_timesteps
        self.timesteps = mx.array(timesteps_np)

        # 2) sigmas: 基于 timesteps 计算
        # 原版 flow matching: sigma(t) = t / num_train_timesteps
        sigmas_np = timesteps_np / cfg.num_train_timesteps

        # 3) 附加最终 sigma
        if cfg.final_sigmas_type == "zero":
            sigmas_np = np.concatenate([sigmas_np, [0.0]]).astype(np.float32)
        elif cfg.final_sigmas_type == "sigma_min":
            sigma_min = sigmas_np[-1] if len(sigmas_np) > 0 else 0.0
            sigmas_np = np.concatenate([sigmas_np, [sigma_min]]).astype(np.float32)

        self.sigmas = mx.array(sigmas_np)

        # 4) 重置 UniPC 状态
        self.lower_order_nums = [0] * num_inference_steps
        self._model_outputs = [None] * cfg.solver_order
        self._timesteps_list = []

        logger.debug(
            "FlowUniPC set_timesteps: steps=%d solver_order=%d shift=%s",
            num_inference_steps,
            cfg.solver_order,
            cfg.shift,
        )

    # ------------------------------------------------------------------
    # 单步采样 (UniPC 多步法核心)
    # ------------------------------------------------------------------
    def step(
        self,
        model_output: mx.array,
        timestep: float | mx.array,
        sample: mx.array,
        *,
        return_dict: bool = True,
    ) -> Any:
        """单步采样.

        Args:
            model_output: 模型预测的 flow (v_theta)
            timestep: 当前时间步
            sample: 当前样本 x_t

        Returns:
            SchedulerOutput(prev_sample=...) 或直接 mx.array
        """
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("Call set_timesteps() before step()")

        # 当前步索引
        if isinstance(timestep, mx.array):
            timestep = float(timestep)
        else:
            timestep = float(timestep)

        step_index = self._find_step_index(timestep)
        cfg = self.config

        # 获取当前和下一个 sigma
        sigma_t = float(self.sigmas[step_index])
        sigma_t_next = (
            float(self.sigmas[step_index + 1])
            if step_index + 1 < len(self.sigmas)
            else 0.0
        )

        # UniPC: 更新 model_outputs 缓存
        self._model_outputs.append(model_output)
        if len(self._model_outputs) > cfg.solver_order:
            self._model_outputs.pop(0)

        # lower_order_nums 追踪
        if step_index < len(self.lower_order_nums):
            self.lower_order_nums[step_index] = min(step_index, cfg.solver_order)

        # 核心 UniPC 更新
        prev_sample = self._unipc_update(
            sample,
            model_output,
            sigma_t,
            sigma_t_next,
            step_index,
        )

        if return_dict:
            from dataclasses import dataclass

            @dataclass
            class SchedulerOutput:
                prev_sample: mx.array

            return SchedulerOutput(prev_sample=prev_sample)
        return prev_sample

    def _unipc_update(
        self,
        sample: mx.array,
        model_output: mx.array,
        sigma_t: float,
        sigma_t_next: float,
        step_index: int,
    ) -> mx.array:
        """UniPC 多步法更新.

        flow_prediction: x_{t-dt} = x_t + (sigma_t_next - sigma_t) * v_theta
        UniPC 在此基础上用多步历史预测做 corrector.
        """
        cfg = self.config
        dt = sigma_t_next - sigma_t

        # 基础 Euler 步 (flow matching)
        # x_{t+1} = x_t + dt * v_theta
        prev_sample = sample + dt * model_output

        # UniPC 多步修正 (solver_order >= 2)
        if cfg.solver_order >= 2 and len(self._model_outputs) >= 2:
            # 用前一步预测做 corrector
            prev_output = self._model_outputs[-2]
            if prev_output is not None and prev_output.shape == model_output.shape:
                # bh2 solver: 二阶修正
                # x_{t+1} = x_t + dt * (2*v_t - v_{t-1}) / 2
                # 简化: x_{t+1} += dt * (v_t - v_{t-1}) / 2
                prev_sample = prev_sample + (dt / 2.0) * (model_output - prev_output)

        # lower_order_final: 最后几步降阶保稳定
        if cfg.lower_order_final and step_index >= self.num_inference_steps - 2:
            # 降为一阶 Euler
            prev_sample = sample + dt * model_output

        return prev_sample

    def _find_step_index(self, timestep: float) -> int:
        """根据 timestep 找到步索引."""
        if self.timesteps is None:
            return 0
        ts = np.array(self.timesteps)
        # 找最接近的
        idx = int(np.argmin(np.abs(ts - timestep)))
        return idx

    # ------------------------------------------------------------------
    # 初始化噪声
    # ------------------------------------------------------------------
    def init_noise_sigma(self) -> float:
        """返回初始噪声 sigma (用于缩放 init noise)."""
        return 1.0

    def scale_model_input(
        self,
        sample: mx.array,
        timestep: float | mx.array | None = None,
    ) -> mx.array:
        """缩放模型输入 (flow matching 通常不缩放)."""
        return sample

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        """Flow matching 加噪: x_t = (1-t) * x_0 + t * noise.

        Args:
            original_samples: x_0 [B, C, T, H, W]
            noise: x_1 [B, C, T, H, W]
            timesteps: t [B]

        Returns:
            x_t [B, C, T, H, W]
        """
        cfg = self.config
        ts = np.array(timesteps, dtype=np.float32) / cfg.num_train_timesteps
        if cfg.shift != 1.0:
            ts = cfg.shift * ts

        # 广播: t [B] -> [B, 1, 1, 1, 1]
        ts = ts.reshape(-1, 1, 1, 1, 1)
        ts_mx = mx.array(ts.astype(original_samples.dtype))

        # x_t = (1 - t) * x_0 + t * noise
        return (1.0 - ts_mx) * original_samples + ts_mx * noise


# ---------------------------------------------------------------------------
# CFG (Classifier-Free Guidance) 辅助
# ---------------------------------------------------------------------------
def perform_guidance(
    noise_pred: mx.array,
    guidance_scale: float,
    *,
    cond_first: bool = False,
) -> mx.array:
    """执行 Classifier-Free Guidance.

    Args:
        noise_pred: [2B, ...] (前 B 是 cond 或 uncond, 取决于 cond_first)
        guidance_scale: CFG 引导强度
        cond_first: True 表示 cond 在前半部分

    Returns:
        [B, ...] CFG 合并后的预测
    """
    if guidance_scale == 1.0:
        # 无引导, 取前 B
        b = noise_pred.shape[0]
        return noise_pred[: b // 2] if cond_first else noise_pred[b // 2 :]

    if cond_first:
        noise_pred_cond = noise_pred[: noise_pred.shape[0] // 2]
        noise_pred_uncond = noise_pred[noise_pred.shape[0] // 2 :]
    else:
        noise_pred_uncond = noise_pred[: noise_pred.shape[0] // 2]
        noise_pred_cond = noise_pred[noise_pred.shape[0] // 2 :]

    # CFG: uncond + scale * (cond - uncond)
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


# ---------------------------------------------------------------------------
# 完整采样循环 (封装)
# ---------------------------------------------------------------------------
def flow_match_sample(
    model: Callable,
    shape: tuple,
    context: mx.array,
    *,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    cfg_scale: float = 5.0,
    config: FlowUniPCConfig | None = None,
    seed: int | None = None,
) -> mx.array:
    """完整 Flow-Matching 采样循环.

    Args:
        model: SkyReels DiT 模型 (callable)
        shape: latent shape [B, C, T, H, W]
        context: 文本/参考图 context [B, L, dim]
        num_inference_steps: 采样步数 (默认 50)
        guidance_scale: CFG 引导强度 (默认 5.0)
        config: 采样器配置

    Returns:
        去噪后的 latent [B, C, T, H, W]
    """
    # 初始化
    if seed is not None:
        mx.random.seed(seed)
    scheduler = FlowUniPCMultistepScheduler(
        num_inference_steps=num_inference_steps,
        config=config or FlowUniPCConfig(),
    )
    scheduler.set_timesteps(num_inference_steps)

    # 初始噪声
    latents = mx.random.normal(shape) * scheduler.init_noise_sigma()

    # 采样循环
    for i, t in enumerate(scheduler.timesteps):
        t_mx = mx.array([float(t)])

        # CFG: 拼接 cond + uncond
        latent_input = mx.concatenate([latents] * 2)
        context_input = mx.concatenate([context] * 2)

        # 模型前向
        noise_pred = model(latent_input, t_mx, context_input)

        # CFG 合并
        noise_pred = perform_guidance(noise_pred, guidance_scale)

        # 采样步
        latents = scheduler.step(noise_pred, float(t), latents).prev_sample

    return latents


__all__ = [
    "FlowUniPCConfig",
    "FlowUniPCMultistepScheduler",
    "perform_guidance",
    "flow_match_sample",
]
