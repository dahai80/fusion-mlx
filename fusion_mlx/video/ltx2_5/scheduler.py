# SPDX-License-Identifier: Apache-2.0
# LTX-2.5 scheduler + denoise reuse layer (P4).
# 2.5 沿用 ltx2_scheduler（同族 sigma shift 公式）与 denoise_distilled /
# denoise_res2s_av 去噪骨架。唯一新增：两阶段 distilled 的 sigma 列表。
# ltx2 generate.py 已硬编码 STAGE_1_SIGMAS / STAGE_2_SIGMAS（distilled 专用），
# 2.5 distilled 复用同一列表（AR §2.1 沿用 ltx2_scheduler）；如真实模型首跑发现
# 2.5 sigma 列表不同，需从 diffusers main 提取 DEV/STAGE sigma 并在此覆写。
#
# UNVERIFIED against real weights (gated 403)。
from __future__ import annotations

import logging

import mlx.core as mx

from ..ltx2.denoise import denoise_dev_av, denoise_distilled, denoise_res2s_av
from ..ltx2.scheduler import ltx2_scheduler

logger = logging.getLogger(__name__)

# 两阶段 distilled sigma 列表。
# LTX-2.5 模型配 RectifiedFlowScheduler + LinearQuadratic sampler
# （embedded_config.json scheduler.sampler="LinearQuadratic"），
# sigma 须由 linear_quadratic_schedule(num_steps) 动态算，非复用 LTX-2 硬编码值。
# 根因（issue #942）：旧值复用 LTX-2 STAGE_2_SIGMAS[0]=0.909375（部分加噪），
# 正确值 Stage2 从 1.0 full-noise 起步。高分辨率/中等 T 崩溃即此错配所致。
#
# linear_quadratic_schedule 移植自 Lightricks/ltx-video rf.py（同公式）：
#   threshold_noise=0.025, linear_steps=num_steps//2, 末尾补 0.0 作终止符。
def _linear_quadratic_schedule(num_steps: int, threshold_noise: float = 0.025) -> list[float]:
    if num_steps == 1:
        return [1.0, 0.0]
    linear_steps = num_steps // 2
    linear_sigma = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    quad_steps = num_steps - linear_steps
    diff = linear_steps - threshold_noise * num_steps
    q_coef = diff / (linear_steps * quad_steps**2)
    l_coef = threshold_noise / linear_steps - 2 * diff / (quad_steps**2)
    const = q_coef * (linear_steps**2)
    quad_sigma = [q_coef * (i**2) + l_coef * i + const for i in range(linear_steps, num_steps)]
    sched = linear_sigma + quad_sigma + [1.0]
    sched = [1.0 - x for x in sched]
    return sched[:-1] + [0.0]  # 末尾 0.0 终止符（denoise 循环 num_steps=len-1）

# Stage1=8 步，Stage2=3 步（ltx2_5 generate.py 硬编码步数）
DISTILLED_STAGE_1_SIGMAS = _linear_quadratic_schedule(8)
DISTILLED_STAGE_2_SIGMAS = _linear_quadratic_schedule(3)

# dev 变体 sigma 列表（P9 后续）：真实模型首跑后从 diffusers main 提取，
# 当前为 None 表示 dev 路径未启用（fail visible）。
DEV_SIGMA_VALUES: list[float] | None = None


def ltx2_5_scheduler(
    steps: int,
    num_tokens: int | None = None,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> mx.array:
    # 薄封装 ltx2_scheduler，便于 2.5 侧统一日志与未来覆写 shift 参数。
    logger.info(
        "ltx2_5_scheduler: steps=%d tokens=%s shift=%.2f/%.2f",
        steps,
        num_tokens,
        max_shift,
        base_shift,
    )
    return ltx2_scheduler(
        steps=steps,
        num_tokens=num_tokens,
        max_shift=max_shift,
        base_shift=base_shift,
        stretch=stretch,
        terminal=terminal,
    )


def resolve_distilled_sigmas(stage: int) -> list[float]:
    # stage=1 → STAGE_1，stage=2 → STAGE_2。越界 fail visible。
    if stage == 1:
        return DISTILLED_STAGE_1_SIGMAS
    if stage == 2:
        return DISTILLED_STAGE_2_SIGMAS
    raise ValueError(f"unknown distilled stage {stage!r}, expect 1 or 2")


def resolve_dev_sigmas(stage: int) -> list[float]:
    # P9 dev sigma：当前未提取，fail visible。
    if DEV_SIGMA_VALUES is None:
        raise NotImplementedError(
            "LTX-2.5 dev sigma values not yet extracted from diffusers main "
            "(P9 后续)。distilled 两阶段已就绪，dev 路径待真实模型首跑后补全。"
        )
    return DEV_SIGMA_VALUES


# 去噪入口复用 ltx2（denoise 函数接受任意 LTXModel 子类，含 LTX2_5Model）。
__all__ = [
    "ltx2_5_scheduler",
    "denoise_distilled",
    "denoise_dev_av",
    "denoise_res2s_av",
    "DISTILLED_STAGE_1_SIGMAS",
    "DISTILLED_STAGE_2_SIGMAS",
    "DEV_SIGMA_VALUES",
    "resolve_distilled_sigmas",
    "resolve_dev_sigmas",
]
