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
from ..ltx2.generate import STAGE_1_SIGMAS, STAGE_2_SIGMAS
from ..ltx2.scheduler import ltx2_scheduler

logger = logging.getLogger(__name__)

# 两阶段 distilled sigma 列表（复用 ltx2 STAGE_1/2，AR §4.6）。
# stage1 用 STAGE_1_SIGMAS，stage2 用 STAGE_2_SIGMAS，
# stage2 noise_scale = STAGE_2_SIGMAS[0]。
DISTILLED_STAGE_1_SIGMAS = list(STAGE_1_SIGMAS)
DISTILLED_STAGE_2_SIGMAS = list(STAGE_2_SIGMAS)

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
