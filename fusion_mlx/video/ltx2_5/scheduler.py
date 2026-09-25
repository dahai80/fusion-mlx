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
import numpy as np

from ..ltx2.denoise import denoise_dev_av, denoise_distilled, denoise_res2s_av
from ..ltx2.generate import STAGE_1_SIGMAS, STAGE_2_SIGMAS
from ..ltx2.scheduler import ltx2_scheduler

logger = logging.getLogger(__name__)

# 两阶段 distilled sigma 列表（复用 ltx2 STAGE_1/2，AR §4.6）。
# stage1 用 STAGE_1_SIGMAS，stage2 用 STAGE_2_SIGMAS，
# stage2 noise_scale = STAGE_2_SIGMAS[0]。
DISTILLED_STAGE_1_SIGMAS = list(STAGE_1_SIGMAS)
DISTILLED_STAGE_2_SIGMAS = list(STAGE_2_SIGMAS)

# #968 EXPERIMENT (default OFF — verified to NOT improve semantics, kept for
# reproducibility): swap in the densified distilled tables (2x steps per stage,
# linear midpoint interpolation). The distilled transformer is step-count
# agnostic (velocity -> x0 -> renoise), so densification only trades time for
# finer sigma resolution.
import os as _os

if _os.environ.get("FUSION_LTX25_DENSIFY_SIGMAS") == "1":
    from ..ltx2.generate import STAGE_1_SIGMAS_DENSE, STAGE_2_SIGMAS_DENSE

    DISTILLED_STAGE_1_SIGMAS = list(STAGE_1_SIGMAS_DENSE)
    DISTILLED_STAGE_2_SIGMAS = list(STAGE_2_SIGMAS_DENSE)
    logger.info(
        "ltx2_5: DENSIFIED sigma tables active (stage1=%d steps, stage2=%d steps)",
        len(DISTILLED_STAGE_1_SIGMAS) - 1,
        len(DISTILLED_STAGE_2_SIGMAS) - 1,
    )

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


# dev（Full/SFT）变体调度：官方无 baked 表（diffusers main ltx2 pipeline 不传
# sigmas，走 FlowMatchEulerDiscreteScheduler 动态 shifting）。此函数是
# set_timesteps(use_dynamic_shifting=True, mu=calculate_shift(tokens)) 的精确
# MLX 侧移植（对拍验证：与 diffusers 0.39 输出逐值一致）。
# 官方 anchors 来自 LTX-2.5-Diffusers scheduler_config.json（base/max_shift
# 与 distilled 一致，1024→4096 token 锚点）。
_MAX_SHIFT_ANCHOR = 4096


def dev_sigmas(steps: int, num_tokens: int | None = None) -> list[float]:
    import math as _math

    tokens = num_tokens if num_tokens is not None else _MAX_SHIFT_ANCHOR
    base_seq, max_seq = 1024, 4096
    base_shift, max_shift = 0.95, 2.05
    m = (max_shift - base_shift) / (max_seq - base_seq)
    mu = tokens * m + (base_shift - m * base_seq)

    # sigma_max=1.0, sigma_min=1/1000（FlowMatchEuler 初始化约定），
    # timesteps = linspace(1000, 1, steps) -> sigmas = t/1000
    sig = np.linspace(1.0, 0.001, steps)
    sig = _math.exp(mu) / (
        _math.exp(mu) + (1 / sig - 1)
    )  # exponential time shift, sigma=1
    sig = np.concatenate([sig, [0.0]])
    return [float(s) for s in sig]


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
