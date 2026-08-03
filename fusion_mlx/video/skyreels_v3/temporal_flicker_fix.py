# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 时序闪烁修复模块.

视频生成常见时序闪烁问题:
  1. 帧间不一致: 相邻帧内容跳变 (采样步未对齐)
  2. 边界闪烁: 视频首尾帧或镜头切换处闪烁
  3. 采样步连贯性: Flow-Matching 各步噪声预测不连贯

修复策略:
  1. 帧间平滑 (Temporal Smoothing)
     - 对相邻帧 latent 做指数移动平均 (EMA)
     - alpha 控制平滑强度 (0=不平滑, 1=完全用前一帧)
     - 仅对中间帧平滑, 首尾帧保留原值

  2. 边界对齐 (Boundary Alignment)
     - 视频续写 (V2V) 时, 续写首帧与输入末帧对齐
     - 对齐方式: 续写首帧 latent = alpha * 输入末帧 latent + (1-alpha) * 续写首帧 latent
     - 防止续写处内容跳变

  3. 采样步连贯 (Step Coherence)
     - Flow-Matching 采样各步噪声预测做低通滤波
     - noise_pred_smoothed = beta * noise_pred_prev + (1-beta) * noise_pred_current
     - beta 控制滤波强度 (0=不滤波, 接近1=强平滑)
     - 防止采样步间预测跳变导致的闪烁

  4. 时序一致性损失 (训练时用, 推理时可选)
     - 推理时可用 latent 空间时序梯度惩罚
     - 对相邻帧 latent 差分做 L2 正则
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 修复配置
# ---------------------------------------------------------------------------
@dataclass
class FlickerFixConfig:
    """时序闪烁修复配置.

    Attributes:
        temporal_ema_alpha: 帧间 EMA 平滑强度 (0~1)
            0 = 不平滑, 1 = 完全用前一帧
            推荐: 0.3 (轻度平滑, 保时序连贯)
        boundary_align_alpha: 边界对齐强度 (0~1)
            0 = 不对齐, 1 = 完全用输入末帧
            推荐: 0.5 (V2V 续写处对齐)
        step_coherence_beta: 采样步连贯滤波强度 (0~1)
            0 = 不滤波, 接近1 = 强平滑
            推荐: 0.1 (轻度滤波, 防采样步跳变)
        enable_temporal_ema: 是否启用帧间 EMA
        enable_boundary_align: 是否启用边界对齐 (V2V 续写)
        enable_step_coherence: 是否启用采样步连贯
        temporal_loss_weight: 时序一致性损失权重 (推理时通常 0)
    """

    temporal_ema_alpha: float = 0.3
    boundary_align_alpha: float = 0.5
    step_coherence_beta: float = 0.1
    enable_temporal_ema: bool = True
    enable_boundary_align: bool = True
    enable_step_coherence: bool = True
    temporal_loss_weight: float = 0.0

    def validate(self) -> None:
        """验证配置参数范围."""
        assert (
            0.0 <= self.temporal_ema_alpha <= 1.0
        ), "temporal_ema_alpha must be in [0, 1]"
        assert (
            0.0 <= self.boundary_align_alpha <= 1.0
        ), "boundary_align_alpha must be in [0, 1]"
        assert (
            0.0 <= self.step_coherence_beta <= 1.0
        ), "step_coherence_beta must be in [0, 1]"
        assert self.temporal_loss_weight >= 0.0, "temporal_loss_weight must be >= 0"


# ---------------------------------------------------------------------------
# 1. 帧间平滑 (Temporal EMA)
# ---------------------------------------------------------------------------
def temporal_ema_smooth(
    latent: mx.array,
    prev_latent: mx.array | None,
    alpha: float = 0.3,
    frame_dim: int = 2,
) -> mx.array:
    """帧间 EMA 平滑.

    对相邻帧做指数移动平均, 仅平滑中间帧, 首尾帧保留原值.

    Args:
        latent: 当前帧 latent [B, C, T, H, W]
        prev_latent: 前一帧 latent (None 表示首帧)
        alpha: 平滑强度 (0=不平滑, 1=完全用前一帧)
        frame_dim: 时序维度 (默认 dim=2)

    Returns:
        平滑后的 latent
    """
    if prev_latent is None or alpha == 0.0:
        return latent

    # EMA: smoothed = alpha * prev + (1 - alpha) * current
    smoothed = alpha * prev_latent + (1.0 - alpha) * latent
    return smoothed


def temporal_ema_batch(
    latent: mx.array,
    alpha: float = 0.3,
    frame_dim: int = 2,
) -> mx.array:
    """对整个 batch 做帧间 EMA 平滑.

    对时序维度做前向 EMA, 仅平滑中间帧, 首帧保留.

    Args:
        latent: [B, C, T, H, W]
        alpha: 平滑强度
        frame_dim: 时序维度索引

    Returns:
        平滑后的 latent
    """
    if alpha == 0.0:
        return latent

    # 沿时序维度做 EMA
    # smoothed[t] = alpha * smoothed[t-1] + (1-alpha) * latent[t]
    # 首帧保留原值
    t_size = latent.shape[frame_dim]
    if t_size <= 1:
        return latent

    smoothed = mx.array(latent)  # 复制
    # 逐帧 EMA (MLX 不支持 in-place, 用循环构建计算图)
    for t in range(1, t_size):
        # 获取前一帧 (沿 frame_dim 切片)
        prev_slice = _slice_frame(smoothed, t - 1, frame_dim)
        curr_slice = _slice_frame(latent, t, frame_dim)
        # EMA 平滑
        ema_slice = alpha * prev_slice + (1.0 - alpha) * curr_slice
        # 写回 (用 concatenate 重建)
        smoothed = _set_frame(smoothed, ema_slice, t, frame_dim)

    return smoothed


def _slice_frame(
    x: mx.array,
    t: int,
    frame_dim: int = 2,
) -> mx.array:
    """沿 frame_dim 切片第 t 帧."""
    # 用 take 沿 frame_dim 取第 t 帧
    return mx.take(x, mx.array([t]), axis=frame_dim).squeeze(frame_dim)


def _set_frame(
    x: mx.array,
    value: mx.array,
    t: int,
    frame_dim: int = 2,
) -> mx.array:
    """设置 x 的第 t 帧 (沿 frame_dim) 为 value.

    MLX 不支持切片赋值, 用 concatenate 重建.
    """
    # 拆分: 前 t 帧 + 新帧 + 后续帧
    # 由于 MLX 切片限制, 用 take + concatenate
    t_size = x.shape[frame_dim]

    # 前 t 帧
    if t > 0:
        before = mx.take(x, mx.arange(t), axis=frame_dim)
    else:
        before = None

    # 新帧 (扩维到匹配 frame_dim)
    new_frame = mx.expand_dims(value, axis=frame_dim)

    # 后续帧
    if t + 1 < t_size:
        after = mx.take(x, mx.arange(t + 1, t_size), axis=frame_dim)
    else:
        after = None

    # 拼接
    parts = []
    if before is not None:
        parts.append(before)
    parts.append(new_frame)
    if after is not None:
        parts.append(after)

    return mx.concatenate(parts, axis=frame_dim)


# ---------------------------------------------------------------------------
# 2. 边界对齐 (Boundary Alignment)
# ---------------------------------------------------------------------------
def boundary_align(
    latent: mx.array,
    input_end_latent: mx.array | None,
    alpha: float = 0.5,
    num_align_frames: int = 3,
    frame_dim: int = 2,
) -> mx.array:
    """边界对齐: 续写首帧与输入末帧对齐.

    V2V 视频续写场景: 续写首帧应与输入视频末帧内容连贯.
    对齐方式: 续写前 num_align_frames 帧做线性插值,
    插值权重从 alpha (首帧) 线性衰减到 0 (第 num_align_frames 帧).

    Args:
        latent: 续写 latent [B, C, T, H, W]
        input_end_latent: 输入视频末帧 latent (None 表示不对齐)
        alpha: 对齐强度 (0=不对齐, 1=完全用输入末帧)
        num_align_frames: 对齐帧数 (前 N 帧做插值)
        frame_dim: 时序维度索引

    Returns:
        对齐后的 latent
    """
    if input_end_latent is None or alpha == 0.0:
        return latent

    t_size = latent.shape[frame_dim]
    align_n = min(num_align_frames, t_size)

    # 对前 align_n 帧做线性插值
    # 权重: alpha -> 0 线性衰减
    aligned = mx.array(latent)
    for t in range(align_n):
        # 插值权重
        w = alpha * (1.0 - t / align_n)
        # 获取当前帧
        curr = _slice_frame(latent, t, frame_dim)
        # 插值: aligned[t] = w * input_end + (1-w) * curr
        aligned_frame = w * input_end_latent + (1.0 - w) * curr
        # 写回
        aligned = _set_frame(aligned, aligned_frame, t, frame_dim)

    return aligned


# ---------------------------------------------------------------------------
# 3. 采样步连贯 (Step Coherence)
# ---------------------------------------------------------------------------
class StepCoherenceFilter:
    """采样步连贯滤波器.

    对 Flow-Matching 采样各步的噪声预测做低通滤波,
    防止采样步间预测跳变导致的闪烁.

    用法:
        filter = StepCoherenceFilter(beta=0.1)
        for step in range(num_steps):
            noise_pred = model(...)
            noise_pred = filter(noise_pred)  # 滤波
            latents = scheduler.step(noise_pred, ...)
    """

    def __init__(self, beta: float = 0.1):
        """
        Args:
            beta: 滤波强度 (0=不滤波, 接近1=强平滑)
        """
        self.beta = beta
        self._prev_noise: mx.array | None = None

    def __call__(self, noise_pred: mx.array) -> mx.array:
        """对当前步噪声预测做低通滤波.

        Args:
            noise_pred: 当前步噪声预测

        Returns:
            滤波后的噪声预测
        """
        if self.beta == 0.0:
            self._prev_noise = noise_pred
            return noise_pred

        if self._prev_noise is None:
            # 首步不滤波
            self._prev_noise = noise_pred
            return noise_pred

        # 检查形状匹配
        if self._prev_noise.shape != noise_pred.shape:
            # 形状不匹配 (可能 CFG 拼接), 跳过滤波
            self._prev_noise = noise_pred
            return noise_pred

        # 低通滤波: smoothed = beta * prev + (1-beta) * current
        smoothed = self.beta * self._prev_noise + (1.0 - self.beta) * noise_pred
        self._prev_noise = smoothed
        return smoothed

    def reset(self) -> None:
        """重置滤波器状态 (每个采样循环开始时调用)."""
        self._prev_noise = None


# ---------------------------------------------------------------------------
# 4. 综合修复器 (TemporalFlickerFix)
# ---------------------------------------------------------------------------
class TemporalFlickerFix:
    """综合时序闪烁修复器.

    集成三大修复策略:
      1. 帧间 EMA 平滑 (temporal_ema_smooth / temporal_ema_batch)
      2. 边界对齐 (boundary_align, V2V 续写)
      3. 采样步连贯滤波 (StepCoherenceFilter)

    用法:
        fix = TemporalFlickerFix(FlickerFixConfig(alpha=0.3, ...))

        # 在采样循环中
        fix.reset_step_filter()
        for step in range(num_steps):
            noise_pred = model(...)
            noise_pred = fix.filter_step(noise_pred)  # 步连贯
            latents = scheduler.step(noise_pred, ...)
            latents = fix.smooth_temporal(latents)  # 帧间平滑
            if is_v2v:
                latents = fix.align_boundary(latents, input_end_latent)
    """

    def __init__(self, config: FlickerFixConfig | None = None):
        self.config = config or FlickerFixConfig()
        self.config.validate()
        self.step_filter = StepCoherenceFilter(beta=self.config.step_coherence_beta)
        self._prev_latent: mx.array | None = None

    def filter_step(self, noise_pred: mx.array) -> mx.array:
        """采样步连贯滤波.

        Args:
            noise_pred: 当前步噪声预测

        Returns:
            滤波后的噪声预测
        """
        if not self.config.enable_step_coherence:
            return noise_pred
        return self.step_filter(noise_pred)

    def smooth_temporal(
        self,
        latent: mx.array,
        prev_latent: mx.array | None = None,
        frame_dim: int = 2,
    ) -> mx.array:
        """帧间 EMA 平滑.

        Args:
            latent: 当前 latent [B, C, T, H, W]
            prev_latent: 前一帧 latent (可选, 用于逐帧平滑)
            frame_dim: 时序维度索引

        Returns:
            平滑后的 latent
        """
        if not self.config.enable_temporal_ema:
            return latent

        alpha = self.config.temporal_ema_alpha

        if prev_latent is not None:
            # 逐帧平滑
            return temporal_ema_smooth(latent, prev_latent, alpha, frame_dim)
        else:
            # 整 batch 平滑
            return temporal_ema_batch(latent, alpha, frame_dim)

    def align_boundary(
        self,
        latent: mx.array,
        input_end_latent: mx.array | None,
        frame_dim: int = 2,
    ) -> mx.array:
        """边界对齐 (V2V 续写).

        Args:
            latent: 续写 latent [B, C, T, H, W]
            input_end_latent: 输入视频末帧 latent (None 表示不对齐)
            frame_dim: 时序维度索引

        Returns:
            对齐后的 latent
        """
        if not self.config.enable_boundary_align:
            return latent
        return boundary_align(
            latent,
            input_end_latent,
            alpha=self.config.boundary_align_alpha,
            frame_dim=frame_dim,
        )

    def reset(self) -> None:
        """重置所有状态 (每个采样循环开始时调用)."""
        self.step_filter.reset()
        self._prev_latent = None

    def reset_step_filter(self) -> None:
        """仅重置步连贯滤波器."""
        self.step_filter.reset()


# ---------------------------------------------------------------------------
# 分支默认配置
# ---------------------------------------------------------------------------
def default_config_for_branch(branch: str) -> FlickerFixConfig:
    """根据分支返回默认闪烁修复配置.

    Args:
        branch: r2v / v2v / a2v

    Returns:
        FlickerFixConfig
    """
    if branch == "r2v":
        # R2V: 参考图引导, 轻度平滑保细节
        return FlickerFixConfig(
            temporal_ema_alpha=0.2,
            boundary_align_alpha=0.0,  # R2V 不续写
            step_coherence_beta=0.1,
            enable_boundary_align=False,
        )
    elif branch == "v2v":
        # V2V: 续写关键, 强边界对齐 + 中度平滑
        return FlickerFixConfig(
            temporal_ema_alpha=0.3,
            boundary_align_alpha=0.5,
            step_coherence_beta=0.15,
            enable_boundary_align=True,
        )
    elif branch == "a2v":
        # A2V: 数字人, 嘴型连贯关键, 强平滑
        return FlickerFixConfig(
            temporal_ema_alpha=0.4,
            boundary_align_alpha=0.0,
            step_coherence_beta=0.2,
            enable_boundary_align=False,
        )
    else:
        raise ValueError(f"Unknown branch: {branch}. Valid: r2v/v2v/a2v")


__all__ = [
    "FlickerFixConfig",
    "TemporalFlickerFix",
    "StepCoherenceFilter",
    "temporal_ema_smooth",
    "temporal_ema_batch",
    "boundary_align",
    "default_config_for_branch",
]
