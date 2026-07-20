# SPDX-License-Identifier: Apache-2.0
"""xfuser 步级注意力策略接入 SkyReels-V3 采样循环.

复用底座 fusion_mlx.custom_kernels.xfuser_attention:
  - FastAttnMethod: FULL / RESIDUAL_WINDOW / SPARSE / CFG_SHARE / OUTPUT_SHARE
  - MLXFastAttention: 步级注意力分派
  - calibrate_attention_strategy: 自动标定每模块每步策略

本模块负责:
  1. 注入 MLXFastAttention 到 SkyReels DiT 各 WanAttentionBlock
  2. 接管采样循环的 step_idx 流转
  3. 配置 R2V/V2V/A2V 三大分支的默认步级策略
  4. M5 Max 上启用更激进的 SPARSE + CFG_SHARE 组合
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    from fusion_mlx.custom_kernels.xfuser_attention import (
        FastAttnMethod,
        MLXFastAttention,
        calibrate_attention_strategy,
    )

    _HAS_XFUSER = True
except Exception:  # pragma: no cover - xfuser optional
    MLXFastAttention = None
    FastAttnMethod = None
    calibrate_attention_strategy = None
    _HAS_XFUSER = False

from . import _device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 步级策略配置
# ---------------------------------------------------------------------------
@dataclass
class StepStrategyConfig:
    """采样步级策略配置.

    Attributes:
        total_steps: 采样总步数 (e.g. 50)
        warmup_steps: 前若干步强制 FULL_ATTN 保画质
        tail_steps: 后若干步允许激进取巧
        use_cfg_share: CFG 共享 (无/条件分支复用 K/V), 减半注意力
        use_output_share: 相邻步输出复用, 跳过部分步计算
        use_residual_window: 残差窗口注意力 (FULL - WINDOW 残差)
        use_sparse: 稀疏注意力 (分块掩码), 最激进
    """

    total_steps: int = 50
    warmup_steps: int = 5
    tail_steps: int = 3
    use_cfg_share: bool = True
    use_output_share: bool = True
    use_residual_window: bool = True
    use_sparse: bool = False  # 默认关闭, M5 才启用
    window_size: int = 64
    threshold: float = 0.10  # calibrate_attention_strategy 阈值

    def build_step_methods(self) -> list[Any]:
        """构造每步的 FastAttnMethod 列表."""
        if not _HAS_XFUSER or FastAttnMethod is None:
            return []

        methods: list[Any] = []
        for step in range(self.total_steps):
            if step < self.warmup_steps:
                # 前期: FULL + CFG_SHARE (保画质)
                m = FastAttnMethod.FULL_ATTN
                if self.use_cfg_share:
                    m = m | FastAttnMethod.CFG_SHARE
            elif step < self.total_steps - self.tail_steps:
                # 中期: 残差窗口 + 稀疏渐进
                if self.use_sparse:
                    m = FastAttnMethod.SPARSE_ATTN
                elif self.use_residual_window:
                    m = FastAttnMethod.RESIDUAL_WINDOW_ATTN
                else:
                    m = FastAttnMethod.FULL_ATTN
                if self.use_cfg_share:
                    m = m | FastAttnMethod.CFG_SHARE
            else:
                # 后期: 输出复用 (最快)
                m = FastAttnMethod.FULL_ATTN
                if self.use_output_share:
                    m = m | FastAttnMethod.OUTPUT_SHARE
                if self.use_cfg_share:
                    m = m | FastAttnMethod.CFG_SHARE
            methods.append(m)
        return methods


# ---------------------------------------------------------------------------
# 三大分支默认配置
# ---------------------------------------------------------------------------
def _default_config_for_branch(
    branch: str, total_steps: int = 50
) -> StepStrategyConfig:
    """根据分支 (r2v/v2v/a2v) 返回默认 StepStrategyConfig.

    M5 Max 上启用更激进的 SPARSE + CFG_SHARE 组合.
    """
    is_m5 = _device.is_m5()

    if branch == "r2v":
        cfg = StepStrategyConfig(
            total_steps=total_steps,
            warmup_steps=5,
            tail_steps=3,
            use_cfg_share=True,
            use_output_share=is_m5,  # M5 才启用 OUTPUT_SHARE
            use_residual_window=True,
            use_sparse=is_m5,  # M5 启用 SPARSE
            window_size=64,
            threshold=0.08,
        )
    elif branch == "v2v":
        # V2V 视频续写: 时序连贯性关键, 谨慎取巧
        cfg = StepStrategyConfig(
            total_steps=total_steps,
            warmup_steps=8,  # 更长 warmup 保时序连贯
            tail_steps=2,
            use_cfg_share=True,
            use_output_share=False,  # 续写不用输出复用
            use_residual_window=True,
            use_sparse=False,  # 续写不用稀疏
            window_size=96,  # 更大窗口保连贯
            threshold=0.05,  # 更严阈值
        )
    elif branch == "a2v":
        # A2V 数字人: 音频驱动, 中等激进度
        # AtomCode 专题优化: total_steps 50→30 (2026-07-18)
        # DiT 74% 主瓶颈 × 30步 vs 50步 = 降 40% DiT 耗时, UniPC corrector 保稳 (solver_order=2 历史预测仍可用)
        cfg = StepStrategyConfig(
            total_steps=total_steps,
            warmup_steps=6,
            tail_steps=3,
            use_cfg_share=True,
            use_output_share=is_m5,
            use_residual_window=True,
            use_sparse=is_m5,
            window_size=64,
            threshold=0.07,
        )
    else:
        raise ValueError(f"Unknown branch: {branch}. Valid: r2v/v2v/a2v")

    return cfg


# ---------------------------------------------------------------------------
# 步级策略管理器
# ---------------------------------------------------------------------------
class SkyReelsStepStrategy:
    """SkyReels 采样循环的 xfuser 步级策略管理器.

    用法:
        strategy = SkyReelsStepStrategy("r2v", total_steps=50)
        strategy.attach_to_model(dit_model)
        for step in range(50):
            strategy.set_current_step(step)
            dit_model(latent, step=step)
    """

    def __init__(
        self,
        branch: str = "r2v",
        total_steps: int = 50,
        config: StepStrategyConfig | None = None,
    ):
        self.branch = branch
        self.config = config or _default_config_for_branch(branch, total_steps)
        self.config.total_steps = total_steps
        self.total_steps = total_steps

        self._current_step = 0
        self._fast_attn_modules: list[Any] = []
        self._attached = False

    def attach_to_model(self, model: Any) -> None:
        """注入 MLXFastAttention 到 DiT 各 attention 模块.

        Args:
            model: SkyReelsDiT 实例, 需暴露 blocks 列表
        """
        if not _HAS_XFUSER:
            logger.warning("xfuser not available, skip step strategy attach")
            return

        self._fast_attn_modules = []
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            logger.warning("model has no 'blocks' attr, cannot attach strategy")
            return

        step_methods = self.config.build_step_methods()

        for block in blocks:
            # 为每个 attention 子模块创建独立 MLXFastAttention
            for attn_attr in ("self_attn", "cross_attn", "temporal_attn"):
                attn = getattr(block, attn_attr, None)
                if attn is None:
                    continue
                fa = MLXFastAttention(
                    window_size=self.config.window_size,
                    cond_first=False,
                )
                fa.set_methods(step_methods, selecting=False)
                attn._fast_attn = fa
                self._fast_attn_modules.append(fa)

        self._attached = True
        logger.info(
            "xfuser step strategy attached: branch=%s steps=%d modules=%d",
            self.branch,
            self.total_steps,
            len(self._fast_attn_modules),
        )

    def set_current_step(self, step: int) -> None:
        """设置当前采样步号 (由 xfuser attention 内部读取).

        Args:
            step: 当前采样步 (0-indexed)
        """
        self._current_step = step
        # 同步到全局 xfuser 状态 (current_step / is_active)
        try:
            from fusion_mlx.custom_kernels.xfuser_attention import (
                set_active as _set_active,
            )
            from fusion_mlx.custom_kernels.xfuser_attention import (
                set_current_step as _set_step,
            )

            _set_step(step)
            _set_active(True)
        except Exception:  # pragma: no cover - xfuser optional
            pass

    def reset(self) -> None:
        """每个采样循环开始时重置步计数和缓存."""
        self._current_step = 0
        for fa in self._fast_attn_modules:
            fa.cached_output = None
            fa.cached_residual = None

    def calibrate(
        self,
        model: Any,
        calib_prompts: list[str],
    ) -> list[list[Any]]:
        """自动标定每模块每步的最佳策略.

        用 calibrate_attention_strategy 贪心搜索最激进且 loss < threshold 的方法.

        Args:
            model: SkyReelsDiT 实例 (需支持 calibration_forward 或 __call__)
            calib_prompts: 标定用 prompt 列表

        Returns:
            strategies: [num_modules][num_steps] 每模块每步的 FastAttnMethod
        """
        if not _HAS_XFUSER or calibrate_attention_strategy is None:
            logger.warning("xfuser calibration not available")
            return []

        if not self._fast_attn_modules:
            self.attach_to_model(model)

        strategies = calibrate_attention_strategy(
            model=model,
            attention_modules=self._fast_attn_modules,
            n_steps=self.total_steps,
            calib_prompts=calib_prompts,
            threshold=self.config.threshold,
            verbose=True,
        )

        logger.info(
            "calibrated %d modules x %d steps: %s",
            len(strategies),
            self.total_steps,
            {s[0].name if s else "NONE" for s in strategies},
        )
        return strategies


__all__ = [
    "StepStrategyConfig",
    "SkyReelsStepStrategy",
]
