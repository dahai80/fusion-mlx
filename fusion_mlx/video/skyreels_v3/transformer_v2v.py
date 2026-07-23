# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 V2V 视频续写主干 (14B-720P) 纯 MLX 端口 (Tier-2 薄子类).

与 R2V 14B 的关键差异:
  1. 启用时序分支 (has_temporal=True), 扩大时序窗口保连贯
  2. window_size=(-1, -1) 全局空间注意力, 时序窗口默认 96 帧
  3. num_frame_list / grid_size_list / num_token_list 多段上下文

V2V 续写场景:
  - 输入 5s 视频片段 -> 续写到 30s
  - 前置帧 KV 可复用, 减少重复计算
  - 时序连贯性关键, 谨慎取巧 (xfuser 策略 v2v 分支阈值更严)

Block 层 (V2VAttentionBlock) 统一在 .blocks (issue #186 modulation fix);
DiT 公共主干统一在 .dit_base.SkyReelsBaseDiT. 本模块仅保留 V2V 变体差异:
  - block = V2VAttentionBlock(temporal_window=...)
  - lazy mx.compile 整 __call__ 融合 40-block 循环 (实测 3.3x, Metal 峰值持平 R2V)
  - forward_partial (issue #177 Phase-2 spec draft, CP3)
"""

from __future__ import annotations

import mlx.core as mx

from .blocks import V2VAttentionBlock
from .dit_base import SkyReelsBaseDiT


class SkyReelsV2VDiT(SkyReelsBaseDiT):
    """SkyReels-V3 V2V 14B-720P 视频续写主干.

    配置:
      - dim=5120, ffn_dim=13824, num_heads=40, num_layers=40
      - has_temporal=True, temporal_window=96 (保时序连贯)
      - context_window_size 控制前置帧 KV 复用窗口

    V2V 续写场景:
      - 输入 5s 视频片段 -> 续写到 30s
      - 前置帧 KV 可复用, 减少重复计算
    """

    def __init__(self, config: dict | None = None):
        super().__init__(
            config,
            num_layers_default=40,
            temporal_window_default=96,
            lazy_compile=True,
        )
        # context_window_size 仅为元数据 (block 调用不再透传, CP1 清理死 kwarg)
        self.context_window_size = (config or {}).get("context_window_size", 0)

    def _build_blocks(self) -> list:
        kw = self._block_kwargs()
        kw.update(temporal_window=self.temporal_window)
        return [V2VAttentionBlock(**kw) for _ in range(self.num_layers)]

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        context: mx.array,
        seq_lens: list,
        grid_sizes: list,
        context_lens: list | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
        controlnet_residuals: list[mx.array] | None = None,
    ) -> mx.array:
        """V2V 前向: 视频 latent + 时间步 + 文本 -> 去噪 latent."""
        return self._lazy_call(
            (
                x,
                t,
                context,
                seq_lens,
                grid_sizes,
                context_lens,
                rope_cos_sin,
                attn_mask,
                temporal_len,
                controlnet_residuals,
            )
        )

    def _call_raw(
        self,
        x: mx.array,
        t: mx.array,
        context: mx.array,
        seq_lens: list,
        grid_sizes: list,
        context_lens: list | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
        controlnet_residuals: list[mx.array] | None = None,
    ) -> mx.array:
        """原始前向路径 (mx.compile 融合目标)."""
        x = self.patch_embedding(x)
        context = self.text_embedding(context)
        e = self._time_embed(t)
        x = self._run_blocks(
            x,
            e,
            seq_lens,
            grid_sizes,
            context,
            context_lens,
            rope_cos_sin,
            attn_mask,
            temporal_len=temporal_len,
            controlnet_residuals=controlnet_residuals,
        )
        out = self.head(x, e)
        out = self._unpatchify(out, grid_sizes)
        return out

    def forward_partial(
        self,
        x: mx.array,
        t: mx.array,
        context: mx.array,
        seq_lens: list,
        grid_sizes: list,
        n_blocks: int | None = None,
        context_lens: list | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
        controlnet_residuals: list[mx.array] | None = None,
    ) -> mx.array:
        x = self.patch_embedding(x)
        context = self.text_embedding(context)
        e = self._time_embed(t)
        x = self._run_blocks(
            x,
            e,
            seq_lens,
            grid_sizes,
            context,
            context_lens,
            rope_cos_sin,
            attn_mask,
            n_blocks=n_blocks,
            temporal_len=temporal_len,
            controlnet_residuals=controlnet_residuals,
        )
        out = self.head(x, e)
        out = self._unpatchify(out, grid_sizes)
        return out


__all__ = [
    "V2VAttentionBlock",
    "SkyReelsV2VDiT",
]
