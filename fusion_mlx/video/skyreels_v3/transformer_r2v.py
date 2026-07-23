# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 R2V 14B-720P DiT 主干 (纯 MLX 端口, Tier-2 薄子类).

Block 层 (WanFFN / Head / WanAttentionBlock) 统一在 .blocks; DiT 公共主干
(config 解析 / embed / freqs / head / _unpatchify / _run_blocks / _lazy_call)
统一在 .dit_base.SkyReelsBaseDiT. 本模块仅保留 R2V 变体差异:
  - block = WanAttentionBlock(has_temporal=False, text_dim=...)
  - 无 lazy mx.compile (R2V 逐 block maybe_compile 已足够, 不改编译行为)
  - forward_partial (issue #177 Phase-2 spec draft)

原版权重映射严格对齐, 参数不可修改.
"""

from __future__ import annotations

import mlx.core as mx

from .blocks import Head, WanAttentionBlock, WanFFN
from .dit_base import SkyReelsBaseDiT


class SkyReelsR2VDiT(SkyReelsBaseDiT):
    """SkyReels-V3 R2V 14B-720P 完整 DiT 主干.

    配置:
      - dim=5120, ffn_dim=13824, num_heads=40, num_layers=40
      - patch_size=(1, 2, 2), in_dim=16, out_dim=16
      - text_dim=4096, text_len=512
      - cross_attn_type="i2v_cross_attn" (参考图引导)
      - window_size=(-1, -1) 全局注意力

    参数量: ~14B
    """

    def __init__(self, config: dict | None = None):
        super().__init__(
            config,
            num_layers_default=40,
            temporal_window_default=96,
            lazy_compile=False,
        )

    def _build_blocks(self) -> list:
        kw = self._block_kwargs()
        kw.update(has_temporal=False, text_dim=self.text_dim)
        return [WanAttentionBlock(**kw) for _ in range(self.num_layers)]

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
        controlnet_residuals: list[mx.array] | None = None,
    ) -> mx.array:
        """前向: 视频 latent + 时间步 + 文本 -> 去噪 latent.

        Args:
            x: [B, C_in, T, H, W] 视频 latent
            t: [B] 时间步
            context: [B, L_ctx, text_dim] 文本/参考图 embedding
            seq_lens: 每个样本的序列长度
            grid_sizes: 每个样本的 (F, H, W) 网格
            context_lens: context 长度
            rope_cos_sin: 预计算 rope
            attn_mask: 注意力掩码
            controlnet_residuals: per-block residuals from ControlNet (optional)

        Returns:
            [B, C_out, T, H, W] 去噪 latent
        """
        x = self.patch_embedding(x)  # [B, L, dim]
        context = self.text_embedding(context)  # [B, L_ctx, dim]
        e = self._time_embed(t)  # [B, dim*6]
        x = self._run_blocks(
            x,
            e,
            seq_lens,
            grid_sizes,
            context,
            context_lens,
            rope_cos_sin,
            attn_mask,
            controlnet_residuals=controlnet_residuals,
        )
        out = self.head(x, e)  # [B, L, prod(patch_size)*out_dim]
        out = self._unpatchify(out, grid_sizes)  # [B, C_out, T, H, W]
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
            controlnet_residuals=controlnet_residuals,
        )
        out = self.head(x, e)
        out = self._unpatchify(out, grid_sizes)
        return out


__all__ = [
    "WanAttentionBlock",
    "WanFFN",
    "Head",
    "SkyReelsR2VDiT",
]
