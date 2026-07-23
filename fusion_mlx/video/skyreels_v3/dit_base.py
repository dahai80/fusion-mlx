# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 DiT 共享主干 (R2V / V2V / A2V 三变体公共层).

Tier-2 统一: 抽取三变体完全一致的 DiT 主干结构到 SkyReelsBaseDiT:
  - config 解析 (dim / ffn / heads / patch / text / time / freqs / cross_attn_type)
  - patch / text / time embedding + time_projection + RoPE freqs + Head
  - _time_embed / _run_blocks / _unpatchify / _lazy_call 公共前向工具

变体子类只覆写两个 hook + __call__ / _call_raw:
  - _build_extra_embeddings(cfg): A2V 在此插入 audio_embedding (构造顺序严格
    保持 text_embedding -> audio_embedding -> time_embedding, 否则 RNG draw
    顺序漂移致 block/head 权重变化, 破坏 bit-identical 前向, issue #186)
  - _build_blocks(): 返回变体 block 列表 (WanAttentionBlock / V2VAttentionBlock
    / A2VAttentionBlock + 变体专属 kwargs)

原版权重映射严格对齐, 参数不可修改 (load_weights 按 key 名映射, 与类身份无关).
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from . import _device
from .blocks import Head
from .common import (
    GELUApprox,
    PatchEmbed3D,
    maybe_compile,
    rope_params,
    sinusoidal_embedding_1d,
)

logger = logging.getLogger(__name__)


class SkyReelsBaseDiT(nn.Module):
    """SkyReels-V3 DiT 三变体 (R2V/V2V/A2V) 共享主干.

    子类契约:
      _build_extra_embeddings(cfg): 默认 no-op; A2V 覆写插入 audio_embedding.
      _build_blocks() -> list[nn.Module]: 必须覆写, 返回变体 block 列表.
      _block_kwargs() -> dict: 9 个公共 block 构造参数 (变体再 append 专属).

    lazy_compile=True 时 (V2V/A2V) __call__ 走 _lazy_call 首步 lazy mx.compile
    融合循环; =False 时 (R2V) __call__ 由子类直跑 _call_raw 等价路径.
    """

    def __init__(
        self,
        config: dict | None = None,
        *,
        num_layers_default: int = 40,
        temporal_window_default: int = 96,
        lazy_compile: bool = False,
    ):
        super().__init__()
        cfg = config or {}
        self.dim = cfg.get("dim", 5120)
        self.ffn_dim = cfg.get("ffn_dim", 13824)
        self.num_heads = cfg.get("num_heads", 40)
        self.num_kv_heads = cfg.get("num_kv_heads", self.num_heads)
        self.num_layers = cfg.get("num_layers", num_layers_default)
        self.patch_size = cfg.get("patch_size", (1, 2, 2))
        self.in_dim = cfg.get("in_dim", 16)
        self.out_dim = cfg.get("out_dim", 16)
        self.text_dim = cfg.get("text_dim", 4096)
        self.text_len = cfg.get("text_len", 512)
        self.freq_dim = cfg.get("freq_dim", 256)
        self.window_size = cfg.get("window_size", (-1, -1))
        self.qk_norm = cfg.get("qk_norm", True)
        self.cross_attn_norm = cfg.get("cross_attn_norm", True)
        self.eps = cfg.get("eps", 1e-6)
        # issue #164: added_kv_proj_dim=null (纯 T2V) -> t2v_cross_attn; 非 None
        # (i2v 风格, 需 k_img/v_img) -> i2v_cross_attn; 显式 cross_attn_type 优先.
        added_kv = cfg.get("added_kv_proj_dim", None)
        if "cross_attn_type" in cfg:
            self.cross_attn_type = cfg["cross_attn_type"]
        elif added_kv is not None:
            self.cross_attn_type = "i2v_cross_attn"
        else:
            self.cross_attn_type = "t2v_cross_attn"
        self.temporal_window = cfg.get("temporal_window", temporal_window_default)

        # Patch embedding (Conv3d)
        self.patch_embedding = PatchEmbed3D(self.in_dim, self.dim, self.patch_size)

        # Text embedding
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.dim),
            GELUApprox(),
            nn.Linear(self.dim, self.dim),
        )

        # hook: A2V 在此插入 audio_embedding (顺序: text -> audio -> time)
        self._build_extra_embeddings(cfg)

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.dim * 6),
        )

        # RoPE 频率表 (buffer)
        head_dim = self.dim // self.num_heads
        self.freqs = rope_params(2048, head_dim)

        # DiT Blocks (变体 hook)
        self.blocks = self._build_blocks()

        # Output head
        self.head = Head(self.dim, self.out_dim, self.patch_size, self.eps)

        # 编译优化 (M5 Max 默认开启): 逐 block maybe_compile
        if _device.should_compile():
            self.blocks = [maybe_compile(b) for b in self.blocks]

        # lazy mx.compile 整 __call__ 融合循环状态 (V2V/A2V 启用, R2V 不用)
        self._lazy_compile = lazy_compile
        self._compiled_call = None

    # ---- variant hooks ----
    def _build_extra_embeddings(self, cfg: dict) -> None:
        """默认 no-op; A2V 覆写插入 audio_embedding (保持构造顺序)."""

    def _build_blocks(self) -> list:
        """必须覆写: 返回变体 block 列表."""
        raise NotImplementedError

    def _block_kwargs(self) -> dict:
        """9 个公共 block 构造参数 (变体 _build_blocks 再 append 专属)."""
        return dict(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            qk_norm=self.qk_norm,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps,
            cross_attn_type=self.cross_attn_type,
            num_kv_heads=self.num_kv_heads,
        )

    # ---- shared forward utils ----
    def _time_embed(self, t: mx.array) -> mx.array:
        t_emb = sinusoidal_embedding_1d(self.freq_dim, t)
        t_emb = self.time_embedding(t_emb)
        return self.time_projection(t_emb)

    def _run_blocks(
        self,
        x: mx.array,
        e: mx.array,
        seq_lens: list,
        grid_sizes: list,
        context: mx.array,
        context_lens: list | None,
        rope_cos_sin: tuple | None,
        attn_mask: mx.array | None,
        n_blocks: int | None = None,
        controlnet_residuals: list[mx.array] | None = None,
        **block_extra,
    ) -> mx.array:
        blocks = self.blocks if n_blocks is None else self.blocks[:n_blocks]
        for idx, block in enumerate(blocks):
            x = block(
                x,
                e,
                seq_lens,
                grid_sizes,
                self.freqs,
                context,
                context_lens,
                rope_cos_sin=rope_cos_sin,
                attn_mask=attn_mask,
                **block_extra,
            )
            if controlnet_residuals is not None and idx < len(controlnet_residuals):
                x = x + controlnet_residuals[idx]
        return x

    def _unpatchify(self, x: mx.array, grid_sizes: list) -> mx.array:
        b = x.shape[0]
        pt, ph, pw = self.patch_size
        outputs = []
        for i, (f, h, w) in enumerate(grid_sizes):
            # grid_sizes (f, h, w) 为 patch 后 token 网格 (对齐 wan2 unpatchify)
            seq_len = f * h * w
            x_i = x[i, :seq_len]
            x_i = x_i.reshape(f, h, w, pt, ph, pw, self.out_dim)
            x_i = x_i.transpose(6, 0, 3, 1, 4, 2, 5)
            x_i = x_i.reshape(self.out_dim, f * pt, h * ph, w * pw)
            outputs.append(x_i)
        return mx.stack(outputs, axis=0)

    def _lazy_call(self, args: tuple) -> mx.array:
        """lazy mx.compile(_call_raw) 融合循环; 首步用原始路径触发后切编译."""
        if not self._lazy_compile:
            return self._call_raw(*args)
        if self._compiled_call is None:
            out = self._call_raw(*args)
            try:
                self._compiled_call = mx.compile(self._call_raw)
            except Exception:
                logger.warning(
                    "mx.compile(_call_raw) 失败, 回退未编译路径", exc_info=True
                )
                self._compiled_call = False
            return out
        if self._compiled_call is False:
            return self._call_raw(*args)
        return self._compiled_call(*args)


__all__ = ["SkyReelsBaseDiT"]
