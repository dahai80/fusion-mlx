# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 A2V 数字人主干 (19B-720P) 纯 MLX 端口 (Tier-2 薄子类).

A2V 与 R2V/V2V 的关键差异:
  1. 新增音频 Embedding 分支 (wav2vec2 + xlm_roberta)
  2. dim=6144, ffn_dim=24576, num_heads=48, num_layers=60 (19B)
  3. 音频驱动口型同步, 启用时序分支保嘴型连贯
  4. cross_attn 同时接收文本 + 音频 context

A2V 数字人场景:
  - 输入: 音频 + 参考图 + 文本 Prompt
  - 输出: 数字人说话视频 (口型同步)

Block 层 (A2VAttentionBlock / AudioEmbedding) 统一在 .blocks (issue #186
modulation fix); DiT 公共主干统一在 .dit_base.SkyReelsBaseDiT. 本模块仅保留
A2V 变体差异:
  - _build_extra_embeddings 插入 audio_embedding (构造顺序 text->audio->time,
    严格保持, 否则 RNG draw 漂移破坏 bit-identical 前向)
  - block = A2VAttentionBlock(temporal_window=...)
  - __call__ 签名 (audio_embeds + text_embeds) + audio context 融合
  - lazy mx.compile 整 __call__ (实测整体编译劣化, 但 maybe_compile 每 block
    单独编译是最优路径, lazy 暂保留回退)
  - forward_partial (issue #177 Phase-2 spec draft, CP3)
"""

from __future__ import annotations

import mlx.core as mx

from .blocks import A2VAttentionBlock, AudioEmbedding
from .dit_base import SkyReelsBaseDiT


class SkyReelsA2VDiT(SkyReelsBaseDiT):
    """SkyReels-V3 A2V 19B-720P 数字人主干.

    配置:
      - dim=6144, ffn_dim=24576, num_heads=48, num_layers=60 (19B)
      - has_temporal=True, temporal_window=32 (保嘴型连贯)
      - 音频分支: AudioEmbedding (wav2vec2 + xlm_roberta 融合)

    A2V 数字人场景:
      - 输入: 音频 + 参考图 + 文本 Prompt
      - 输出: 数字人说话视频 (口型同步)
    """

    def __init__(self, config: dict | None = None):
        super().__init__(
            config,
            num_layers_default=60,
            temporal_window_default=32,
            lazy_compile=True,
        )

    def _build_extra_embeddings(self, cfg: dict) -> None:
        # A2V-19B 真实权重主干 dim=5120 (与 R2V/V2V 一致), audio_embedding 分支
        # audio_dim 独立 (wav2vec2 hidden 768), 不影响主干 dim.
        # 构造顺序: text_embedding -> audio_embedding -> time_embedding (严格).
        self.audio_dim = cfg.get("audio_dim", 768)
        self.audio_embedding = AudioEmbedding(
            dim=self.dim,
            audio_dim=self.audio_dim,
            text_dim=self.text_dim,
            eps=self.eps,
        )

    def _build_blocks(self) -> list:
        kw = self._block_kwargs()
        kw.update(temporal_window=self.temporal_window)
        return [A2VAttentionBlock(**kw) for _ in range(self.num_layers)]

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        audio_embeds: mx.array,
        text_embeds: mx.array,
        seq_lens: list,
        grid_sizes: list,
        context_lens: list | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
        cross_kv_cache: tuple | None = None,
    ) -> mx.array:
        """A2V 前向: 视频 latent + 时间步 + 音频 + 文本 -> 去噪 latent.

        Args:
            x: [B, C_in, T, H, W] 视频 latent
            t: [B] 时间步
            audio_embeds: [B, L_audio, audio_dim] 音频 embedding (wav2vec2)
            text_embeds: [B, L_text, text_dim] 文本 embedding (xlm_roberta)
            temporal_len: 时序长度 (帧数), A2V 必传 (保嘴型连贯)
            cross_kv_cache: AtomCode 跨步复用 cross-attn KV 缓存 (None=每步重算)

        Returns:
            [B, C_out, T, H, W] 去噪 latent
        """
        # AtomCode 专题优化实测 (2026-07-18):
        # - 未编译路径: 3132ms/step + 129GB Metal (11.7× 暴涨, 不可用)
        # - 60-block 整体 mx.compile: 36817ms/step + 129GB Metal (137× 暴涨, 不可用)
        # - 每 block maybe_compile (当前 __init__ 已做): 267ms/step + 24.8GB Metal (基线最优)
        # 结论: 整体 mx.compile 劣化, 未编译更糟, 当前 maybe_compile 每 block 单独编译是最优路径
        # lazy compile 首步触发后切 mx.compile 融合循环 (实测劣化但比未编译好, 暂保留)
        return self._lazy_call(
            (
                x,
                t,
                audio_embeds,
                text_embeds,
                seq_lens,
                grid_sizes,
                context_lens,
                rope_cos_sin,
                attn_mask,
                temporal_len,
                cross_kv_cache,
            )
        )

    def _call_raw(
        self,
        x: mx.array,
        t: mx.array,
        audio_embeds: mx.array,
        text_embeds: mx.array,
        seq_lens: list,
        grid_sizes: list,
        context_lens: list | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
        cross_kv_cache: tuple | None = None,
    ) -> mx.array:
        """原始前向路径 (mx.compile 融合目标)."""
        x = self.patch_embedding(x)
        # Audio + Text 融合 context (数字人专用)
        context = self.audio_embedding(audio_embeds, text_embeds)
        # 纯音频 context (audio_cross_attn 用): 取 audio_embeds 原始 768 维, 不经融合投影
        audio_ctx = audio_embeds  # [B, L_audio, 768]
        e = self._time_embed(t)
        # AtomCode: cross_kv_cache 跨步复用透传到每 block (每 block 内 cross_attn 复用同 KV)
        # issue #186: 统一 block __call__ 签名, context_lens 位置参数 7, audio_ctx 关键字末尾
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
            cross_kv_cache=cross_kv_cache,
            audio_ctx=audio_ctx,
        )
        out = self.head(x, e)
        out = self._unpatchify(out, grid_sizes)
        return out

    def forward_partial(
        self,
        x: mx.array,
        t: mx.array,
        audio_embeds: mx.array,
        text_embeds: mx.array,
        seq_lens: list,
        grid_sizes: list,
        n_blocks: int | None = None,
        context_lens: list | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
        temporal_len: int | None = None,
        cross_kv_cache: tuple | None = None,
    ) -> mx.array:
        # issue #177 Phase-2: layer-pruned draft 前向 (首 n_blocks + 共享 head/patch/time embed).
        # 与 __call__ 完全一致, 仅 self.blocks[:n_blocks]; n_blocks==num_layers 时 bit-identical.
        # CP3: A2V 扩展 forward_partial (不经 lazy compile, draft 路径直跑原始).
        x = self.patch_embedding(x)
        context = self.audio_embedding(audio_embeds, text_embeds)
        audio_ctx = audio_embeds
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
            cross_kv_cache=cross_kv_cache,
            audio_ctx=audio_ctx,
        )
        out = self.head(x, e)
        out = self._unpatchify(out, grid_sizes)
        return out


__all__ = [
    "AudioEmbedding",
    "A2VAttentionBlock",
    "SkyReelsA2VDiT",
]
