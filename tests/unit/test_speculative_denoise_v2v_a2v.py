# SPDX-License-Identifier: Apache-2.0
"""issue #186 item 3: V2V/A2V 投机去噪管线接入回归测试.

验证 FUSION_SPECULATIVE_DENOISE 开启时:
  1. V2V base _denoise_sample 路由到 spec 路径 (单 context 约定, 同 R2V)
  2. A2V 自有 _denoise_sample_speculative override (audio + text 约定) 可跑通
  3. A2V override draft==full 时 spec 不变量 == baseline Euler (正确性)
  4. A2V base _denoise_sample 被 branch guard 跳过 (不走 R2V 约定 spec 路径)

注: V2V/A2V 真实权重未本地下载, 用 TINY 随机初始化 DiT 验证管线接入逻辑.
"""

import os

os.environ.setdefault("FUSION_DISABLE_COMPILE", "1")

import mlx.core as mx
import pytest

from fusion_mlx.video.skyreels_v3.pipelines import (
    SkyReelsA2VPipeline,
    SkyReelsPipelineConfig,
    SkyReelsV2VPipeline,
)
from fusion_mlx.video.skyreels_v3.scheduler.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler,
    perform_guidance,
)
from fusion_mlx.video.skyreels_v3.speculative_denoise import baseline_euler
from fusion_mlx.video.skyreels_v3.transformer_a2v import SkyReelsA2VDiT
from fusion_mlx.video.skyreels_v3.transformer_v2v import SkyReelsV2VDiT

TINY = dict(
    dim=80,
    ffn_dim=160,
    num_heads=4,
    num_kv_heads=4,
    num_layers=4,
    patch_size=(1, 2, 2),
    in_dim=16,
    out_dim=16,
    text_dim=80,
    text_len=32,
    freq_dim=32,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
)

F, H, W = 2, 4, 4
SEQ = F * H * W
LATENT_SHAPE = (1, 16, F, H * 2, W * 2)
GRID = [(F, H, W)]


def _build_v2v_dit():
    mx.random.seed(7)
    return SkyReelsV2VDiT(
        dict(TINY, cross_attn_type="t2v_cross_attn", temporal_window=96)
    )


def _build_a2v_dit():
    mx.random.seed(7)
    return SkyReelsA2VDiT(
        dict(
            TINY,
            cross_attn_type="t2v_cross_attn",
            temporal_window=32,
            audio_dim=768,
        )
    )


def _make_pipe(cls, branch, dit, temporal_window):
    # 绕过 __init__ (_load_models 会建全量 60 层 DiT, 单测过重), 手动注入 TINY DiT.
    pipe = cls.__new__(cls)
    pipe.model_path = None
    pipe.config = SkyReelsPipelineConfig(
        branch=branch,
        temporal_window=temporal_window,
        num_inference_steps=4,
        guidance_scale=5.0,
        text_dim=80,
        audio_dim=768,
    )
    pipe.dit = dit
    pipe.vae = None
    pipe.text_encoder = None
    pipe.clip_encoder = None
    pipe.m5_optimizer = None
    pipe.step_strategy = None
    pipe._last_spec_stats = None
    return pipe


def _allclose(a, b, atol=1e-4, rtol=0.0):
    # np.allclose 语义: |a-b| <= atol + rtol*|b|. rtol 容忍 float32 批量 vs 单步
    # reduction 噪声 (spec 批量 K 步单次前向, baseline 逐步前向, GPU 归约顺序不同).
    max_abs = float(mx.max(mx.abs(a - b)).item())
    max_mag = float(mx.max(mx.abs(b)).item())
    return max_abs <= atol + rtol * max_mag


def test_v2v_base_denoise_routes_to_spec(monkeypatch):
    # item 3: V2V 用 base spec 路径 (单 context 约定, 同 R2V). 开启 spec 时 base
    # _denoise_sample 必须路由到 _denoise_sample_speculative.
    dit = _build_v2v_dit()
    pipe = _make_pipe(SkyReelsV2VPipeline, "v2v", dit, temporal_window=96)
    monkeypatch.setenv("FUSION_SPECULATIVE_DENOISE", "1")
    called = {}

    def spy(latents, context, *, seq_lens, grid_sizes):
        called["yes"] = True
        return latents

    monkeypatch.setattr(pipe, "_denoise_sample_speculative", spy)
    x = mx.random.normal(LATENT_SHAPE)
    ctx = mx.random.normal((1, 32, 80))
    out = pipe._denoise_sample(x, ctx, seq_lens=[SEQ], grid_sizes=GRID)
    assert called.get("yes")
    mx.eval(out)
    assert tuple(out.shape) == LATENT_SHAPE


def test_a2v_spec_override_runs(monkeypatch):
    # item 3: A2V 自有 _denoise_sample_speculative override (audio + text 约定) 可跑通.
    dit = _build_a2v_dit()
    pipe = _make_pipe(SkyReelsA2VPipeline, "a2v", dit, temporal_window=32)
    monkeypatch.setenv("FUSION_SPEC_DRAFT_BLOCKS", str(dit.num_layers))
    latents = mx.random.normal(LATENT_SHAPE)
    audio_embeds = mx.random.normal((1, 10, 768))
    text_embeds = mx.random.normal((1, 32, 80))
    out = pipe._denoise_sample_speculative(
        latents, audio_embeds, text_embeds, seq_lens=[SEQ], grid_sizes=GRID
    )
    mx.eval(out)
    assert tuple(out.shape) == LATENT_SHAPE
    stats = pipe._last_spec_stats
    assert stats is not None
    assert stats.macro_steps >= 1
    assert sum(stats.accepted) >= 1  # draft==full -> 全接受


def test_a2v_spec_override_draft_equals_full_invariant(monkeypatch):
    # item 3 正确性: draft==full (n_blocks=num_layers) 时 spec 输出 == baseline Euler.
    dit = _build_a2v_dit()
    num_layers = dit.num_layers
    pipe = _make_pipe(SkyReelsA2VPipeline, "a2v", dit, temporal_window=32)
    monkeypatch.setenv("FUSION_SPEC_DRAFT_BLOCKS", str(num_layers))
    latents = mx.random.normal(LATENT_SHAPE)
    audio_embeds = mx.random.normal((1, 10, 768))
    text_embeds = mx.random.normal((1, 32, 80))
    seq_lens = [SEQ]
    grid_sizes = GRID
    out = pipe._denoise_sample_speculative(
        latents,
        audio_embeds,
        text_embeds,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
    )

    # baseline Euler with same full_velocity (A2V audio + text 约定)
    scheduler = FlowUniPCMultistepScheduler(
        num_inference_steps=pipe.config.num_inference_steps
    )
    scheduler.set_timesteps(pipe.config.num_inference_steps)
    timesteps = scheduler.timesteps
    guidance = pipe.config.guidance_scale

    def full_velocity(x_batch, t_batch):
        k = x_batch.shape[0]
        x_2k = mx.concatenate([x_batch, x_batch], axis=0)
        t_2k = mx.concatenate([t_batch, t_batch], axis=0)
        audio_2k = mx.concatenate([audio_embeds] * (2 * k), axis=0)
        text_2k = mx.concatenate([text_embeds] * (2 * k), axis=0)
        seq_2k = list(seq_lens) * (2 * k)
        grid_2k = list(grid_sizes) * (2 * k)
        noise = dit(x_2k, t_2k, audio_2k, text_2k, seq_2k, grid_2k)
        return perform_guidance(noise, guidance)

    base_out = baseline_euler(full_velocity, latents[0], timesteps)
    mx.eval(out)
    mx.eval(base_out)
    # rtol=1e-3: 实测 float32 批量 vs 逐步噪声 ~2.8e-4 (magnitude ~1225).
    # 真实 bug 会 O(1) 发散, 远超此阈值.
    assert _allclose(out[0], base_out, atol=1e-4, rtol=1e-3)


def test_a2v_base_denoise_skips_spec_guard(monkeypatch):
    # item 3: base spec 路径 branch guard 跳过 a2v (A2V 走自有 override, 签名不同).
    dit = _build_a2v_dit()
    pipe = _make_pipe(SkyReelsA2VPipeline, "a2v", dit, temporal_window=32)
    monkeypatch.setenv("FUSION_SPECULATIVE_DENOISE", "1")

    def boom(*args, **kwargs):
        raise AssertionError("base spec path must NOT run for a2v branch")

    monkeypatch.setattr(pipe, "_denoise_sample_speculative", boom)
    x = mx.random.normal(LATENT_SHAPE)
    ctx = mx.random.normal((1, 32, 80))
    # guard 跳过 spec -> 落 UniPC -> step_strategy None -> RuntimeError (非 boom)
    with pytest.raises(RuntimeError):
        pipe._denoise_sample(x, ctx, seq_lens=[SEQ], grid_sizes=GRID)
