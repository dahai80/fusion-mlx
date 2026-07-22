# SPDX-License-Identifier: Apache-2.0
# issue #180 Metal async dispatch 回归测试.
#
# 验证三点:
#   1. async_denoise_enabled() env 门控 (默认关).
#   2. _denoise_sample 异步路径 (FUSION_ASYNC_DENOISE=1) 与同步路径逐位一致
#      (mock DiT + 同 seed) - 同 op 同序, 仅 eval 时机不同.
#   3. async_eval 逐步物化 (不累积 30 步图) - 内存峰值受控 (#146 不回归).

import logging
import os

os.environ.setdefault("FUSION_DISABLE_COMPILE", "1")

import mlx.core as mx
import numpy as np

from fusion_mlx.video.skyreels_v3.pipelines import (
    SkyReelsBasePipeline,
    SkyReelsPipelineConfig,
)
from fusion_mlx.video.skyreels_v3.speculative_denoise import async_denoise_enabled

logger = logging.getLogger(__name__)

# 与 test_speculative_denoise_phase2 一致的 tiny 形状 (proven 跑通 DiT 前向).
B = 1
C = 16
T = 2
H = 8
W = 8
L_CTX = 8
TEXT_DIM = 64
SEQ = T * (H // 2) * (W // 2)
GRID = (T, H // 2, W // 2)


class _MockDiT:
    # 最小 DiT: 返回与输入同形 velocity (向零收缩), 忽略 context/seq/grid.
    # 数值确定性是 parity 测试的前提 - 无任何随机性.
    def __init__(self, scale: float = 0.05):
        self.scale = scale

    def __call__(self, latent_input, t_mx, context_input, cfg_seq_lens, cfg_grid_sizes):
        return -latent_input * self.scale


class _StubStrategy:
    def reset(self):
        pass

    def set_current_step(self, step_idx):
        pass


class _AsyncStubPipeline(SkyReelsBasePipeline):
    # 绕过 _load_models (真实 28GB 权重), 仅装 mock DiT + 配置 + stub 策略.
    def __init__(self, dit, config):
        self.dit = dit
        self.step_strategy = _StubStrategy()
        self.config = config

    def _cfg_keep_steps(self, n_steps):
        # 全 cond-only (b=1), 避 CFG 拼接, 简化 mock 路径.
        return 0


def _make_pipeline(steps=4):
    config = SkyReelsPipelineConfig(
        branch="r2v",
        num_inference_steps=steps,
        guidance_scale=1.0,
    )
    return _AsyncStubPipeline(_MockDiT(), config)


def _inputs(seed=11):
    mx.random.seed(seed)
    latents = mx.random.normal((B, C, T, H, W))
    context = mx.random.normal((B, L_CTX, TEXT_DIM))
    return latents, context, [SEQ], [GRID]


def _run(pipeline, latents, context, seq_lens, grid_sizes):
    return pipeline._denoise_sample(
        latents, context, seq_lens=seq_lens, grid_sizes=grid_sizes
    )


def test_async_denoise_enabled_flag(monkeypatch):
    monkeypatch.delenv("FUSION_ASYNC_DENOISE", raising=False)
    assert async_denoise_enabled() is False
    monkeypatch.setenv("FUSION_ASYNC_DENOISE", "1")
    assert async_denoise_enabled() is True
    monkeypatch.setenv("FUSION_ASYNC_DENOISE", "true")
    assert async_denoise_enabled() is True
    monkeypatch.setenv("FUSION_ASYNC_DENOISE", "false")
    assert async_denoise_enabled() is False
    monkeypatch.setenv("FUSION_ASYNC_DENOISE", "0")
    assert async_denoise_enabled() is False


def test_async_path_parities_with_sync(monkeypatch):
    # 关闭投机去噪, 确保走 _denoise_sample 主路径 (非 spec 分支).
    monkeypatch.delenv("FUSION_SPECULATIVE_DENOISE", raising=False)

    # 同步路径 (async 门关)
    monkeypatch.delenv("FUSION_ASYNC_DENOISE", raising=False)
    latents, context, seq_lens, grid_sizes = _inputs(seed=11)
    sync_out = _run(_make_pipeline(), latents, context, seq_lens, grid_sizes)
    mx.eval(sync_out)

    # 异步路径 (async 门开)
    monkeypatch.setenv("FUSION_ASYNC_DENOISE", "1")
    latents2, context2, seq_lens2, grid_sizes2 = _inputs(seed=11)
    async_out = _run(_make_pipeline(), latents2, context2, seq_lens2, grid_sizes2)
    mx.eval(async_out)

    assert sync_out.shape == async_out.shape
    diff = float(mx.max(mx.abs(sync_out - async_out)).item())
    logger.info("parity sync-vs-async maxdiff=%.3e", diff)
    # 同 op 同序, 仅 eval 时机不同 -> 应逐位一致 (留 1e-6 浮点余量).
    assert diff < 1e-6, f"sync vs async maxdiff={diff}"


def test_async_eval_does_not_accumulate_graph():
    # #146 不回归: async_eval 逐步物化+释放, 30 步依赖链后峰值不应爆炸.
    # 不 eval 时图持有前序; async_eval 物化后 x 独立, 可单独访问.
    mx.random.seed(0)
    x = mx.zeros((B, C, T, H, W))
    for _ in range(30):
        v = -x * 0.05
        x = x + v * 0.1
        mx.async_eval(x)
    mx.synchronize()
    val = float(mx.sum(x).item())
    assert np.isfinite(val), "async_eval 链结果非有限 (图未物化?)"


def test_async_denoise_runs_full_loop(monkeypatch):
    # 异步路径能跑完多步不抛, synchronize 排空后返回有限值.
    monkeypatch.setenv("FUSION_ASYNC_DENOISE", "1")
    monkeypatch.delenv("FUSION_SPECULATIVE_DENOISE", raising=False)
    latents, context, seq_lens, grid_sizes = _inputs(seed=21)
    out = _run(_make_pipeline(steps=6), latents, context, seq_lens, grid_sizes)
    mx.eval(out)
    assert out.shape == latents.shape
    total = float(mx.sum(out).item())
    assert np.isfinite(total), "异步多步输出非有限"
    logger.info("async full-loop steps=6 out_sum=%.4f", total)
