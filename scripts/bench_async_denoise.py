#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# issue #180 P2: 异步去噪基准 - 同步 vs 异步 wall-clock + GPU-idle 恢复评估.
#
# 用真实 (tiny) SkyReelsR2VDiT 跑 N 步去噪, 对比:
#   - sync:  每步 mx.eval (CPU 建图阻塞 GPU, GPU 空闲等建图)
#   - async: 每步 mx.async_eval + 末尾 mx.synchronize (CPU 建图与 GPU 计算重叠)
# 报告 wall-clock 加速比 + 峰值显存.
#
# 注意: tiny DiT 下 CPU 建图占比偏高, 加速比会高估; 真实 14B (P3) GPU 占主导,
# 加速比收敛到 ~CPU 建图占比 (预期 1-5%, 见 #177 先例).

import argparse
import logging
import os
import time

os.environ.setdefault("FUSION_DISABLE_COMPILE", "1")

import mlx.core as mx

from fusion_mlx.video.skyreels_v3.pipelines import (
    SkyReelsBasePipeline,
    SkyReelsPipelineConfig,
)
from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 与 test_speculative_denoise_phase2 一致的 tiny 配置 (proven 跑通).
TINY_CFG = {
    "dim": 64,
    "ffn_dim": 128,
    "num_heads": 4,
    "num_kv_heads": 4,
    "num_layers": 4,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "out_dim": 16,
    "text_dim": 64,
    "text_len": 32,
    "freq_dim": 32,
    "window_size": (-1, -1),
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}

B = 1
C = 16
T = 2
H = 8
W = 8
L_CTX = 8
SEQ = T * (H // 2) * (W // 2)
GRID = (T, H // 2, W // 2)


class _StubStrategy:
    def reset(self):
        pass

    def set_current_step(self, step_idx):
        pass


class _BenchPipeline(SkyReelsBasePipeline):
    # 绕过 _load_models (真实 28GB 权重), 装真实 tiny DiT + 配置.
    def __init__(self, dit, config):
        self.dit = dit
        self.step_strategy = _StubStrategy()
        self.config = config

    def _cfg_keep_steps(self, n_steps):
        # 全 cond-only (b=1), 避 CFG 拼接, 对齐 tiny 测试路径.
        return 0


def _build(steps, layers, dim, ffn_dim):
    mx.random.seed(7)
    cfg = dict(TINY_CFG)
    cfg["num_layers"] = layers
    cfg["dim"] = dim
    cfg["ffn_dim"] = ffn_dim
    dit = SkyReelsR2VDiT(cfg)
    config = SkyReelsPipelineConfig(
        branch="r2v",
        num_inference_steps=steps,
        guidance_scale=1.0,
    )
    return _BenchPipeline(dit, config)


def _inputs(seed=11):
    mx.random.seed(seed)
    latents = mx.random.normal((B, C, T, H, W))
    context = mx.random.normal((B, L_CTX, 64))
    return latents, context, [SEQ], [GRID]


def _time_run(pipeline, latents, context, seq_lens, grid_sizes):
    mx.synchronize()
    mem_before = mx.metal.get_active_memory()
    t0 = time.perf_counter()
    out = pipeline._denoise_sample(
        latents, context, seq_lens=seq_lens, grid_sizes=grid_sizes
    )
    mx.eval(out)
    mx.synchronize()
    t1 = time.perf_counter()
    mem_after = mx.metal.get_active_memory()
    return t1 - t0, max(mem_before, mem_after)


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def main():
    parser = argparse.ArgumentParser(description="#180 async denoise bench")
    parser.add_argument("--steps", type=int, default=8, help="denoise steps")
    parser.add_argument("--runs", type=int, default=3, help="timed runs per mode")
    parser.add_argument("--layers", type=int, default=4, help="DiT layers (size knob)")
    parser.add_argument(
        "--dim", type=int, default=64, help="DiT hidden dim (size knob)"
    )
    parser.add_argument(
        "--ffn-dim", type=int, default=128, help="DiT FFN dim (size knob)"
    )
    args = parser.parse_args()

    pipeline = _build(args.steps, args.layers, args.dim, args.ffn_dim)

    # warmup (编译/首次图构建), 不计时.
    for _ in range(2):
        _time_run(pipeline, *_inputs())

    # sync 模式 (async 门关, spec 门关)
    os.environ.pop("FUSION_ASYNC_DENOISE", None)
    os.environ.pop("FUSION_SPECULATIVE_DENOISE", None)
    sync_times, sync_mem = [], 0
    for _ in range(args.runs):
        dt, mem = _time_run(pipeline, *_inputs())
        sync_times.append(dt)
        sync_mem = max(sync_mem, mem)
    logger.info("sync runs=%s", [f"{t*1000:.1f}ms" for t in sync_times])

    # async 模式 (async 门开)
    os.environ["FUSION_ASYNC_DENOISE"] = "1"
    async_times, async_mem = [], 0
    for _ in range(args.runs):
        dt, mem = _time_run(pipeline, *_inputs())
        async_times.append(dt)
        async_mem = max(async_mem, mem)
    logger.info("async runs=%s", [f"{t*1000:.1f}ms" for t in async_times])
    os.environ.pop("FUSION_ASYNC_DENOISE", None)

    sync_med = _median(sync_times)
    async_med = _median(async_times)
    speedup = sync_med / async_med if async_med > 0 else float("inf")

    print("=" * 64)
    print(
        f"#180 async denoise bench  (tiny DiT, {args.steps} steps x {args.runs} runs)"
    )
    print(
        f"  sync  median: {sync_med*1000:8.1f} ms   per-step: {sync_med/args.steps*1000:6.1f} ms"
    )
    print(
        f"  async median: {async_med*1000:8.1f} ms   per-step: {async_med/args.steps*1000:6.1f} ms"
    )
    print(f"  speedup:      {speedup:8.3f}x   ({(speedup-1)*100:+.1f}%)")
    print(f"  peak mem  sync: {sync_mem/1e6:8.1f} MB   async: {async_mem/1e6:8.1f} MB")
    print("=" * 64)
    print("NOTE: tiny DiT overstates win (CPU build is large fraction of step).")
    print("      Real 14B (P3) win converges to ~CPU-build fraction (expect 1-5%).")
    print(
        "      mem async ~= sync confirms #146 no-OOM (async_eval materializes/step)."
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
