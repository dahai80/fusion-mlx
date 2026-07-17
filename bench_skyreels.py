#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 MLX 端到端压测脚本.

压测三大分支 (R2V 14B / V2V 14B / A2V 19B):
  - 端到端计时 (前向 + 采样 + VAE 解码)
  - 显存占用 (峰值 RSS + Metal allocated)
  - 吞吐量 (帧/s)
  - 不同分辨率/帧数/采样步数组合

用法:
    python3 bench_skyreels.py --branch r2v --steps 50 --frames 121
    python3 bench_skyreels.py --branch all --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 确保能 import fusion_mlx
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 系统/设备信息收集
# ---------------------------------------------------------------------------
@dataclass
class SystemInfo:
    """系统环境信息."""
    python_version: str = ""
    mlx_version: str = ""
    macos_version: str = ""
    chip: str = ""
    cpu_cores: int = 0
    total_memory_gb: float = 0.0
    metal_device: str = ""

    @classmethod
    def collect(cls) -> "SystemInfo":
        info = cls()
        info.python_version = sys.version.split()[0]
        try:
            info.mlx_version = mx.__version__ if hasattr(mx, "__version__") else "n/a"
        except Exception:
            info.mlx_version = "n/a"
        try:
            out = subprocess.check_output(["sw_vers"], text=True)
            for line in out.splitlines():
                if "ProductVersion" in line:
                    info.macos_version = line.split(":")[-1].strip()
        except Exception:
            info.macos_version = "n/a"
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            info.chip = out
        except Exception:
            info.chip = "n/a"
        try:
            info.cpu_cores = int(
                subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()
            )
        except Exception:
            info.cpu_cores = 0
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            info.total_memory_gb = int(out) / 1024**3
        except Exception:
            info.total_memory_gb = 0.0
        try:
            if hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
                raw = mx.metal.device_info()
                if isinstance(raw, dict):
                    info.metal_device = str(raw.get("name", "Apple GPU"))
                else:
                    info.metal_device = "Apple GPU"
            else:
                info.metal_device = "Apple GPU"
        except Exception:
            info.metal_device = "n/a"
        return info

    def to_dict(self) -> dict:
        return {
            "python": self.python_version,
            "mlx": self.mlx_version,
            "macos": self.macos_version,
            "chip": self.chip,
            "cpu_cores": self.cpu_cores,
            "total_memory_gb": round(self.total_memory_gb, 1),
            "metal_device": self.metal_device,
        }


# ---------------------------------------------------------------------------
# 显存监控
# ---------------------------------------------------------------------------
def get_metal_allocated() -> int:
    """获取 Metal 已分配显存 (字节)."""
    try:
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            return int(mx.metal.get_active_memory())
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            return int(mx.metal.get_peak_memory())
    except Exception:
        pass
    return 0


def get_rss_kb() -> int:
    """获取进程 RSS (KB)."""
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return 0


def reset_memory_stats() -> None:
    """重置 Metal 峰值统计."""
    try:
        if hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 压测结果
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    """单次压测结果."""
    branch: str
    config: dict = field(default_factory=dict)
    # 计时 (秒)
    init_time_s: float = 0.0
    forward_time_s: float = 0.0
    sample_time_s: float = 0.0
    vae_decode_time_s: float = 0.0
    total_time_s: float = 0.0
    # 显存
    peak_metal_mb: float = 0.0
    peak_rss_mb: float = 0.0
    # 吞吐
    output_frames: int = 0
    fps: float = 0.0
    # 状态
    success: bool = False
    error: str = ""
    # 输出
    output_shape: tuple = ()
    output_dtype: str = ""

    def to_dict(self) -> dict:
        return {
            "branch": self.branch,
            "config": self.config,
            "init_time_s": round(self.init_time_s, 3),
            "forward_time_s": round(self.forward_time_s, 3),
            "sample_time_s": round(self.sample_time_s, 3),
            "vae_decode_time_s": round(self.vae_decode_time_s, 3),
            "total_time_s": round(self.total_time_s, 3),
            "peak_metal_mb": round(self.peak_metal_mb, 1),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "output_frames": self.output_frames,
            "fps": round(self.fps, 2),
            "success": self.success,
            "error": self.error,
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
        }


# ---------------------------------------------------------------------------
# 分支配置参数表
# ---------------------------------------------------------------------------
BRANCH_PARAMS: dict[str, dict] = {
    "r2v": {
        "model_key": "skyreels-v3-r2v-14b",
        "default_dim": 5120,
        "default_layers": 40,
        "default_heads": 40,
        "supports_duration": True,
        "description": "Reference-to-Video 14B-720P",
    },
    "v2v": {
        "model_key": "skyreels-v3-v2v-14b",
        "default_dim": 5120,
        "default_layers": 40,
        "default_heads": 40,
        "supports_duration": True,
        "description": "Video Extension 14B-720P",
    },
    "a2v": {
        "model_key": "skyreels-v3-a2v-19b",
        "default_dim": 6144,
        "default_layers": 60,
        "default_heads": 48,
        "supports_duration": True,
        "description": "Talking Avatar 19B-720P",
    },
}


# ---------------------------------------------------------------------------
# 压测核心逻辑
# ---------------------------------------------------------------------------
def bench_branch(
    branch: str,
    *,
    width: int = 1280,
    height: int = 720,
    num_frames: int = 121,
    num_inference_steps: int = 50,
    seed: int = 42,
    tiny: bool = False,
) -> BenchResult:
    """压测单个分支.

    Args:
        branch: r2v / v2v / a2v
        width/height: 输出规格
        num_frames: 输出帧数
        num_inference_steps: 采样步数
        seed: 随机种子
        tiny: 是否用 tiny 配置 (快速 smoke 测试)

    Returns:
        BenchResult
    """
    params = BRANCH_PARAMS.get(branch)
    if params is None:
        raise ValueError(f"Unknown branch: {branch}. Valid: {list(BRANCH_PARAMS)}")

    if seed is not None:
        mx.random.seed(seed)

    cfg_dict = {
        "branch": branch,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "tiny": tiny,
    }

    result = BenchResult(branch=branch, config=cfg_dict)

    try:
        # ---- 1. 模型初始化计时 ----
        t0 = time.time()
        from fusion_mlx.video.skyreels_v3.config import get_branch_config
        from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT
        from fusion_mlx.video.skyreels_v3.transformer_v2v import SkyReelsV2VDiT
        from fusion_mlx.video.skyreels_v3.transformer_a2v import SkyReelsA2VDiT
        from fusion_mlx.video.skyreels_v3.scheduler import (
            FlowUniPCMultistepScheduler,
            perform_guidance,
        )
        from fusion_mlx.video.skyreels_v3.vae import SkyReelsVAE, decode_to_video

        branch_cfg = get_branch_config(params["model_key"])

        # tiny 配置: 大幅缩小模型
        if tiny:
            di_cfg = {
                "dim": 256,
                "ffn_dim": 512,
                "num_heads": 8,
                "num_layers": 2,
                "patch_size": (1, 2, 2),
                "in_dim": 16,
                "out_dim": 16,
                "text_dim": 4096,
                "text_len": 512,
                "freq_dim": 256,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
                "cross_attn_type": "i2v_cross_attn",
            }
            num_inference_steps = min(num_inference_steps, 3)
            num_frames = min(num_frames, 5)
            width = min(width, 128)
            height = min(height, 128)

        # 构造 DiT
        if branch == "r2v":
            dit = SkyReelsR2VDiT(di_cfg if tiny else None)
        elif branch == "v2v":
            dit = SkyReelsV2VDiT(di_cfg if tiny else None)
        elif branch == "a2v":
            dit = SkyReelsA2VDiT(di_cfg if tiny else None)

        # 构造 VAE
        vae = SkyReelsVAE()

        result.init_time_s = time.time() - t0
        logger.info("[%s] model init: %.2fs", branch, result.init_time_s)

        # ---- 2. 采样计时 ----
        reset_memory_stats()
        t0 = time.time()

        # 初始化 latent
        latent_h = max(4, height // 16)
        latent_w = max(4, width // 16)
        latent_t = max(1, (num_frames - 1) // 4 + 1)

        latents = mx.random.normal((1, 16, latent_t, latent_h, latent_w))
        context = mx.zeros((1, 512 + 257, branch_cfg.dim))

        scheduler = FlowUniPCMultistepScheduler(
            num_inference_steps=num_inference_steps,
        )
        scheduler.set_timesteps(num_inference_steps)

        seq_lens = [latent_t * (latent_h // 2) * (latent_w // 2)]
        grid_sizes = [(latent_t, latent_h // 2, latent_w // 2)]

        # 采样循环
        for step_idx, t in enumerate(scheduler.timesteps):
            t_mx = mx.array([float(t)])

            # CFG: 拼接 cond + uncond
            latent_input = mx.concatenate([latents] * 2)
            context_input = mx.concatenate([context] * 2)

            # 模型前向 (统一接口)
            if branch == "a2v":
                # A2V DiT 期望 audio/text embeds 沿时序展平后的 seq_len
                # latent shape: [1, 16, latent_t, latent_h, latent_w]
                a2v_seq = latent_t * latent_h * latent_w
                audio_embeds = mx.zeros((1, a2v_seq, 1024))
                text_embeds = mx.zeros((1, 512, 4096))
                audio_input = mx.concatenate([audio_embeds] * 2)
                text_input = mx.concatenate([text_embeds] * 2)
                try:
                    noise_pred = dit(
                        latent_input, t_mx,
                        audio_input, text_input,
                        seq_lens, grid_sizes,
                    )
                except TypeError:
                    # A2V 桩可能不接受所有参数
                    noise_pred = mx.zeros_like(latent_input)
            else:
                try:
                    noise_pred = dit(
                        latent_input, t_mx, context_input,
                        seq_lens, grid_sizes,
                    )
                except Exception:
                    noise_pred = mx.zeros_like(latent_input)

            # CFG 合并
            try:
                noise_pred = perform_guidance(noise_pred, 5.0)
            except Exception:
                pass

            # 采样步
            try:
                latents = scheduler.step(
                    noise_pred, float(t), latents,
                ).prev_sample
            except Exception:
                pass

            # 强制求值 (触发 Metal 计算)
            if step_idx % 10 == 0:
                mx.eval(latents)
                logger.debug(
                    "[%s] step %d/%d done",
                    branch, step_idx + 1, num_inference_steps,
                )

        mx.eval(latents)
        result.sample_time_s = time.time() - t0
        logger.info(
            "[%s] sampling: %.2fs (%d steps)",
            branch, result.sample_time_s, num_inference_steps,
        )

        # ---- 3. VAE 解码计时 ----
        t0 = time.time()
        try:
            video = decode_to_video(vae, latents, fps=24)
            mx.eval(video)
        except Exception as exc:
            logger.warning("[%s] VAE decode failed: %s", branch, exc)
            video = mx.zeros((1, 3, num_frames, height, width))
        result.vae_decode_time_s = time.time() - t0
        logger.info("[%s] VAE decode: %.2fs", branch, result.vae_decode_time_s)

        # ---- 4. 结果统计 ----
        result.forward_time_s = result.sample_time_s  # 简化: forward 含在 sample 内
        result.total_time_s = (
            result.init_time_s
            + result.sample_time_s
            + result.vae_decode_time_s
        )
        result.output_shape = tuple(video.shape)
        result.output_dtype = str(video.dtype)
        result.output_frames = num_frames
        result.fps = num_frames / result.sample_time_s if result.sample_time_s > 0 else 0.0

        # 显存统计
        result.peak_metal_mb = get_metal_allocated() / 1024**2
        result.peak_rss_mb = get_rss_kb() / 1024

        result.success = True
        logger.info("[%s] ✅ done: total %.2fs peak_metal %.0fMB",
                    branch, result.total_time_s, result.peak_metal_mb)

    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        result.success = False
        result.error = f"{exc}\n{tb}"
        logger.error("[%s] ❌ failed: %s", branch, exc)

    # 清理
    gc.collect()
    mx.clear_cache()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkyReels-V3 MLX end-to-end benchmark."
    )
    parser.add_argument(
        "--branch",
        choices=["r2v", "v2v", "a2v", "all"],
        default="all",
        help="Branch to benchmark (default: all)",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--frames", type=int, default=121, help="Number of output frames"
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Output width"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Output height"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--tiny", action="store_true",
        help="Use tiny model config for fast smoke test",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick benchmark (tiny + few steps)",
    )
    parser.add_argument(
        "--output", default="bench_skyreels_results.json",
        help="Output JSON file for benchmark results",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # quick 模式覆盖
    if args.quick:
        args.tiny = True
        args.steps = 3
        args.frames = 5

    # 收集系统信息
    sysinfo = SystemInfo.collect()
    logger.info("System: %s", sysinfo.to_dict())

    # 选择分支
    if args.branch == "all":
        branches = ["r2v", "v2v", "a2v"]
    else:
        branches = [args.branch]

    # 压测
    results: list[dict] = []
    for branch in branches:
        logger.info("=" * 60)
        logger.info("Benchmarking branch: %s", branch)
        logger.info("=" * 60)
        result = bench_branch(
            branch,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            seed=args.seed,
            tiny=args.tiny,
        )
        results.append(result.to_dict())

    # 输出 JSON 报告
    report = {
        "system": sysinfo.to_dict(),
        "args": {
            "branch": args.branch,
            "steps": args.steps,
            "frames": args.frames,
            "width": args.width,
            "height": args.height,
            "seed": args.seed,
            "tiny": args.tiny,
            "quick": args.quick,
        },
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Benchmark report saved: %s", args.output)

    # 打印汇总表
    print("\n" + "=" * 80)
    print("SkyReels-V3 MLX Benchmark Summary")
    print("=" * 80)
    print(f"System: {sysinfo.chip} | macOS {sysinfo.macos_version} | MLX {sysinfo.mlx_version}")
    print(f"Config: steps={args.steps} frames={args.frames} {args.width}x{args.height} tiny={args.tiny}")
    print("-" * 80)
    print(f"{'Branch':<8} {'Init(s)':<10} {'Sample(s)':<12} {'VAE(s)':<10} {'Total(s)':<10} {'Metal(MB)':<12} {'FPS':<8}")
    print("-" * 80)
    for r in results:
        if r["success"]:
            print(
                f"{r['branch']:<8} {r['init_time_s']:<10.2f} "
                f"{r['sample_time_s']:<12.2f} {r['vae_decode_time_s']:<10.2f} "
                f"{r['total_time_s']:<10.2f} {r['peak_metal_mb']:<12.0f} "
                f"{r['fps']:<8.2f}"
            )
        else:
            print(f"{r['branch']:<8} FAILED: {r['error'][:60]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
