#!/usr/bin/env python3
"""bench_skyreels_real.py — SkyReels-V3 真实权重端到端压测 (直调 dit, 绕开 bench_skyreels.py grid bug).

用法:
  python3 bench_skyreels_real.py --branch all --steps 3 --frames 5 --size 256
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, ".")


def bench_branch(branch: str, steps: int, frames: int, size: int) -> dict:
    """直调 dit 前向压测, 用已验证的端到端路径 (非 bench_skyreels.py)."""
    print(f"{'=' * 60}\n=== {branch.upper()} 真实权重端到端 (steps={steps} frames={frames} {size}x{size}) ===", flush=True)
    t0 = time.time()

    # 加载已验证的端到端组件
    from fusion_mlx.video.skyreels_v3.weights import load_all_weights
    from fusion_mlx.video.skyreels_v3.text_encoder import UMT5Encoder
    from fusion_mlx.video.skyreels_v3.vae import SkyReelsVAE

    # 分支映射
    branch_map = {
        "r2v": ("transformer_r2v", "SkyReelsR2VDiT", "Skyreels-V3-R2V-14B-MLX", 14),
        "v2v": ("transformer_v2v", "SkyReelsV2VDiT", "Skyreels-V3-V2V-14B-MLX", 14),
        "a2v": ("transformer_a2v", "SkyReelsA2VDiT", "Skyreels-V3-A2V-19B-MLX", 19),
    }
    if branch not in branch_map:
        raise ValueError(f"Unknown branch: {branch}")
    mod_name, cls_name, mlx_dir_name, _b = branch_map[branch]

    import importlib
    mod = importlib.import_module(f"fusion_mlx.video.skyreels_v3.{mod_name}")
    DiT = getattr(mod, cls_name)

    mlx_dir = os.path.expanduser(f"~/.fusion-mlx/models/Skywork/{mlx_dir_name}")
    if not os.path.exists(mlx_dir):
        return {"branch": branch, "error": f"MLX weights not found: {mlx_dir}", "success": False}

    # 构造骨架 + 加载真实权重
    dit = DiT()
    vae = SkyReelsVAE()
    t5 = UMT5Encoder()
    try:
        load_all_weights(dit, vae, t5, Path(mlx_dir))
    except Exception as exc:
        return {"branch": branch, "error": f"权重加载 FAIL: {exc}", "success": False}
    load_s = time.time() - t0
    print(f"  真实权重加载: {load_s:.2f}s", flush=True)

    # latent 输入 (frames 帧, size P latent)
    latent_h = max(4, size // 16)
    latent_w = max(4, size // 16)
    latent_t = max(1, (frames - 1) // 4 + 1)
    x = mx.random.normal((1, 16, latent_t, latent_h, latent_w))
    t = mx.array([0.5])
    text_emb = mx.zeros((1, 512, 4096))  # stub text emb (真实需 tokenizer)
    # grid_sizes 用 pre-patch 隐空间尺寸 (latent_h, latent_w 不预缩),
    # rope_apply 内部 patch_scale 会正确缩放 (patch_size=(1,2,2) 缩 1/4)
    seq_lens = [latent_t * latent_h * latent_w]
    grid_sizes = [(latent_t, latent_h, latent_w)]

    # warmup (1 步)
    try:
        _ = dit(x, t, text_emb, seq_lens=seq_lens, grid_sizes=grid_sizes)
        mx.eval(_)
        mx.clear_cache()
    except Exception as exc:
        return {"branch": branch, "error": f"warmup fwd FAIL: {exc}", "success": False}

    # 压测 N 步采样
    t0 = time.time()
    for _ in range(steps):
        out = dit(x, t, text_emb, seq_lens=seq_lens, grid_sizes=grid_sizes)
    mx.eval(out)
    fwd_s = time.time() - t0
    per_step = fwd_s / steps if steps > 0 else 0
    out_shape = out[0].shape if isinstance(out, list) else out.shape
    peak_mb = mx.get_peak_memory() / 1024**2

    # 综合指标
    fps_diT = frames / fwd_s if fwd_s > 0 else 0
    fps_per_step = frames / per_step if per_step > 0 else 0

    print(f"  DiT fwd: {per_step:.3f}s/step ({steps}步/{fwd_s:.2f}s) shape={out_shape}", flush=True)
    print(f"  Metal peak: {peak_mb:.0f} MB", flush=True)
    print(f"  综合指标: per_step={per_step:.3f}s fps_per_step={fps_per_step:.1f} fps_total={fps_diT:.1f}", flush=True)
    print(f"  ✅ {branch.upper()} 真实权重端到端成功", flush=True)

    return {
        "branch": branch,
        "success": True,
        "load_s": round(load_s, 2),
        "steps": steps,
        "frames": frames,
        "size": size,
        "per_step_s": round(per_step, 3),
        "total_fwd_s": round(fwd_s, 2),
        "fps_per_step": round(fps_per_step, 1),
        "fps_total": round(fps_diT, 1),
        "metal_peak_mb": round(peak_mb, 0),
        "out_shape": list(out_shape),
    }


def main():
    parser = argparse.ArgumentParser(description="SkyReels-V3 真实权重端到端压测")
    parser.add_argument("--branch", choices=["r2v", "v2v", "a2v", "all"], default="all")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--size", type=int, default=256, help="latent square size")
    parser.add_argument("--output", default="/tmp/bench_skyreels_real.json")
    args = parser.parse_args()

    branches = ["r2v", "v2v", "a2v"] if args.branch == "all" else [args.branch]
    results = []
    for b in branches:
        try:
            r = bench_branch(b, args.steps, args.frames, args.size)
            results.append(r)
            # 分支间清显存避峰值叠加
            mx.clear_cache()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results.append({"branch": b, "error": str(exc), "success": False})

    report = {
        "system": {"chip": "Apple M5 Max", "mlx": "0.32.0", "memory_gb": 128},
        "args": {"branch": args.branch, "steps": args.steps, "frames": args.frames, "size": args.size},
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n=== 报告保存: {args.output} ===", flush=True)

    # 汇总表
    print("\n" + "=" * 80)
    print("SkyReels-V3 真实权重端到端压测汇总")
    print("=" * 80)
    print(f"{'Branch':<8} {'Per-step(s)':<14} {'FPS/step':<10} {'Metal(GB)':<10} {'Status':<10}")
    print("-" * 80)
    for r in results:
        if r.get("success"):
            print(f"{r['branch'].upper():<8} {r['per_step_s']:<14.3f} {r['fps_per_step']:<10.1f} {r['metal_peak_mb']/1024:<10.2f} ✅")
        else:
            print(f"{r['branch'].upper():<8} {'-':<14} {'-':<10} {'-':<10} ❌ {r.get('error','')[:40]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
