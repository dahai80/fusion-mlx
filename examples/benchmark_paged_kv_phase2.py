import os
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import stream_generate

_MODEL = os.environ.get("FUSION_PAGED_KV_MODEL", "mlx-community/Qwen3-0.6B-4bit")
_PROMPT = os.environ.get(
    "FUSION_PAGED_KV_PROMPT", "Explain paged attention in three sentences."
)
_MAX_TOKENS = int(os.environ.get("FUSION_PAGED_KV_MAX_TOKENS", "256"))
_OUT = os.environ.get(
    "FUSION_PAGED_KV_REPORT",
    os.path.expanduser("~/fusion/audit/paged-kv-phase2-perf-report.md"),
)

import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _measure(model_path, fused):
    if fused:
        os.environ["FUSION_PAGED_FUSED_KERNEL"] = "on"
    else:
        os.environ.pop("FUSION_PAGED_FUSED_KERNEL", None)

    from fusion_mlx.fusion_takeover.config import FusionConfig
    from fusion_mlx.fusion_takeover.patcher import FusionModulePatcher

    model, tokenizer = mlx_lm.load(model_path)
    cfg = FusionConfig(
        enabled=True,
        paged_kv_enabled=True,
        fused_decode_enabled=fused,
    )
    FusionModulePatcher.patch_model(model, cfg)

    t0 = time.perf_counter()
    n = 0
    for resp in stream_generate(model, tokenizer, _PROMPT, max_tokens=_MAX_TOKENS):
        n += 1
        if n >= _MAX_TOKENS:
            break
    mx.eval(mx.array(0))
    dt = time.perf_counter() - t0
    tps = n / dt if dt > 0 else 0.0
    peak = mx.get_peak_memory() / 1e9
    logger.info(
        "bench path=%s tok/s=%.2f wall=%.2f peak_gb=%.2f",
        "fused" if fused else "base",
        tps,
        dt,
        peak,
    )
    del model
    del tokenizer
    import gc

    gc.collect()
    mx.clear_cache()
    return tps, peak, dt


def main():
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    base_tps, base_peak, base_dt = _measure(_MODEL, fused=False)
    fused_tps, fused_peak, fused_dt = _measure(_MODEL, fused=True)
    delta = (fused_tps - base_tps) / base_tps * 100 if base_tps > 0 else 0.0
    verdict = (
        "Phase 2 fused kernel beats concat path."
        if delta > 0
        else "Phase 2 fused kernel does NOT beat concat path — naive scalar kernel "
        "(one-thread-per-head, scalar dot-product loop) expected slow vs "
        "mx.fast.scaled_dot_product_attention. Two optimization rounds attempted "
        "(simd_shuffle reduction, float4 vectorization) — both abandoned for "
        "correctness (cross-simd-group reduction + bf16 dtype). Kernel needs a "
        "full Steel-style tiled rewrite with threadgroup-memory reduction — out "
        "of this phase's scope. Bit-exact-correct kernel shipped gated OFF by "
        "default (FUSION_PAGED_FUSED_KERNEL=on to enable)."
    )
    md = f"""# Paged-KV Phase 2 Perf Report

> Model: `{_MODEL}` | prompt: `{_PROMPT}` | decode tokens: {_MAX_TOKENS}
> Date: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Decode (L=1) — concat path (Phase 1) vs fused kernel (Phase 2)

| path | tok/s | wall (s) | peak GPU mem (GB) |
|------|-------|----------|--------------------|
| concat (Phase 1) | {base_tps:.2f} | {base_dt:.2f} | {base_peak:.2f} |
| fused kernel (Phase 2) | {fused_tps:.2f} | {fused_dt:.2f} | {fused_peak:.2f} |
| **delta** | **{delta:+.1f}%** | | |

## Verdict

{verdict}
"""
    with open(_OUT, "w") as f:
        f.write(md)
    print(md)
    print(f"report written to {_OUT}")


if __name__ == "__main__":
    main()
