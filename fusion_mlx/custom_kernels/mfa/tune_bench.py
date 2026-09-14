# SPDX-License-Identifier: Apache-2.0
# O4.2 bench -> tuning pipeline. Micro-benchmarks each attention backend
# per (head_dim, phase, batch_size) shape and emits a JSON tuning table
# that dispatch_policy._tuning_lookup loads at runtime (env
# FUSION_MFA_TUNING_TABLE). Replaces the static 5-tier device heuristic
# with measured per-shape winners (llama.cpp 310KB tuning-table philosophy).
#
# GATE: produces real timing data on the running device. The table is only
# as good as the shapes benched — extend _BENCH_SHAPES for wider coverage.

from __future__ import annotations

import json
import logging
import os
import time

import mlx.core as mx

logger = logging.getLogger(__name__)

# Representative shapes: (head_dim, is_decode, batch_size). Decode = q_len 1
# + long KV; prefill = q_len == kv_len (chunked). Covers the shapes the
# heuristic in select_backend branches on (hd 64/128/256, b 1/2/4/8).
_BENCH_SHAPES: list[tuple[int, bool, int]] = [
    (64, False, 1),
    (64, False, 2),
    (64, False, 4),
    (64, True, 1),
    (64, True, 2),
    (64, True, 4),
    (128, False, 1),
    (128, False, 2),
    (128, False, 4),
    (128, False, 8),
    (128, True, 1),
    (128, True, 2),
    (128, True, 4),
    (128, True, 8),
    (256, False, 1),
    (256, False, 2),
    (256, True, 1),
    (256, True, 2),
]

_WARMUP_ITERS = 3
_TIMED_ITERS = 10
_PREFILL_SEQ = 512
_DECODE_KV = 2048
_NUM_HEADS = 8


def _build_qkv(head_dim: int, is_decode: bool, batch: int):
    q_len = 1 if is_decode else _PREFILL_SEQ
    kv_len = _DECODE_KV if is_decode else _PREFILL_SEQ
    shape_q = (batch, _NUM_HEADS, q_len, head_dim)
    shape_kv = (batch, _NUM_HEADS, kv_len, head_dim)
    q = mx.random.normal(shape_q).astype(mx.float16)
    k = mx.random.normal(shape_kv).astype(mx.float16)
    v = mx.random.normal(shape_kv).astype(mx.float16)
    return q, k, v, q_len, kv_len


def _time_fn(fn, args) -> float:
    for _ in range(_WARMUP_ITERS):
        out = fn(*args)
        mx.eval(out)
        mx.clear_cache()
    t0 = time.perf_counter()
    for _ in range(_TIMED_ITERS):
        out = fn(*args)
        mx.eval(out)
    dt = (time.perf_counter() - t0) / _TIMED_ITERS
    mx.clear_cache()
    return dt


def _bench_mlxsdpa(q, k, v, scale, mask):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def _bench_mfa_ext(ext, q, k, v, scale, mask, causal):
    # mlx_mfa.flash_attention has no `mask` kwarg — uses attn_bias/causal.
    # Bench the raw causal=False path (mask=None) for apples-to-apples
    # vs the MLX_SDPA raw matmul timing.
    return ext.flash_attention(q, k, v, scale=scale, causal=causal)


def _key(head_dim: int, is_decode: bool, batch: int) -> str:
    phase = "decode" if is_decode else "prefill"
    return f"d{head_dim}_{phase}_b{batch}"


def run_tuning_bench(output_path: str | None = None) -> dict:
    # Time MLX_SDPA vs MFA-ext (STEEL/NAX) per shape. Winner -> tuning entry.
    # PAGED_FUSED is A/B-gated separately (O4.1) — not benched here.
    try:
        from fusion_mlx.custom_kernels.mfa.attention import _HAS_MFA_EXT, _MFA_EXT
    except Exception:
        _HAS_MFA_EXT = False
        _MFA_EXT = None
    logger.info("tune_bench: MFA ext available=%s", _HAS_MFA_EXT)

    table: dict[str, dict] = {}
    for head_dim, is_decode, batch in _BENCH_SHAPES:
        q, k, v, q_len, kv_len = _build_qkv(head_dim, is_decode, batch)
        scale = 1.0 / (head_dim**0.5)
        mask = None  # causal mask adds overhead uniformly; bench raw matmul

        sdpa_dt = _time_fn(_bench_mlxsdpa, (q, k, v, scale, mask))
        candidates = [("MLX_SDPA", sdpa_dt, (1, 64) if is_decode else (64, 64))]

        if _HAS_MFA_EXT and _MFA_EXT is not None:
            try:
                mfa_dt = _time_fn(
                    _bench_mfa_ext, (_MFA_EXT, q, k, v, scale, mask, False)
                )
                # NAX is the ext path on M3+ batch>=2 small-d; STEEL otherwise.
                be_name = "NAX" if (batch >= 2 and head_dim <= 128) else "STEEL"
                if head_dim >= 256:
                    be_name = "STEEL_DSPLIT"
                candidates.append((be_name, mfa_dt, (64, 64)))
            except Exception as exc:
                logger.debug(
                    "MFA ext bench failed for %s: %s",
                    _key(head_dim, is_decode, batch),
                    exc,
                )

        winner = min(candidates, key=lambda c: c[1])
        key = _key(head_dim, is_decode, batch)
        table[key] = {
            "backend": winner[0],
            "block_size": list(winner[2]),
            "ms": round(winner[1] * 1000, 3),
            "candidates": {c[0]: round(c[1] * 1000, 3) for c in candidates},
        }
        logger.info(
            "tune %s: %s %.3fms (candidates=%s)",
            key,
            winner[0],
            winner[1] * 1000,
            {c[0]: round(c[1] * 1000, 3) for c in candidates},
        )

    path = output_path or os.path.expanduser("~/.fusion-mlx/mfa_tuning_table.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)
    logger.info("tune_bench: wrote %d entries to %s", len(table), path)
    print(f"\nMFA tuning table: {len(table)} entries -> {path}")
    for key, entry in sorted(table.items()):
        print(f"  {key}: {entry['backend']} ({entry['ms']}ms)")
    return table


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="MFA per-shape tuning microbench")
    p.add_argument("--output", "-o", default=None, help="Output JSON path")
    args = p.parse_args()
    run_tuning_bench(args.output)
