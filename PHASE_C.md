# Phase C — Custom Kernels (weight/activation fusion)

Branch: `feat/phase-c-kernels` (off `main` @ 19d50beb).
Scope: the "Phase C" kernel-fusion items from the FR differentiation plan.

This doc is the **evidence-based re-scope** produced by the Phase C kickoff
investigation (2026-07-07). It records what is already done, what is blocked,
and what is genuinely open — with measurements — so future sessions do not
re-derive it or re-implement already-covered work.

## Already covered — NOT Phase C gaps (do not re-implement)

| Item | Where | Evidence |
|---|---|---|
| Fused dequant SDPA (quantized KV → attention, dequant on-the-fly) | `mlx_vlm/turboquant.py` | `turboquant_fused_mse_sdpa_k{k}_v{v}_d{d}` kernel (L2400) + `quantized_attention` (L5105) / `prefill_attention` (L5200) / `decode_attention` (L5771) / `_compiled_split_decode_attention` (L5419) / `_compiled_integer_decode_attention` (L5548). Fusion-mlx uses this via `from mlx_vlm.turboquant import TurboQuantKVCache` (`fusion_mlx/turboquant_kv.py:22`). |
| W4 weight fused MatMul (4-bit weight dequant fused into MatMul) | MLX core | `mx.quantized_matmul` + `mlx.nn.QuantizedLinear` confirmed present (mlx 0.31.2). |

This is WHY the prior "no-promise" assessment held: the two highest-value
fusion targets were already implemented upstream. Phase C is not "build fused
dequant SDPA" — that exists.

## Blocked — not startable this branch

| Item | Blocker |
|---|---|
| NVFP4 | upstream mlx#2962 (NVFP4 dtype support not merged) |
| ANE dispatch | platform: mlx has no ANE backend |
| tensor-parallel | multi-GPU; single-SoC Mac has no inter-device fabric |
| JetSpec | upstream draft-model spec variant, not in mlx |

## Open — startable

### W4A8 fused MatMul (4-bit weight + 8-bit activation) — MEASURED

`mx.quantized_matmul` accepts only **fp16 activations**. W4A8 would also
quantize activations to int8 and fuse that into MatMul. MLX core has no
int8-activation MatMul, so the upside cannot be measured directly; the
viability harness measures the **overhead a fused kernel must absorb**
(quantize A→int8 + dequant round-trip) relative to the W4 baseline.

Harness: `scripts/bench_phase_c_w4a8_viability.py` (mlx 0.31.2, M-series GPU).

| Regime | W4 baseline | A8 roundtrip | overhead % | verdict |
|---|---|---|---|---|
| decode b1 (1×1×4096 @ 11008) | 0.42 ms | 0.21 ms | 49.7% | MARGINAL |
| decode b4 | 0.67 ms | 0.18 ms | 26.4% | MARGINAL |
| prefill 512 | 4.11 ms | 0.19 ms | 4.7% | PROMISING |
| prefill 2048 | 15.9 ms | 0.39 ms | 2.4% | PROMISING |

**Verdict:** W4A8 is **promising at prefill** (compute-bound: overhead tiny,
int8 MatMul upside may net-win) and **likely decode-losing** (bandwidth-bound:
overhead 26–50% of an already-cheap MatMul; int8 does not help the
weight-fetch-dominated bandwidth path). A native W4A8 kernel should be
**prefill-gated** (seq-length threshold), not decode.

**Slice plan (future sessions):**
- Slice 1: `w4a8_fused_matmul` Metal kernel via the `glm_moe_dsa` CMake path
  (`fusion_mlx/custom_kernels/phase_c/csrc/`), prefill-gated. The `phase_c`
  Python module + `w4a8_fused_matmul` fallback wrapper already exist
  (`fusion_mlx/custom_kernels/phase_c/__init__.py`) — falls back to
  `mx.quantized_matmul` (fp16 activations) until the native extension builds.
- Slice 2: engine integration — route prefill MatMul through `w4a8_fused_matmul`
  when native available and seq ≥ threshold; decode stays on `mx.quantized_matmul`.
- Slice 3: end-to-end bench vs baseline (W4) at prefill to confirm net win.

### Fused GDN projections — MEASURED

`mx.compile` measured non-viable for this op (prior session). This harness
measures the **projection-fusion** speedup (the fork's `qwen3_5.py` GDN class
already fuses q/k/v/z into one `in_proj_qkvz` matmul + b/a into one
`in_proj_ba` matmul = 2 matmuls, vs the unfused 6-matmul form) on real
Qwen3.6-27B GDN shapes. It is the lower bound for a full fused megakernel
(which would additionally fuse conv1d + delta update + out_proj, eliminating
the 22528-wide intermediate round-trip through global memory).

Harness: `scripts/bench_phase_c_fused_gdn.py` (mlx 0.31.2, M-series GPU).
Shapes: hidden=4096, key_dim=3072, value_dim=8192, num_v_heads=64 →
qkvz out=22528, ba out=128. Correctness verified: fused-split == unfused.

| Regime | fused (2 matmul) | unfused (6 matmul) | speedup | verdict |
|---|---|---|---|---|
| decode b1 (1×1×4096) | 0.88 ms | 0.90 ms | 1.03x | BREAK-EVEN |
| decode b4 (4×1×4096) | 1.04 ms | 1.12 ms | 1.08x | BREAK-EVEN |
| prefill 512 | 2.25 ms | 2.60 ms | 1.16x | BREAK-EVEN |
| prefill 2048 | 7.85 ms | 11.20 ms | 1.43x | FUSED WINS |
| prefill 8192 | 35.1 ms | 44.2 ms | 1.26x | FUSED WINS |

**Verdict:** projection fusion is a **net win at prefill** (1.26–1.43x at
seq≥2048, the compute-bound regime) and **break-even at decode** (kernel-
launch-bound, no harm). The fork's fused `qkv3_5.py` GDN class is justified
and stays — it is a free prefill win with no decode downside, and it is what
unblocks quant2-all loading (fused `in_proj_qkvz`/`in_proj_ba`).

A **full fused GDN megakernel is NOT worth implementing** this branch: the
projection-fusion floor is only 1.26–1.43x, and the remaining headroom
(conv1d + delta glue, a smaller fraction of GDN cost) does not justify a
multi-day Metal effort relative to W4A8, which is a genuine mlx-core gap with
measured prefill promise (2.4–4.7% overhead to absorb). Defer the megakernel
indefinitely; revisit only if GDN layers become a profiled hotspot on real
Qwen3.6-27B end-to-end runs.

## Artifacts on this branch

- `fusion_mlx/custom_kernels/phase_c/__init__.py` — module scaffolding
  (matches `glm_moe_dsa` convention: `_ext` + `NATIVE_SYMBOLS` +
  `is_native_available`/`has_symbol`/`missing_symbols` + `w4a8_fused_matmul`
  fallback wrapper). Native not built yet → `is_native_available()` is False.
- `scripts/bench_phase_c_w4a8_viability.py` — the W4A8 viability harness (run:
  `python scripts/bench_phase_c_w4a8_viability.py`).
- `scripts/bench_phase_c_fused_gdn.py` — the Fused-GDN projection viability
  harness (run: `python scripts/bench_phase_c_fused_gdn.py`).
- `PHASE_C.md` — this doc.

## Why this is the right kickoff (not "write a kernel now")

Per Rule 12 (fail visibly) and Rule 5 (decide with code, not tokens): the
prior Phase C status was "no-promise" because the obvious targets were
already done upstream. Measuring before writing a Metal kernel prevents
spending a multi-day C++/Metal effort on an op that is decode-losing. The
harness converts "no-promise (unmeasured)" into a measured, regime-specific
verdict that gates the next slice.
