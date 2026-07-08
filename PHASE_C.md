# Phase C — Custom Kernels (weight/activation fusion)

Branch: `feat/phase-c-kernels` (off `main` @ 19d50beb).
Scope: the "Phase C" kernel-fusion items from the FR differentiation plan.

This doc is the **evidence-based record** of the Phase C investigation
(2026-07-07). It records what is already done, what is blocked, and what is
genuinely open — with measurements — so future sessions do not re-derive it
or re-implement already-covered work. All slices are now resolved; see
"Phase C outcome — COMPLETE" at the end.

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

### W4A8 fused MatMul (4-bit weight + 8-bit activation) — MEASURED, NOT BUILDING

`mx.quantized_matmul` accepts only **fp16 activations**. W4A8 would also
quantize activations to int8 and fuse that into MatMul. MLX core has no
int8-activation MatMul (confirmed: `mx.matmul` rejects int8 — "Only inexact
types are supported"), so the upside cannot be measured directly; the
viability harness measures the **overhead a fused kernel must absorb**
(quantize A→int8 + dequant round-trip) relative to the W4 baseline.

Harness: `scripts/bench_phase_c_w4a8_viability.py` (mlx 0.31.2, M-series GPU).

| Regime | W4 baseline | A8 roundtrip | overhead % | verdict |
|---|---|---|---|---|
| decode b1 (1×1×4096 @ 11008) | 0.42 ms | 0.21 ms | 49.7% | MARGINAL |
| decode b4 | 0.67 ms | 0.18 ms | 26.4% | MARGINAL |
| prefill 512 | 4.11 ms | 0.19 ms | 4.7% | PROMISING |
| prefill 2048 | 15.9 ms | 0.39 ms | 2.4% | PROMISING |

**Verdict:** W4A8 is **prefill-promising on overhead alone** (2.4–4.7% to
absorb) and **decode-losing** (26–50%). BUT the overhead analysis shows a
fused W4A8 (round-trip removed) ≈ W4 baseline — the **entire** win rests on
int8 GEMM compute being cheaper than fp16 compute on Apple GPU, which is
speculative (Apple GPU int8 dot-product is not uniformly faster than the
highly-tuned fp16 steel/GEMM path) and **cannot be fairly tested without an
optimized kernel** (a naive kernel would lose to the tuned W4 baseline for
kernel-quality reasons, giving an inconclusive negative).

**Upstream status — maintainer-declined (decisive):** ml-explore/mlx issue
[#1293 "How can we enable w4a8 GEMM in MLX?"](https://github.com/ml-explore/mlx/issues/1293)
(closed). Core maintainer angeloskath: *"A matmul kernel where both matrices
are quantized is not currently implemented... I don't think we plan to
implement this in the near future... The speed is likely to be slower because
we also need to dequantize the second matrix on the fly."* A W4A8 PR to
ml-explore/mlx would go against explicit maintainer guidance — **not a viable
upstream contribution.**

**Decision: NOT BUILDING a native W4A8 kernel.** The measured + maintainer-
confirmed ROI is marginal/speculative at prefill and losing at decode, the
upstream is declined, and a fair test requires a multi-day optimized kernel
for a return that the maintainer assesses as "likely slower." The fork-local
`phase_c` wrapper (`fusion_mlx/custom_kernels/phase_c/__init__.py`) keeps the
`w4a8_fused_matmul` fallback to `mx.quantized_matmul` (fp16 activations) and
remains the extension point should the ROI change — per the maintainer's
suggested path, a future fork-local extension would use `QuantizedBlockLoader`
from `mlx/backend/metal/kernels/quantized.h` to load both `x` and `w`.

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
multi-day Metal effort. Defer the megakernel indefinitely; revisit only if
GDN layers become a profiled hotspot on real Qwen3.6-27B end-to-end runs.

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

## Phase C outcome — COMPLETE (all slices measured)

Per Rule 12 (fail visibly) and Rule 5 (decide with code, not tokens): the
prior Phase C status was "no-promise" because the obvious targets were
already done upstream. Measuring before writing a Metal kernel prevented
spending multi-day C++/Metal effort on ops that are decode-losing or
maintainer-declined. Every Phase C slice now has a measured, documented
outcome:

| Slice | Outcome |
|---|---|
| NVFP4 / fp-quant modes | **DONE** — fusion-mlx convert CLI supports `mxfp4`/`nvfp4`/`mxfp8` (commit b6acd3bd); the dtype support is already on mlx main, no mlx PR needed. |
| Fused dequant SDPA | **Already upstream** — `mlx_vlm.turboquant`; fusion-mlx uses it. Not a gap. |
| W4 weight fused MatMul | **Already upstream** — `mx.quantized_matmul` / `nn.QuantizedLinear`. Not a gap. |
| Fused GDN projections | **DONE** — fork's fused `qwen3_5.py` class measured 1.26–1.43x prefill win, break-even decode; kept (commit 8707d2f). Full megakernel deferred (marginal headroom). |
| W4A8 fused MatMul | **MEASURED, NOT BUILDING** — prefill-marginal, decode-losing; maintainer-declined upstream (mlx #1293). Fork-local `phase_c` wrapper keeps the fallback + extension point. |
| NVFP4 dtype / ANE / TP / JetSpec | **Blocked** — upstream mlx#2962 / no ANE backend / single-SoC / not in mlx. |

**Net:** the high-value Phase C kernel work was already upstream; the
remaining candidates are measured-marginal or maintainer-declined. No native
Metal kernel is worth writing this branch. The `phase_c` module stays as the
scaffolded extension point (fallback to `mx.quantized_matmul`) for future
fork-local experimentation.
