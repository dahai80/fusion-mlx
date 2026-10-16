# #1200 GQA KV read-amplification — close-out

## Verdict: NOT ACTIONABLE (gate passed, implementation falsified by prototype)

## 1. Gate result (formal)

`gqa_kv_read_amp_1200.json`: decode-shape SDPA (Hq=32/Hkv=4/D=128, fp16/bf16)
useful-BW amplification vs 2GiB-copy streaming reference (450 GB/s):
**2.1-3.3x on BW-bound rows (S >= 32768) — gate > 1.3x formally PASSES.**

## 2. Why the gate result is not actionable

### (a) Premise invalid: named kernels have no decode callers
Issue named `custom_kernels/mfa` + `flash_kda` as bench targets. Path mapping
(explored 2026-10-16): LLM decode runs `mx.fast.scaled_dot_product_attention`
(sched_batch.py) — mfa_bridge is video-only, flash_kda has zero prod callers.
The measured amplification lives **inside Apple's closed Metal SDPA kernel**;
KV layout changes on our side cannot reach it.

### (b) Layout remedy already satisfied
Alignment probe: D=128 (128B-aligned fp16 rows) is the fastest variant;
misaligned D=96 is disproportionately slower. Our KV cache is the already-
aligned contiguous [B, n_kv, S, D] layout the remedy prescribes — nothing to fix.

### (c) Vectorized-fetch remedy falsified by prototype
Two prototype generations (archived under `benchmarks/prototypes/gqa_decode_1200/`):

| gen | design | numerics vs fp16 SDPA ref | best vs SDPA |
|-----|--------|---------------------------|--------------|
| v2  | register-direct: warp=Q-head, simd_sum row dots, register float4 acc, split-KV + combine | maxerr 2e-4 (combine-kernel w*l*acc bug found & fixed) | 0.26-2.40x, never reproducible >1x |
| v3  | shared-mem staging: K/V tile staged ONCE per threadgroup, all 8 GQA heads share (cuts global KV traffic 8x — the actual read-amp remedy) | maxerr 0.04 (fp16-input rounding) | 0.25-0.92x on every shape |

v3 is the correct design for read amplification — it eliminates the 8x KV
re-read across GQA heads — and it still loses to `mx.fast.scaled_dot_product_attention`
on every decode shape (S 32k-131k, splits 32-256). The 8x-traffic win does not
translate: the kernel is latency-bound in score/accum phases and the shared-mem
round trip costs more than the traffic saved.

Consistent with prior falsification (2026-09-21 POC): Apple SDPA runs at ~86%
of roofline; handwritten replacement ceiling is parity at best.

## 3. DoD accounting

- [x] Bench report committed regardless of outcome
- [x] Gate > 1.3x branch: implementation attempted (2 prototype generations,
      numerics validated), no sustainable >1.3x step-time improvement achievable
      — remedy targets closed kernel + already-optimal layout
- [x] Archived: bench + JSON report + prototypes + this close-out

## 4. Re-plan note

Only reachable lever remains: upstream (Apple MLX SDPA) — file upstream issue
if decode-GQA BW ever becomes a production bottleneck. No fusion-mlx-side work.
