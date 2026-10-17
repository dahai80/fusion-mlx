# Spec-decode CPU↔GPU bubble-fraction bench (#1199)

Gate: implement the composite-MSL zero-CPU verification loop only if
CPU↔GPU control handoff exceeds **10%** of per-step time. Measured
2026-10-17 on the production spec-decode path. **Result: ~5–6% upper
bound — gate fails.** Combined with #1196 Exit B (no queue-priority
access, upstream ml-explore/mlx#4716), this issue closes
not-worth-implementing.

## Setup

- Debug server 11433: `Qwen3.5-9B-4bit`, profile full, spec-decode
  auto → **n-gram suffix** (max_draft 8, order 8, num_draft 3 — the
  per-step draft→verify loop this issue targets). Model family is GDN
  (recurrent), so draft-model/MTP paths are disabled engine-side; the
  n-gram verify loop exercises the same CPU↔GPU handoff pattern.
- Sustained load: 3 concurrent 800-token generations (repeated rounds,
  ~770 tokens per stream), `requests_running=3` confirmed at capture
  time. Scheduler-reported `completion_tps=931.7` (spec acceptance
  boosting), `completion_eval=1.1ms/token`.

## Methodology

1. **macOS `sample` (1 ms interval, 10 s)** of the server PID during
   active decode — call-graph sample counts per thread, leaf-frame
   classification of the engine decode thread (the one with the deep
   `method_vectorcall` chain driving model steps).
2. **faulthandler USR1 cross-check** — 100 point-in-time all-thread
   stack dumps at 80 ms interval under the same load shape.

`sample` distinguishes where each 1 ms tick lands at the C leaf:
`iokit_user_client_trap` = blocked in the IOGPU driver (kernel
submission / completion wait — GPU pipeline engaged), `__psynch_cvwait`
= parked thread, interpreter/Metal-encoder frames = CPU-active control
work.

## Results — engine decode thread (8418 samples, 10 s)

| Leaf frame class | samples | share |
|---|---|---|
| `iokit_user_client_trap` (GPU driver — submission/completion) | 7907 | 93.9% |
| `__psynch_cvwait` / `mach_msg2_trap` (parked) | 64 | 0.8% |
| CPU-active: `_PyEval_EvalFrameDefault`, `_platform_memmove`, `_xzm_free`, `AGX::*Encoder*` (Metal encoding), `mlx::core::array::~array`, `eval_impl`, alloc/hash | ~445 | **~5.3%** |

faulthandler cross-check (100 active-thread samples): 85% inside MLX
kernel-dispatch frames (`decode_attention`, `make_mask`), ~15% in
Python-level control frames — an **upper bound**, since MLX async
execution overlaps most CPU-side graph-building with in-flight GPU
work (only the non-overlapped remainder is true bubble).

## Conclusion

- `sample`-measured CPU-active share of the decode thread: **≤ 5.3%**;
  the true GPU-idle-due-to-CPU bubble is a subset of that (async
  overlap) — well under the 10% gate.
- faulthandler upper-bound cross-check lands at ≤ 15% but is
  methodologically pessimistic for the same overlap reason; the
  1 ms-resolution `sample` split is the primary number.
- Independent closure condition: #1196 spike exited B — no
  command-queue priority/isolation API exists (Metal has no
  per-queue priority; MLX creates private per-encoder queues;
  upstream feature request ml-explore/mlx#4716). A zero-CPU
  composite verify kernel cannot gain queue-level benefits and the
  measured bubble does not justify it on its own.

**#1199 closes as not-worth-it, gate failed, bench archived (this
document). Raw capture: `sample` output retained in
`/tmp/1199_sample2.txt` at bench time; the tables above are its
verbatim leaf aggregation.**

## Reproduction

```bash
fusion-mlx serve mlx-community/Qwen3.5-9B-4bit --port 11433   # spec auto → suffix
# 3 concurrent 800-token generations against /v1/chat/completions
sample <server-pid> 10 -mayDie -file out.txt                  # during decode
```
