# Issue #180 - Metal Async Dispatch Pipeline (double-buffered denoise)

> Last open issue in fusion-mlx. Active directive: "首先修复所有issue和pr".
> Consumer = fusion-comfyUI Phase 3 (currently DEFERRED per #177 memory) -
> #180 lands as **infrastructure**, env-gated default OFF (zero prod risk), same
> pattern as #177 speculative-denoise.

## The problem

SkyReels video denoise loop `SkyReelsPipeline._denoise_sample`
(`fusion_mlx/video/skyreels_v3/pipelines/__init__.py:467-536`) is **serial**:

```
for step_idx, t in enumerate(timesteps):
    noise_pred = self.dit(...)          # CPU builds DiT graph (no GPU work yet)
    noise_pred = perform_guidance(...)  # lazy
    noise_pred = flicker_fix.filter_step(noise_pred)   # lazy
    latents = scheduler.step(...)       # lazy
    latents = flicker_fix.smooth_temporal(latents)     # lazy (T-frame loop graph)
    ...
    mx.eval(latents)                    # <-- GPU runs; CPU BLOCKS until done
```

Per step the CPU (a) builds the next step's graph only **after** `mx.eval`
returns, while the GPU sits **idle**; (b) then blocks again on the next eval.
GPU idle per step ≈ Python graph-build time. Over 20-30 steps this serializes
CPU graph-build against GPU compute - the gap #180 wants to close (target
<5% GPU idle).

## MLX 0.32 capability constraint (verified)

MLX 0.32.0 exposes **NO** native Metal command-buffer / async-event API:
no `mx.async_event`, `AsyncEvent`, `CommandBuffer`, `command_buffer`, `wait`,
`atomic`. The issue's "split command buffer pipeline" / "async schedule API"
is **not directly available**. Available async primitives:
- `mx.async_eval(arr)` - evaluate an array **non-blocking** (queue + return).
- `mx.synchronize()` - block until all queued work done.
- `mx.metal.start_capture()` / `get_active_memory()` / `device_info()`.

So double-buffering is built on `mx.async_eval` + lazy graph dependency, NOT
explicit Metal command buffers. (Honest scope note in PR/README.)

## Why this is memory-safe (does NOT regress #146)

The per-step `mx.eval(latents)` is a **hard-won #146 OOM fix**: without eval,
the lazy graph `latents_{N+1}=latents_N+dt*dit(...)` accumulates all 30 steps'
DiT forward activations -> 128GB OOM -> kernel kill (comment lines 449-452).

**Key insight**: `mx.async_eval` materializes the graph and frees intermediates
on completion - exactly like `mx.eval` - it just doesn't block the CPU. The
#146 OOM was about **skipping** eval entirely (lazy graph never materialized),
NOT about eval timing. With async:
- `async_eval(latents_N)` queues N; N's forward graph freed once GPU completes.
- CPU builds N+1's graph referencing the **small materialized** `latents_N`
  (graph description only - no activations allocated until N+1 evaluates).
- Peak ≈ **1 forward working set** - identical to sync. NOT 2×, NOT 30×.

Verified no `.item()`/`.tolist()`/`np.array(pending)` sync blockers in the
per-step path: `perform_guidance` (pure arith), `StepCoherenceFilter` (lazy
`beta*prev+(1-beta)*cur`, holds graph-node ref = fine lazy dep),
`temporal_ema_batch` (`.shape` is metadata, no sync; T-frame loop = more CPU
overlap work), `boundary_align` (same). Scheduler `float(self.sigmas[...])` /
`np.array(self.timesteps)` sync on **tiny pre-materialized** arrays independent
of `latents` -> don't block on GPU heavy work.

## Design - env-gated async double-buffer

New branch in `_denoise_sample` (before the existing sync loop), gated by
`FUSION_ASYNC_DENOISE=1` (default OFF -> prod path byte-identical):

```python
if async_denoise_enabled():
    return self._denoise_sample_async(latents, context, seq_lens, grid_sizes)
# ... existing sync loop unchanged ...
```

`_denoise_sample_async` (same body, 3 changes):
1. Replace per-step `mx.eval(latents)` -> `mx.async_eval(latents)` (queue, return).
2. CPU immediately iterates: builds step N+1 graph (references pending
   `latents_N`) while GPU computes N. MLX schedules N+1 after N (dep tracking).
3. After loop: `mx.synchronize()` (drain) before VAE decode / return.

Step-level progress logging unchanged (logs fire on graph-build, not on GPU
completion - acceptable; note in code comment).

## Verification (Rule 4/9/10 - loop until verified)

1. **Correctness parity** (gate): same prompt+seed, sync vs async path ->
   output latents `mx.allclose` (async must be numerically identical: same ops,
   same order, only eval timing differs). Unit test with mock DiT (tiny TINY
   config like test_skyreels_dit_unify.py).
2. **No #146 OOM regression**: monitor `mx.metal.get_active_memory()` peak
   across full loop - assert async peak ≤ sync peak + epsilon (NOT 30×).
3. **GPU-idle measurement harness** `scripts/bench_async_denoise.py`:
   - Wall-clock per step (sync vs async) via `time.perf_counter` around the loop.
   - Optional `mx.metal.start_capture` trace for gap analysis.
   - Report GPU-idle estimate = (wall − sum step-GPU-time)/wall.
4. **Real-model E2E** (CLAUDE.md: 真实加载模型, port 11434, hf-mirror):
   R2V 14B (~28GB local, config already patched by #188/PR#193), 20-step
   denoise, sync vs async -> measure wall-clock + memory peak. This is the
   **decisive** test (per #177 lesson: real model falsified the hypothesis).
   **Honest risk**: win is bounded by per-step graph-build time (estimate
   ~50-300ms vs ~5-15s step -> 1-5%). If graph-build is negligible, GPU is
   already near-100% and #180 delivers marginal gain - machinery still lands
   as infrastructure for future heavier CPU-prep workloads.

## Scope / files

| File | Change |
|---|---|
| `fusion_mlx/video/skyreels_v3/pipelines/__init__.py` | + `_denoise_sample_async` method + `async_denoise_enabled()` gate (env-gated default OFF) |
| `fusion_mlx/video/skyreels_v3/__init__.py` or config | `FUSION_ASYNC_DENOISE` env helper (mirror `speculative_enabled()`) |
| `tests/unit/test_async_denoise_180.py` | NEW - parity (sync vs async allclose) + no-OOM-regression (memory peak) with mock DiT |
| `scripts/bench_async_denoise.py` | NEW - GPU-idle / wall-clock harness (sync vs async) |
| `README.md` | #180 section: async dispatch, env-gated, MLX 0.32 constraint, measured result |
| `MEMORY.md` + memory file | update with #180 outcome (merged or falsified) |

Out of scope: `_denoise_sample_speculative` (already env-gated default-off,
separate module), image_gen.py Flux loop (separate consumer), comfyUI Phase 3
wiring (deferred consumer - #180 is the substrate).

## Phased checkpoints

- **P1**: `_denoise_sample_async` + env gate + mock-DiT parity test (allclose)
  + no-OOM test. CI green. -> commit 1.
- **P2**: `scripts/bench_async_denoise.py` harness. -> commit 2.
- **P3**: Real-model E2E on R2V 14B (port 11434, hf-mirror if weights needed).
  Measure wall-clock + memory. Document result honestly (win or marginal).
  -> commit 3 + PR.

## Decision point for user

- **Cost note**: session already ~$375 (COST CRITICAL). P1+P2 are cheap (mock
  tests, no model load). **P3 (real 28GB model load + 20-step denoise, sync AND
  async) is the expensive part** - required by CLAUDE.md real-model rule and
  is the decisive measurement, but costs significant time/tokens.
- Recommend: **do P1+P2 now** (land machinery, cheap, zero prod risk), then
  **pause before P3** for user to confirm the real-model E2E spend (or defer
  P3 to when comfyUI Phase 3 consumer is actually being built).
- If P3 measurement shows marginal gain (<2%), #180 still closes as
  "infrastructure landed, hypothesis measured marginal - documented" (honest,
  per #177 precedent), NOT over-claimed.
