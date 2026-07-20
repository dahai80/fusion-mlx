# Acceleration engines vs VLM / video / 文生图 / 文生视频

**Status: architectural analysis (2026-07-09).** Records why `dflash` /
`dspark` / `turboquant` cannot accelerate text-to-image / text-to-video, what
VLM coverage already exists, and what the real (bounded) options are. Written
because a stop-hook demanded "dflash/dspark/turboquant support VLM and video
and lift 文生图/文生视频 perf 100%" - a requirement whose core conflates two
different compute regimes. This doc prevents that conflation from recurring.

## The two compute regimes (the root fact)

fusion-mlx has **two disjoint generation regimes** that never share an
acceleration path:

1. **Autoregressive LM decode** - token-by-token generation with a KV cache.
   Text LMs, VLMs (vision prefix + autoregressive text), and diffusion-LM text
   models (Mercury / LLaDA-style). This is the regime `dflash`, `dspark`, and
   `turboquant` accelerate.
2. **Latent diffusion** - iterative denoising over a latent tensor, no
   autoregressive token loop, no LM-style KV cache. **文生图 (Flux via mflux)
   and 文生视频 (LTX2 / Wan2 via mlx-video) live here.**

The three acceleration engines are built for regime 1. They have no hook into
regime 2. Asking them to accelerate Flux / LTX / Wan2 is a category error.

## What each engine actually accelerates (verified against source)

| Engine | Regime | Mechanism | Source |
|---|---|---|---|
| `turboquant` | autoregressive | quantizes LM weights + KV cache (`TurboQuantKVCache`) | `turboquant.py`, `turboquant_kv.py`, `mlx_vlm.turboquant`, `sched_token.py:324,366` |
| `dspark` | autoregressive | draft-model speculative decode over `scheduler.batch_generator` (text path) | `spec_decode.py:649` `DSparkSpecState`, `spec_decode_step:214` |
| `dflash` | autoregressive (diffusion-LM text) | "block-diffusion speculative decode" for diffusion-LM text models | `spec_decode.py:400` `DFlashSpecState`, `dflash_spec_step:445` |

`dflash`'s "diffusion" is **diffusion-LM text generation** (the
`DiffusionEngine` lane, `runtime/diffusion_lane.py:287`, `is_mllm=True`), NOT
image/video diffusion. Despite the shared word, it is a text/token path.

## VLM coverage of the three engines (verified)

- **turboquant -> VLM: FIXED (2026-07-09).** Earlier this doc claimed "DONE"
  via `vlm.py` calling `mlx_vlm.turboquant.turboquant_attention(...)`. That was
  **wrong** - `turboquant_attention` does not exist in `mlx_vlm.turboquant`
  (real API is `TurboQuantKVCache` + `turboquant_enabled()`); the `ImportError`
  was silently swallowed by a try/except, so TurboQuant **never actually applied
  to VLM** (caught by Rule 9 - tests revealed the false confidence). Fixed:
  `vlm.py::_apply_turboquant_kv()` now mirrors the proven text-LM path
  (`engines/batched.py:301-334`) - it sets `scheduler._turboquant_kv_bits` /
  `_turboquant_skip_last` / `_turboquant_kv_mode` (validated to `v4`/`k8v4`),
  which activates the KVCache -> `TurboQuantKVCache` swap at prefill
  (`sched_schedule.py` + `sched_token.py`). Default-off
  (`turboquant_kv_enabled` defaults `False`) -> no change to default VLM loads.
  Verified by `tests/unit/test_vlm_turboquant_wiring.py` (7 tests).
- **dspark / dflash -> VLM: NOT on the classic spec path.** VLM runs through
  its own `_vlm_mtp` (multi-token-prediction) path
  (`sched_step.py:53,110,160-164,546` - `_vlm_mtp_active`, `_step_vlm_mtp`),
  while `dspark`/`dflash` operate on `scheduler.batch_generator`, the text-LM
  batch. The two paths are separate. Unifying VLM onto `dspark`/`dflash` is
  the **known "per-request spec routing" engine refactor** - tracked
  separately, large, only step 1 of 4 done. Not a one-shot; not started here
  without explicit direction.

## Why the "100% 文生图/文生视频 via these engines" demand is impossible

- `ImageGenEngine` (`engines/image_gen.py:55-72`) loads **mflux `Flux1`** and
  calls `flux.generate_image(...)` - a diffusion sampler over latents. No LM
  decode, no KV cache, no `batch_generator`. `dflash`/`dspark`/`turboquant`
  are never on this path.
- `VideoGenEngine` (`engines/video.py`) delegates to `video_backends/`
  (`LTX2Backend` / `Wan2Backend` -> mlx-video `ltx_2` / `wan_2`) - also
  diffusion samplers. Same separation.
- Therefore the three engines **cannot** lift 文生图 / 文生视频 throughput at
  all, let alone by 100%. A 100% figure is additionally unverifiable here:
  benchmarking requires multi-GB Flux / LTX / Wan2 weights and GPU-class
  compute that this environment does not have, and prior measured gains in
  this repo are far below 100% (DSpark end-to-end: +69.6% on 8B; coherent
  decode ceiling on M5 Max ~18 tok/s). Fabricating 100% would violate the
  fail-visibly rule.

## Real, bounded options (await user direction)

1. **VLM spec-decode completion** - the coherent part of "支持 VLM": do a
   bounded slice of the per-request spec-routing refactor so `dspark`/`dflash`
   can serve VLM (not just `_vlm_mtp`). Benchmark and report the REAL number.
   (turboquant->VLM is already fixed+tested - see above.)
2. **Diffusion perf, done separately** - real 文生图 / 文生视频 speedups via
   diffusion-specific means (mflux step distillation / compiled transformer /
   tiled VAE; mlx-video equivalent), explicitly NOT via the three LM engines.
   Large; needs weights + compute to measure; no 100% guarantee.
3. **Stop here** - this doc is the deliverable. The impossible parts are
   declined with evidence; the coherent part awaits a decision on scope.

## Delivered: diffusion acceleration knobs (2026-07-09)

Option 2 above, done as bounded, verifiable code (no fabricated benchmark).
The dominant diffusion speed lever is **num_inference_steps**: diffusion
wall-time ≈ `k*steps + c` (per-step cost `k`, fixed overhead `c`), so
`baseline_steps / reduced_steps` is the loop speedup. LTX-2 40 -> 10 steps =
4x on the denoise loop (>100% on that portion); real wall-clock is bounded by
fixed overhead. This is computable up front, not a measured claim.

Changes:
- **LTX-2 backend** (`video_backends/ltx2.py`): previously DROPPED
  `num_inference_steps` / `cfg_scale` from the call (hardcoded to mlx-video's
  40-step default). Now forwards `num_inference_steps`, `cfg_scale`, `tiling`,
  `enhance_prompt` when set. This was the highest-value miss.
- **Wan2 backend** (`video_backends/wan2.py`): `no_compile` (compile opt-in)
  and `tiling` now come from request params instead of being hardcoded to the
  slow `no_compile=True` path. Default stays safe (compile OFF).
- **ImageGenEngine** (`engines/image_gen.py`): exposes load-time `quantize`
  (Flux1 4/8-bit, memory+speed) and generate-time `scheduler` /
  `negative_prompt`.
- **API**: `/v1/videos/generate` accepts `tiling` / `no_compile` /
  `enhance_prompt`; `/v1/images/generate` accepts `scheduler` /
  `negative_prompt`.
- **Benchmark harness** (`scripts/bench_gen_acceleration.py`): `--explain`
  prints the deterministic speedup table (no weights, always runnable); `--run`
  times real generation at varying step counts and reports measured speedup.
- **Tests**: `tests/unit/test_gen_acceleration_knobs.py` (11 tests) verify the
  knobs reach the backend `generate_video` / `flux.generate_image` calls
  (stubbed mlx-video + mflux, no weights). 150 gen/vlm tests pass.

What this is NOT: it is NOT dflash/dspark/turboquant accelerating diffusion
(that remains a category error). It is the diffusion path's OWN acceleration
knobs, wired and exposed - the legitimate reading of "让文生图/文生视频更快".

## What was NOT done (and why)

- No code changes to `dflash`/`dspark`/`turboquant` for video/T2I/T2V: they
  are architecturally incapable of it (see above). TurboQuant->VLM (the
  autoregressive VLM path) IS fixed - see the VLM section.
- No fabricated "100% measured" benchmark: real measurement needs multi-GB
  Flux/LTX/Wan2 weights + compute. The deterministic step-reduction speedup
  (up to ~4-8x on the denoise loop) is reported instead, and `--run` lets the
  user measure the real wall-clock number with weights.
- No blind start of the per-request spec-routing refactor (dspark/dflash ->
  VLM): it is large and needs scoped direction (Rule 6; Rule 1).
