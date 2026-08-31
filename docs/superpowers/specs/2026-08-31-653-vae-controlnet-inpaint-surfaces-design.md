# #653 VAE-encode / ControlNet / Inpaint-mask Engine + public_api Surfaces — Design

> Status: DRAFT (brainstorming in progress). Not yet approved. No implementation.

## Context gathered (2026-08-31)

### What already exists (verified against current `main`, v0.8.56)

**Engine layer (`VideoGenEngine`, `fusion_mlx/engines/video.py`)** — all 8 stage methods already present and delegated to backend:
- `encode_text`, `load_dit`, `denoise(latent, pos, neg, steps, cfg, seed, num_frames, control=None)`, `unload_dit`
- `load_vae_encoder`, `encode_control(**kwargs)`, `unload_vae_encoder`
- `load_vae`, `decode`, `encode(pixels) -> latent`, `decode_tiled`, `unload_vae`

**Backend abstract (`video_backends/base.py`)** — `VideoBackend` defines all of the above as `NotImplementedError` defaults. `VideoGenParams` already carries `controlnet_image` / `controlnet_strength` / `control_type` (base.py:62-67) + VACE `control_video` / `control_mask` / `reference_images` / `camera_conditions`.

**public_api** — `fusion_mlx/public_api.py` re-exports the **engine classes** (not methods). `VideoGenEngine` is public; its methods are the stable surface. **So "export through public_api" = already satisfied at the class level** — the real work is making the methods exist and work on backends.

**Model-layer VAE `.encode()` exists in EVERY video family + image sdxl/sd3:**
| family | encode | backend wired? |
|---|---|---|
| wan2 (`vae.py:611`), wan2/vae22 | ✓ | ✓ (only Wan2Backend) |
| cosmos (`vae.py:786`) | ✓ | ✗ |
| hunyuanvideo | ✓ | ✗ |
| cogvideox | ✓ | ✗ |
| ltx_video_legacy | ✓ | ✗ |
| skyreels_v3 | ✓ | ✗ |
| minimax_h3 | ✓ | ✗ |
| svd | ✓ | ✗ |
| mage / latentsync / musetalk | ✓ | (not engine backends) |
| image sdxl (`vae.py:291`), sd3 (`vae.py:267`) | ✓ | ✗ (ImageGenEngine) |

→ "All backends" = wire the **existing** model encoders through 11 backends' `load_vae_encoder`/`encode`/`unload_vae_encoder`. NOT porting encoders.

**ControlState (`video/wan2/stage.py`)** — the Wan2 control mechanism. Fields: `control_hidden_states` (VACE list), `control_scales`, `y_camera` (Fun-Camera), `y_i2v` (I2V-14B channel-concat), `z_img` + `i2v_mask` + `i2v_mask_tokens` (TI2V-5B blend), `is_i2v_mask_blend`, `is_i2v_channel_concat`. Fed into DiT `forward(control_hidden_states=, control_scales=, y_camera=)`. Built by `Wan2Backend.encode_control`.

**WanControlnet adapter (`video/adapters/controlnet.py`)** — separate, untested. `ControlNet.compute_residuals()` returns `list[mx.array]` per-block residuals; `modify_denoise_step` is a per-step injection hook. **Does NOT feed ControlState today** — different consumption (residuals added every `stride` blocks in main DiT `_run_blocks`).

**SkyReels ControlNet (already wired at model layer)** — `_denoise_sample` (pipelines/__init__.py:605-724) has a FULL ControlNet path using SkyReels's OWN `create_adapter("controlnet")` (NOT WanControlnet): per-step `modify_denoise_step` + `get_residuals`, fed to `dit(..., controlnet_residuals=, controlnet_stride=)`. Triggered by `config.controlnet_image`/`control_type`/`controlnet_strength`. Engine layer (`SkyReelsBackend.encode_control`) does NOT expose this — that's the #653 gap. Two distinct mechanisms (Rule 7 — kept separate, uniform engine surface only).

**Inpaint/mask precedent** — VACE `control_mask` path in `generate.py:136-204`: `inactive = video*(1-mask)`, `reactive = video*mask`, patch-downsample mask to latent tokens. TI2V-5B `i2v_mask` blend in `ControlState`. But **no generic "denoise only masked region" hook** — mask handling is path-specific, not a denoise-level knob.

### ComfyUI consumer (separate repo — `fusion_comfyui_plugin`)
5 dead-path stub nodes need these surfaces: `VAEEncodeForInpaint` (encode+mask), `InpaintModelConditioning` (mask conditioning), `ControlNetApply`/`ControlNetApplyAdvanced`/`QwenImageDiffsynthControlnet` (controlnet denoise conditioning). #653 tracks fusion-mlx surface only; ComfyUI wiring follows.

### User decisions (this session)
- Backend scope: **all video backends** (wire existing model encoders through all 11 non-Wan2 backends; not portable to families without encode).
- ControlNet gap: **wire WanControlnet adapter into ControlState** (one control path, not two; Rule 7).

## User decisions locked

- **Inpaint semantics: frozen-region.** Latent-init + per-step re-composite `latents = mask*denoised + (1-mask)*init_latent`. Matches ComfyUI `VAEEncodeForInpaint` + `InpaintModelConditioning` canonical contract (the 2 nodes #653 names). NOT VACE conditioning-only.
- **Backend scope: all 11 non-Wan2 backends in this issue.** Wire existing family-VAE `.encode()` through each backend's `load_vae_encoder`/`encode`/`unload_vae_encoder`. Every family VAE verified to have `.encode()`.
- **ControlNet: wire WanControlnet adapter into ControlState** (one control path).

## Approaches

### Approach A — extend ControlState with mask + residuals
Add `controlnet_residuals`, `inpaint_mask`, `inpaint_init_latent` to `ControlState`. `run_denoise` does re-composite in-loop. `encode_control` builds mask+init too.
- Pro: one conditioning struct, single source.
- Con: conflates two orthogonal concepts (controlnet guidance vs inpaint region-freeze). ControlState's docstring/purpose is "VACE/i2v/camera control" — inpaint is not control, it's a denoise-loop gate. Rule 7 risk: two meanings in one type. Mask must thread through every backend's `encode_control` even when only inpaint (no control) is wanted.

### Approach B — mask as denoise sibling param (RECOMMENDED)
`denoise(..., control=None, inpaint_mask=None, init_latent=None)`. Orthogonal to `control`. Re-composite via shared helper `apply_inpaint_mask(latents, init, mask)` called in each backend's `denoise` after `sched.step`. ControlState gains ONLY `controlnet_adapter` + `controlnet_latent` (for the adapter wire-in, Surface B — adapter instance + preprocessed latent, NOT residuals). Surface C (mask) never touches ControlState.
- Pro: matches ComfyUI's orthogonality (InpaintModelConditioning and ControlNetApply are independent nodes). Clean signature. Helper makes per-backend wiring mechanical. ControlState stays single-purpose.
- Con: 12 backends each call the helper in `denoise` (but it's one line + the method already exists on each).

### Approach C — mask via on_step callback
Reuse `on_step`; change its contract to allow latent writeback. No denoise signature change.
- Con: `on_step` is read-only progress today; re-purposing as a latent-mutation hook is an abuse, changes a released callback contract, and ordering vs `sched.step` is fragile. Rejected.

**Recommendation: B.** Orthogonal surfaces stay orthogonal. The per-backend cost is one helper call; the cost of A (semantic conflation) propagates forever.

## Section 1 — Architecture (Approach B)

Three surfaces, one issue, backend-tier work + one engine-tier helper:

```
ComfyUI node ──> VideoGenEngine.<method>  (already public; class re-exported)
                      │ delegates
                      ▼
                VideoBackend (base: NotImplementedError defaults)
                      │ each backend implements
                      ▼
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
 Surface A        Surface B        Surface C
 VAE encode       ControlNet       Inpaint mask
 (11 backends)    adapter→State    (frozen-region)
```

**Surface A — VAE encode (all backends).** Each backend implements `load_vae_encoder`/`encode`/`unload_vae_encoder` over its own family VAE (`.encode()` already exists at model layer). Wan2 is the reference (already done). Pattern: mirror `Wan2Backend`'s `load_vae_encoder` (load encoder weights or reuse decoder-weights-into-full-VAE with the #670 full-VAE fallback) + `encode(pixels)` (preprocess pixels → `vae.encode` → latent) + `unload`. Backends whose family VAE lacks a separable encoder (e.g. some ship decoder-only checkpoints) reuse the #670 full-VAE-fallback pattern.

**Surface B — ControlNet adapter → ControlState (Wan2).** Add `controlnet_adapter: ControlNet | None` + `controlnet_latent: mx.array | None` to `ControlState` (NOT residuals — residuals are step-dependent). `Wan2Backend.encode_control` gains a path: when `controlnet_image` is set (and not VACE/i2v/camera), build a `ControlNet` adapter (adapters/controlnet.py), preprocess control image to `controlnet_latent` (one-time expensive part), store both on `ControlState`. `run_denoise` calls `adapter.compute_residuals(latents, t, context)` each step and adds residuals to the DiT hidden states. `ControlState` carries adapter+latent, NOT precomputed residuals.

**Surface B — SkyReels: model-layer ControlNet already exists, expose it.** SkyReels `_denoise_sample` (pipelines/__init__.py:605-724) ALREADY has full per-step ControlNet: `create_adapter("controlnet")`, `modify_denoise_step`, `get_residuals`, `dit(..., controlnet_residuals=, controlnet_stride=)`, fed by `self.config.controlnet_image` / `control_type` / `controlnet_strength`. This is SkyReels's OWN adapter (`create_adapter`), NOT WanControlnet. **Rule 7 — surface, don't average:** two distinct ControlNet mechanisms (WanControlnet for Wan2, SkyReels `create_adapter("controlnet")` for SkyReels) stay separate. Engine surface = uniform: `encode_control(controlnet_image=, control_type=, controlnet_strength=)` on both backends; each routes to its own model-layer adapter. SkyReels's `encode_control` sets `pipeline.config.controlnet_image/control_type/controlnet_strength` (the params its `_denoise_sample` already reads); no new model-layer code on SkyReels.

**Surface C — Inpaint frozen-region mask.** `denoise(..., inpaint_mask=None, init_latent=None)`. Shared helper `apply_inpaint_mask(latents, init, mask) -> latents` in `engines/video_backends/_inpaint.py` (neutral, not Wan2-specific). Each backend's `denoise` calls it after `sched.step`: `if inpaint_mask is not None: latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)`. Mask is patch-downsampled to latent resolution (precedent: `generate.py:190-204` VACE mask reshape). `init_latent` comes from `encode(base_image)` (Surface A) — surfaces compose. **Wan2:** insert in `run_denoise` loop (stage.py) after `sched.step`. **SkyReels:** insert in `_denoise_sample` loop (pipelines/__init__.py:739) after `scheduler.step(...).prev_sample` — the per-step loop is explicit and exposed. Both viable. Other 10 backends defer (no `denoise` — see Section 2 gap).

## Section 2 — File structure + per-backend task shape

### Scope ruling (2026-08-31, user-confirmed)

- **Surface A (VAE encode): all 11 real backends.** Wire existing family-VAE `.encode()` through each backend's `load_vae_encoder`/`encode`/`unload_vae_encoder`. Standalone — no denoise needed.
- **Surface B (ControlNet adapter) + Surface C (inpaint mask): Wan2 + SkyReels only.** Only those 2 backends have a `denoise()` loop today. Wiring B/C requires the per-step denoise loop.
- **10 denoise-less backends: follow-up issues, NOT this issue.** Each (cogvideox, cosmos, hunyuanvideo, ltx_video_legacy, ltx2_5, ltx2, minimax_h3, opensora, svd, uniworld) routes through a monolith `generate()` with no exposed staged `denoise`. Porting a staged `denoise` per family is a separate, larger effort (the #652 Wan2 port was a multi-task reverse-engineering extraction). This issue files those 10 follow-ups and ships B+C only where a denoise loop exists.

### Verified wiring state (2026-08-31, grep on `main` v0.8.56)

| backend | denoise | load_vae_encoder | encode | Surface A | Surface B+C |
|---|---|---|---|---|---|
| wan2 | ✓ | ✓ | ✓ | reference (done) | target (B+C) |
| skyreels | ✓ | ✗ | ✗ | wire | wire (loop exists, explicit) |
| cogvideox | ✗ | ✗ | ✗ | wire | defer → issue |
| cosmos | ✗ | ✗ | ✗ | wire | defer → issue |
| hunyuanvideo | ✗ | ✗ | ✗ | wire | defer → issue |
| ltx_video_legacy | ✗ | ✗ | ✗ | wire | defer → issue |
| ltx2_5 | ✗ | ✗ | ✗ | wire | defer → issue |
| ltx2 | ✗ | ✗ | ✗ | wire | defer → issue |
| minimax_h3 | ✗ | ✗ | ✗ | wire | defer → issue |
| opensora | ✗ | ✗ | ✗ | wire | defer → issue |
| svd | ✗ | ✗ | ✗ | wire | defer → issue |
| uniworld | ✗ | ✗ | ✗ | stub-only (defer all) | n/a |

### Files created

- `fusion_mlx/engines/video_backends/_inpaint.py` — neutral helper `apply_inpaint_mask(latents, init_latent, mask) -> mx.array`. Patch-downsamples a (H,W) or (T,H,W) mask to the latent grid (VACE precedent `generate.py:190-204`). Pure MLX ops, no backend dependency. Imported by Wan2 + SkyReels `denoise`.
- `tests/unit/test_inpaint_mask_helper.py` — unit tests for `apply_inpaint_mask`: frozen region unchanged, reactive region follows denoised, mask reshape to latent grid, shape validation, broadcast over batch/time.

### Files modified — Surface A (per-backend, mechanical)

Each of the 10 non-Wan2 real backends + SkyReels gains, mirroring `Wan2Backend` (wan2.py:684-753):
- `load_vae_encoder(self) -> None` — lazy-load family VAE encoder (or reuse full-VAE with the #670 fallback for decoder-only checkpoints).
- `async def encode(self, pixels: mx.array) -> mx.array` — preprocess pixels → family `vae.encode` → 5D latent. Thread-portability via numpy-bridge + executor + `mx.eval` (the stream-ownership invariant from #630: encode runs on worker thread, pixels are main-thread-owned; bridge through numpy on main thread, rebuild `mx.array` inside worker — see wan2.py:697-720 comment).
- `unload_vae_encoder(self) -> None` — drop encoder ref, clear flags, `mx.clear_cache`.

Per-backend files: `skyreels.py`, `cogvideox.py`, `cosmos.py`, `hunyuanvideo.py`, `ltx_video_legacy.py`, `ltx2_5.py`, `ltx2.py`, `minimax_h3.py`, `opensora.py`, `svd.py`.

### Files modified — Surface B

**Wan2 (WanControlnet adapter):**
- `fusion_mlx/video/wan2/stage.py` — `ControlState` gains `controlnet_adapter: ControlNet | None = None` + `controlnet_latent: mx.array | None = None` (NOT residuals — see Section 1 B-detail: residuals are step-dependent, computed in-loop). `run_denoise` loop calls `adapter.compute_residuals(latents, t, context)` per step and adds residuals to DiT hidden states.
- `fusion_mlx/engines/video_backends/wan2.py` — `encode_control` gains a `controlnet_image` path: when `controlnet_image` set (and not VACE/i2v/camera), build `ControlNet` adapter, preprocess control image to `controlnet_latent`, store both on `ControlState`.
- `fusion_mlx/video/adapters/controlnet.py` — verify `ControlNet.compute_residuals` signature matches the per-step call (current signature: `compute_residuals(latents, t, context, control_states)`). Light tests to pin the contract before wiring (currently untested).

**SkyReels (own adapter — model layer DONE, engine layer = config plumbing):**
- `fusion_mlx/engines/video_backends/skyreels.py` — `encode_control(controlnet_image=, control_type=, controlnet_strength=)` sets `pipeline.config.controlnet_image` / `control_type` / `controlnet_strength`. The staged `denoise()` calls `_denoise_sample` which already reads these (pipelines/__init__.py:607-609) and runs the full per-step ControlNet path (605-724). NO model-layer change on SkyReels — purely engine-layer config plumbing.
- NO `ControlState` on SkyReels — it has no ControlState (Wan2-only type). SkyReels control is config-driven, not struct-driven. Rule 7: keep the two mechanisms separate, uniform engine surface only.

### Files modified — Surface C (Wan2 + SkyReels)

- `fusion_mlx/engines/video.py` — `denoise` signature gains `inpaint_mask: mx.array | None = None`, `init_latent: mx.array | None = None`; forwarded to backend `denoise`.
- `fusion_mlx/engines/video_backends/base.py` — abstract `denoise` signature gains the two params (default `None`, backward-compatible).
- `fusion_mlx/engines/video_backends/wan2.py` — `denoise` accepts the two params, threads them into `run_denoise`.
- `fusion_mlx/video/wan2/stage.py` — `run_denoise` loop inserts `if inpaint_mask is not None: latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)` after each `sched.step`.
- `fusion_mlx/engines/video_backends/skyreels.py` — `denoise` accepts the two params, threads them into `_denoise_sample` via new kwargs `inpaint_mask`/`init_latent`.
- `fusion_mlx/video/skyreels_v3/pipelines/__init__.py` — `_denoise_sample` gains `inpaint_mask: mx.array | None = None`, `init_latent: mx.array | None = None` kwargs. Insert after line 743 (`latents = scheduler.step(...).prev_sample`), before flicker smooth: `if inpaint_mask is not None: latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)`. Loop is explicit and exposed — confirmed viable. Same insertion point applies to `_denoise_sample_async` (780) and `_denoise_sample_speculative` (898/1501) if those paths ship — but default path first; async/speculative gated behind env flags (default off per #177/#180), so default path insertion suffices for v1.

### Per-backend task shape (Surface A)

One task per backend, same shape:
1. Locate family VAE `.encode()` at model layer (verified to exist for all 11).
2. Add `load_vae_encoder`/`encode`/`unload_vae_encoder` mirroring Wan2.
3. Unit test: fake VAE encoder returns a fixed latent; assert `encode` returns 5D, raises on wrong-ndim pixels, thread-bridges correctly.
4. Real-model roundtrip test (gated `FUSION_MLX_REAL_MODEL_TESTS`): `encode(pixels)` → `decode(latent)` reconstructs; assert shape + finite values. Composes Surface A (encode) with existing `decode`.
5. Commit per backend (or batch the 9 identical-shape ones — see execution).

**Batchable:** the 9 denoise-less backends' Surface A are identical-shape mechanical edits (same 3 methods, family-specific VAE path only). Per the subagent batching rule, compose ONE dispatch for the 9 + a separate task each for SkyReels (also needs B/C-adjacent review) and any backend whose VAE lacks a separable encoder (#670 fallback).

## Section 3 — Data flow

### Surface A (VAE encode) — all backends

```
caller (main thread)
  │ pixels: mx.array (T,H,W,3) or (1,T,H,W,3)
  ▼
VideoGenEngine.encode(pixels)
  │ delegates
  ▼
Backend.encode(pixels)
  │ 1. lazy-load encoder if None (load_vae_encoder)
  │ 2. ndim guard: 4 or 5 only, else ValueError
  │ 3. numpy-bridge on MAIN thread: src_np = np.array(pixels[0] if ndim==5 else pixels)
  │    (stream-ownership: pixels built on main thread; worker can't eval lazy graph
  │     referencing main thread's GPU stream — #630 invariant)
  │ 4. run in executor("video"):
  │      x = _pixels_thwc_to_ncthw(src_np)   # family-specific preprocess
  │      lat = family_vae.encode(x)           # model-layer .encode()
  │      lat_5d = lat if lat.ndim==5 else lat[None]
  │      mx.eval(lat_5d)                      # force on worker-owned stream
  │      return lat_5d
  ▼
latent: mx.array (1,z,t,h,w)
```

Compose with `decode`: `latent = encode(base_image)` (Surface A) feeds `init_latent` (Surface C).

### Surface B (ControlNet) — Wan2 only

```
caller
  │ controlnet_image: str (path) or pixels
  ▼
VideoGenEngine.encode_control(controlnet_image=...)
  │ delegates
  ▼
Wan2Backend.encode_control
  │ detect: controlnet_image set AND not VACE/i2v/camera
  │ 1. build ControlNet adapter (adapters/controlnet.py)
  │ 2. preprocess control image → controlnet_latent (one-time, expensive)
  │ 3. store on ControlState: controlnet_adapter, controlnet_latent
  ▼
VideoGenEngine.denoise(latent, ..., control=ControlState)
  │ delegates
  ▼
Wan2Backend.denoise → run_denoise (stage.py)
  │ per step t:
  │   residuals = adapter.compute_residuals(latents, t, context)   # step-dependent
  │   add residuals to DiT hidden states (per-block, every stride)
  │   latents = sched.step(...)
  ▼
denoised latent
```

**Key:** `encode_control` builds the adapter + preprocesses ONCE (the expensive part). `run_denoise` computes `residuals` each STEP (step-dependent — latents change every step). `ControlState` carries the adapter instance + preprocessed control latent, NOT precomputed residuals.

### Surface C (inpaint mask) — Wan2 + SkyReels

```
caller
  │ base_image (for frozen region) + mask (H,W) or (T,H,W)
  ▼
init_latent = VideoGenEngine.encode(base_image)        # Surface A composes
mask_patch = patch_downsample(mask, latent_grid)        # to latent token resolution
  ▼
VideoGenEngine.denoise(latent, ..., inpaint_mask=mask_patch, init_latent=init_latent)
  │ delegates
  ▼
Backend.denoise → denoise loop
  │ per step:
  │   latents = sched.step(...)
  │   if inpaint_mask is not None:
  │     latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)
  │     # frozen region: mask*denoised + (1-mask)*init_latent
  ▼
denoised latent (frozen region preserved)
```

`apply_inpaint_mask(latents, init, mask)`: `return mask * latents + (1 - mask) * init`. Mask broadcast over batch/time; patch-downsampled to latent channel/time/grid dims before the loop (not per-step — mask is static).

## Section 4 — Error handling

- **Wrong-ndim pixels (encode):** raise `ValueError` with the received shape (mirror wan2.py:692-695). Fail visibly (Rule 12). NOT a silent clamp.
- **Encoder not loaded:** `encode` lazy-loads if `None` (Wan2 pattern). If family VAE genuinely lacks a separable encoder (decoder-only checkpoint), fall back to #670 full-VAE-load pattern; if that too unavailable, raise `RuntimeError("VAE encoder not available for <family>")` — fail visibly, never return garbage latent.
- **Mask shape mismatch:** `apply_inpaint_mask` validates mask broadcasts to `latents.shape` over batch/time/grid; mismatch → `ValueError` with both shapes. Caught early (before loop), not mid-denoise.
- **ControlNet adapter missing model:** `ControlNet` adapter raises if its model config doesn't match the loaded DiT family. `encode_control` propagates — do NOT swallow into VACE fallback (Rule 7: one control path, surface the mismatch).
- **SkyReels denoise (Surface C):** `_denoise_sample` has an explicit per-step loop (pipelines/__init__.py:661-757) with `latents = scheduler.step(...).prev_sample` at line 739 — mask insertion is clean (resolved, not open). Insert after line 743. Default path only for v1; async/speculative paths (gated off by default per #177/#180) get mask in a follow-up if those paths ship.
- **Thread/stream (encode):** the numpy-bridge is load-bearing. Skipping it → `RuntimeError "no Stream(gpu, N) in current thread"` (#630 GOTCHA). Every backend's `encode` MUST bridge on the caller thread then rebuild `mx.array` in the executor.
- **Unload without load:** `unload_vae_encoder` is idempotent — `None` ref is a no-op (mirror wan2.py:748-753). `mx.clear_cache` always runs.

## Section 5 — Testing

### Unit tests (no model load)

- `test_inpaint_mask_helper.py`:
  - frozen region: `mask=[[1,0]]` → col 0 from init, col 1 from denoised.
  - reactive region: `mask=[[0,1]]` → inverse.
  - mask patch-downsample: (H,W) mask → latent grid shape; (T,H,W) → (t,h,w) grid.
  - shape validation: mask incompatible with latents → `ValueError`.
  - broadcast: (1,z,t,h,w) latent with (t,h,w) mask broadcasts over batch+channel.
- `test_vae_encode_<backend>.py` (per backend, fake VAE):
  - `encode` returns 5D.
  - wrong-ndim pixels → `ValueError`.
  - lazy-load: `encode` works without explicit `load_vae_encoder`.
  - `unload_vae_encoder` idempotent; clears flag.
  - thread-bridge: monkeypatch executor to run synchronously; assert numpy-bridge path taken (pixels detached from main stream).
- `test_controlnet_wire.py` (Surface B, fake adapter):
  - `encode_control(controlnet_image=...)` sets `ControlState.controlnet_adapter` + `controlnet_latent` (not residuals).
  - `run_denoise` calls `adapter.compute_residuals` per step (count == steps).
  - control path routing: `controlnet_image` set WITHOUT VACE/i2v/camera → ControlNet path; WITH VACE → VACE path (not ControlNet); Rule 7 single-path.
  - adapter model mismatch → raises (not swallowed).
- `test_skyreels_controlnet_wire.py` (Surface B SkyReels, config-driven):
  - `encode_control(controlnet_image=, control_type=, controlnet_strength=)` sets `pipeline.config.controlnet_image` / `control_type` / `controlnet_strength` (NOT ControlState — SkyReels has none).
  - staged `denoise()` → `_denoise_sample` reads those config fields (assert via mock pipeline).
  - Rule 7: assert Wan2 path (ControlState) and SkyReels path (config) are distinct — no shared struct.

### Real-model tests (gated `FUSION_MLX_REAL_MODEL_TESTS`)

- VAE encode roundtrip per backend: `encode(pixels)` → `decode(latent)` → shape + finite. Composes Surface A + existing decode.
- Wan2 ControlNet end-to-end: `encode_control(controlnet_image)` → `denoise` → output shape + finite; residuals computed (not None).
- Wan2 inpaint end-to-end: `encode(base)` → `denoise(latent, inpaint_mask, init_latent)` → frozen region matches `init_latent` at those positions (assert `mx.allclose(masked_out, masked_init)`), reactive region differs.
- SkyReels inpaint: `_denoise_sample` has explicit per-step loop (line 739) — real-model test: `encode(base)` → `denoise(latent, inpaint_mask, init_latent)` → frozen region matches init. Confirmed viable.

### Out of scope (follow-up issues, NOT tested here)

- 10 denoise-less backends' Surface B/C: filed as follow-up issues, no tests (no denoise to wire). Surface A IS tested for those 10.

## Section 6 — Follow-up issues to file

10 issues (one per denoise-less backend family) for staged-`denoise` port → then Surface B+C wiring:
cogvideox, cosmos, hunyuanvideo, ltx_video_legacy, ltx2_5, ltx2, minimax_h3, opensora, svd, uniworld(stub).

Each issue: "Port staged `denoise()` to `<backend>` (mirrors #652 Wan2 extraction) to enable ControlNet + inpaint-mask surfaces (#653)." Reference this spec. NOT blocking #653 — #653 ships A on all 11 + B+C on Wan2/SkyReels.

Also: SkyReels async/speculative denoise paths (`_denoise_sample_async` line 780, `_denoise_sample_speculative` lines 898/1501) are gated OFF by default (#177/#180). If those paths ship as default, Surface C mask insertion must be added to them too — follow-up at that time.

## Scope summary

| surface | backends | files | follow-ups |
|---|---|---|---|
| A (encode) | all 11 real | 11 backend files + base.py | #670 fallback where needed |
| B (ControlNet) | Wan2 (WanControlnet) + SkyReels (own adapter) | stage.py, wan2.py, controlnet.py, skyreels.py | 10 denoise-port issues |
| C (inpaint mask) | Wan2 + SkyReels | _inpaint.py, video.py, base.py, wan2.py, stage.py, skyreels.py, pipelines/__init__.py | 10 denoise-port issues (+async/speculative paths if they ship) |
| helper | — | _inpaint.py (new) + test | — |

