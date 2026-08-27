# #653 VAE-encode Surface — Design Spec (1 of 3)

> Issue: dahai80/fusion-mlx#653 — Expose VAE-encode, controlnet, and inpaint/mask conditioning surfaces on engine + public_api.
>
> This is spec **1 of 3** in the #653 decomposition (decided 2026-08-27):
> 1. **VAE-encode** (this doc) — foundation, lowest risk, unblocks spec 3.
> 2. Controlnet — independent, dead adapter wiring (separate spec, later).
> 3. Inpaint/mask — depends on VAE-encode (separate spec, later).

## Status

Draft 2026-08-27. Brainstorming architectural path, section-by-section approved.

## Context

fusion-comfyui ports ComfyUI nodes to fusion-mlx. `VAEEncodeForInpaint` (and downstream inpaint) need VAE **encode** (image→latent). Model-level `.encode(x)` exists on every VAE (`fusion_mlx/video/wan2/vae.py:584`, `image/sd3/vae.py`, `image/sdxl/vae.py:291`, `video/cosmos/vae.py`), but `VideoGenEngine` / `ImageGenEngine` expose only `decode` / `decode_tiled`. No `encode(pixels)→latent` entry point on the engine surface. This spec adds it.

## Non-Goals

- Controlnet wiring (spec 2).
- Inpaint/mask conditioning (spec 3, depends on this).
- VAE-encode on backends other than wan2 (video) and flux2 (image). Other video backends (cosmos, cogvideo, ltx2) inherit the base `NotImplementedError` and override later when needed.
- Tiled image encode (non-tiled path only for spec 1; tiled encode is a later optimization if large-image OIM appears).
- img2img denoise blending / noise scheduling — encode produces the clean packed latent only; mixing with noise is the caller's job (matches mflux img2img where `add_noise_by_interpolation` is applied AFTER encode, outside the VAE).

## Architecture & Surface Shape

Three new methods, two engines. **No `public_api` edit** — `public_api.py` re-exports engine *classes* (`VideoGenEngine`, `ImageGenEngine`); `encode` is a new method on already-exported classes, so it rides for free.

```
VideoBackend (base)   — async def encode(pixels) -> mx.array   [raises NotImplementedError]
Wan2Backend           — override: lazy-load VAE encoder, run WanVAE.encode, return 5D
VideoGenEngine        — async def encode(pixels: mx.array) -> mx.array   [thin delegate to backend]
ImageGenEngine        — async def encode(pixels: mx.array) -> mx.array   [vae.encode + patchify + bn-norm → packed 4D, inverse of decode_packed_latents]
```

### Input contract

`pixels: mx.array` float32, caller-prepared (fusion-comfyui bridges `np.ndarray float32 [N,H,W,C]` → `mx.array` at call site — same bridge it already uses to consume `decode()` output):

- Video: `(1, T, H, W, 3)` NHWC, values `[0,1]`. Also accepts unbatched `(T, H, W, 3)`.
- Image: `(1, H, W, 3)` NHWC, values `[0,1]`.

### Output contract

Latent, exact mirror of `decode()`'s input:

- Video: `(1, c, t_lat, h_lat, w_lat)` 5D — the shape `VideoGenEngine.decode()` already accepts. `c = z_dim` (16 for wan2.1, 48 for wan2.2). `t_lat = (T - 1) // vae_stride[0] + 1` (WanVAE streaming-causal temporal compression). Normalized `(mu - mean) * inv_std`, the exact inverse of `WanVAE.decode`'s `z / inv_std + mean` denorm (`vae.py:655`).
- Image: `(1, 128, h_lat, w_lat)` 4D **packed** — the shape `ImageGenEngine.decode()` consumes via `decode_packed_latents` and `denoise()` returns (`(batch, c, h, w)` where `c = latent_channels * 4 = 128`, `h_lat = H / 16`, `w_lat = W / 16`). This is **not** the raw `vae.encode` output: it is patchified (2×2 → 4× channels, ½ spatial) then batch-norm normalized, matching `decode_packed_latents`' inverse (`_unpatchify` 128→32 then bn-unnorm `*std+mean`). `encode→decode` roundtrip symmetry depends on this; bare `VAEUtil.encode` produces unpacked `(1,32,H/8,W/8)` which decode cannot consume.

### Thread-affinity

Same rule as existing `decode` / `denoise` stages: encode runs on `get_executor("video")` (wan2) / `get_executor("image")` (flux). MLX Metal streams are thread-local; a caller-cross-thread `pixels` array passed straight through would raise "no Stream(gpu, N) in current thread" at the encode-side `mx.eval`. Therefore:

- The `mx.array(pixels)` source conversion happens **inside the executor thread**, so the source binds to the executor's streams.
- The output latent is `mx.eval`'d **inside the executor** before return (matches the `denoise()` materialization pattern at `engines/video_backends/wan2.py:532`). The returned array is concrete and portable across threads.

### VAE lifecycle (A1 — lazy-load independent encoder)

Stage `load_vae()` loads **decoder-only** (`load_vae_decoder`, `utils.py:329` — `WanVAE(z_dim=16)` with `encoder=False`). This is deliberate: decode-only workflows pay no encoder memory. Encode cannot reuse `_stage_vae` (no encoder present).

- `Wan2Backend._stage_vae_encoder` starts `None`. On first `encode()` call, if `None`, the override lazy-loads via `load_vae_encoder(resolve_vae_path(...), config)` **on the video executor** and caches it in `self._stage_vae_encoder`. This mirrors the monolith i2v path (`generate.py:709`, `vae_enc = load_vae_encoder(...)`) but holds the encoder until `unload_vae()` instead of discarding per-call (avoids repeated reload).
- `unload_vae()` is updated to also free the encoder: `self._stage_vae_encoder = None` + `gc.collect()` + `_clear_mlx_cache()`. One unified cleanup for both VAE halves.
- Image side has no decoder-only split — mflux `__init__` loads the full VAE, so `flux.vae` already exposes `.encode`. Image `encode()` reuses the existing `load_vae()` lifecycle unchanged.

Rationale for A1 over a separate `load_vae_encoder()`/`unload_vae_encoder()` stage pair: the stage contract surface should grow by exactly one method (`encode`), with lifecycle symmetric and collected under the existing `load_vae`/`unload_vae`. A separate pair would make encode's lifecycle asymmetric with decode's (decode: `load_vae`+`decode`+`unload_vae`; encode: `load_vae_encoder`+`encode`+`unload_vae_encoder`) and force fusion-comfyui to wire two extra nodes for no benefit.

## Data Flow & Per-Backend Implementation

### VideoGenEngine.encode — thin delegate

```python
async def encode(self, pixels: mx.array) -> mx.array:
    return await self._backend.encode(pixels)
```

No activity tracking (encode is a sub-step stage, not a top-level generate — matches `decode()` which also has none).

### VideoBackend.encode — base default

```python
async def encode(self, pixels: mx.array) -> mx.array:
    raise NotImplementedError(
        f"{self.name} stage API not implemented (issue #170 phase 2)"
    )
```

Same NotImplementedError-default message and pattern as every other stage method in `VideoBackend` (`engines/video_backends/base.py:165-222` — all use `f"{self.name} stage API not implemented (issue #170 phase 2)"`). Convention over novelty (Rule 11).

### Wan2Backend.encode — core implementation

```python
async def encode(self, pixels: mx.array) -> mx.array:
    from fusion_mlx.video.wan2.stage import encode_wan_vae

    if self._stage_vae_encoder is None:
        await self._load_vae_encoder_stage()
    config = self._ensure_stage_config()

    ndim = pixels.ndim
    if ndim == 5:
        src = pixels[0]          # (T,H,W,3)
    elif ndim == 4:
        src = pixels
    else:
        raise ValueError(
            f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
        )
    vae_enc = self._stage_vae_encoder

    def _encode():
        x = _pixels_thwc_to_ncthw(src)            # (1,3,T,H,W)
        lat = encode_wan_vae(x, config, vae_enc)  # (c,t_lat,h_lat,w_lat)
        lat_5d = lat[None]                         # (1,c,t_lat,h_lat,w_lat)
        mx.eval(lat_5d)
        return lat_5d

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(get_executor("video"), _encode)
    logger.info("stage:vae encode wan2 out_shape=%s", tuple(result.shape))
    return result
```

**New stage helper** `encode_wan_vae(x_ncthw, config, vae_encoder)` in `video/wan2/stage.py`, sibling to the existing `decode_wan_vae` (`stage.py:455`): calls `vae_encoder.encode(x)` and returns the 4D latent `(c, t_lat, h_lat, w_lat)`. No tiling for encode (WanVAE.encode streaming-causal path has no tiled variant upstream; tiling is decode-only).

**Layout conversion** `_pixels_thwc_to_ncthw(src)` where `src` is `(T,H,W,3)` float32 `[0,1]`:
```python
def _pixels_thwc_to_ncthw(src):
    x = mx.array(src).astype(mx.float32)     # bind to executor stream
    return x.transpose(3, 0, 1, 2)[None]      # (1,3,T,H,W) NCTHW
```
The `WanVAE.encode` contract reads `t = x.shape[2]` (`vae.py:594`) — NCTHW puts T at axis 2, matching.

**New private loader** `_load_vae_encoder_stage()` on `Wan2Backend`, mirroring `load_vae()` (`wan2.py:552`):
```python
async def _load_vae_encoder_stage(self) -> None:
    from pathlib import Path
    from fusion_mlx.video.wan2.stage import resolve_vae_path
    from fusion_mlx.video.wan2.utils import load_vae_encoder

    config = self._ensure_stage_config()
    vae_path = resolve_vae_path(Path(self._model_dir))

    def _load():
        return load_vae_encoder(vae_path, config)

    loop = asyncio.get_running_loop()
    self._stage_vae_encoder = await asyncio.wait_for(
        loop.run_in_executor(get_executor("video"), _load),
        timeout=_T5_PRELOAD_TIMEOUT,
    )
    self._stage_flags["vae_encoder"] = True
    gc.collect()
    logger.info("stage:vae_encoder load wan2 active_mem=%s", _active_mem())
```
Runs on `get_executor("video")` (same stream-affinity reason as `load_vae`/`decode` — `wan2.py:565-570`).

**unload_vae update** — add encoder free:
```python
async def unload_vae(self) -> None:
    self._stage_vae = None
    self._stage_vae_encoder = None
    self._stage_flags["vae"] = False
    self._stage_flags.pop("vae_encoder", None)
    gc.collect()
    await _clear_mlx_cache()
    logger.info("stage:vae unload wan2 (decoder+encoder)")
```

### ImageGenEngine.encode — mflux, full inverse of decode_packed_latents

`ImageGenEngine.decode()` calls `flux.vae.decode_packed_latents(latent)`, whose inverse is **not** bare `vae.encode`. The decode path (`flux2_vae/vae.py:45`): packed `(B,128,h,w)` → `_unpatchify` (128→32, 2× spatial) → bn-unnorm `*std+mean` → `vae.decode` → image. The encode path must mirror it backward: image → `vae.encode` → raw `(B,32,H/8,W/8)` → `crop_to_even_spatial` → `patchify_latents` (32→128, ½ spatial) → bn-norm `(x-mean)/std` → packed `(B,128,H/16,W/16)`. This is the exact sequence mflux's own img2img uses (`flux2_klein.py:189-205`).

```python
async def encode(self, pixels: mx.array) -> mx.array:
    flux = self._require_flux()
    if flux.vae is None:
        raise RuntimeError("vae is unloaded; call load_vae().")
    if pixels.ndim != 4:
        raise ValueError(f"encode expects (1,H,W,3); got {tuple(pixels.shape)}")
    if pixels.shape[1] % 16 != 0 or pixels.shape[2] % 16 != 0:
        raise ValueError(
            f"encode expects H,W divisible by 16 (vae_scale*patch); got {tuple(pixels.shape)}"
        )

    def _encode():
        from mflux.models.flux2.latent_creator.flux2_latent_creator import (
            Flux2LatentCreator,
        )
        from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import (
            _Flux2KleinEditHelpers,
        )
        from mflux.models.common.vae.vae_util import VAEUtil

        encoded = VAEUtil.encode(flux.vae, pixels)              # (1,32,H/8,W/8) raw
        encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
        encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
        encoded = Flux2LatentCreator.patchify_latents(encoded)  # (1,128,H/16,W/16)
        encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(
            encoded, vae=flux.vae
        )
        mx.eval(encoded)
        return encoded

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(get_executor("image"), _encode)
    logger.info("stage:vae encode img out_shape=%s", tuple(result.shape))
    return result
```

`flux.vae` is loaded by existing `load_vae()` (mflux loads the full VAE in `__init__`), so no new lifecycle method on the image side. Output `(1, 128, H/16, W/16)` is the 4D packed, bn-normalized form `decode()`/`denoise()` consume — `encode→decode` roundtrips. No tiling for encode in spec 1 (mirror mflux img2img's non-tiled `VAEUtil.encode` path; tiled encode is a later optimization if a large-image OOM appears).

### public_api — no change

`fusion_mlx/public_api.py` re-exports `VideoGenEngine` and `ImageGenEngine` from `fusion_mlx.engines`. Both gain `encode` as a new async method. No new symbol to re-export. The boundary test `tests/unit/test_public_api_boundary.py` continues to pass (it checks the re-exported set, not per-class methods).

## Error Handling

| Situation | Behavior |
|---|---|
| Backend without `encode` override (cosmos / cogvideo / ltx2 today) | Base `VideoBackend.encode` raises `NotImplementedError(f"{name} stage API not implemented (issue #170 phase 2)")`. Same contract as every other stage method — backends opt in per-stage. |
| VAE encoder not loaded (video) | `Wan2Backend.encode` lazy-loads via `_load_vae_encoder_stage()` on first call. If `load_vae_encoder` fails (missing weight file), the original exception propagates with a `logger.error` — fail visibly, no silent fallback. |
| VAE not loaded (image) | `flux.vae is None` → `RuntimeError("vae is unloaded; call load_vae().")`. Mirrors `decode()` guard (`image_gen.py:853`). |
| Wrong `pixels` ndim | Video: ndim not in {4,5} → `ValueError(f"encode expects (T,H,W,3) or (1,T,H,W,3); got {shape}")`. Image: ndim != 4 → `ValueError(f"encode expects (1,H,W,3); got {shape}")`. |
| Image H/W not divisible by 16 | Image: `pixels.shape[1] % 16 != 0` or `shape[2] % 16 != 0` → `ValueError(f"encode expects H,W divisible by 16 (vae_scale*patch); got {shape}")`. Encode patchifies at ½ spatial after vae's 8× downsample, so the input must be divisible by `8*2=16`; an odd size silently mis-patchifies. |
| Wrong `pixels` dtype / range | Range not validated (caller's contract — `[0,1]`). Dtype normalized inside the executor: `_pixels_thwc_to_ncthw` does `.astype(mx.float32)` (video); `VAEUtil.encode` handles mflux's expected dtype (image). Prevents silent garbage from a caller passing uint8. |
| Cross-thread `pixels` array (caller built it on another thread) | Input conversion (`mx.array(src)`) happens inside the executor, binding source to executor streams. Output `mx.eval(lat_5d)` materializes before return. No "no Stream(gpu, N) in current thread" can reach the caller — same guard as `denoise()` (`wan2.py:532`). |
| Encode called after `unload_vae()` | `_stage_vae_encoder` was set to `None` by `unload_vae`, so next `encode` re-lazy-loads. Idempotent recoverability — matches how `load_vae`/`decode` work after an unload+reload cycle. |

## Testing

Real model load for inference tests (project rule: 涉及大模型测试须真实加载). Three tiers:

### Tier 1 — Video encode unit / contract (no model, mocked)

Add to `tests/unit/test_pipeline_stage_api.py`:

- `test_video_encode_not_implemented_base`: instantiate base `VideoBackend` subclass stub (no encode override), call `encode()`, assert `NotImplementedError`.
- `test_wan2_encode_lazy_loads_encoder`: mock `Wan2Backend` with `_stage_vae_encoder=None`, patch `load_vae_encoder` to return a fake vae, call `encode()` once, assert `load_vae_encoder` was called and `_stage_vae_encoder` is now the fake vae.
- `test_wan2_encode_shape`: fake vae `.encode` returns `(16, 3, 8, 16)`; assert `encode()` returns 5D `(1, 16, 3, 8, 16)` and `mx.eval` was invoked on the output.
- `test_wan2_encode_layout`: pass `pixels` `(1, 7, 512, 512, 3)`; capture the array handed to `vae.encode`, assert it is `(1, 3, 7, 512, 512)` NCTHW.
- `test_wan2_encode_ndim_guard`: pass 3D pixels → `ValueError`.
- `test_wan2_unload_vae_frees_encoder`: after lazy-loading the encoder, call `unload_vae()`, assert `_stage_vae_encoder is None` and `_stage_flags` has no `vae_encoder`.

### Tier 2 — Image encode unit (mock mflux)

Add to `tests/unit/test_image_gen_flux2.py`:

- `test_image_encode_requires_started`: engine not started → `RuntimeError` (mflux missing / not started).
- `test_image_encode_requires_vae`: `flux.vae = None` → `RuntimeError("vae is unloaded...")`.
- `test_image_encode_packed_output`: mock `flux.vae` (fake `bn.running_mean`/`running_var`/`eps`, `encode` returns `(1,32,64,64)`); patch `VAEUtil.encode`. Assert output is 4D `(1, 128, 32, 32)` (patchified 32→128, ½ spatial from 64→32) and `mx.eval` invoked. Proves encode produces the packed form decode consumes, not bare `vae.encode`.
- `test_image_encode_bn_normalized`: same mock; assert output ≈ `(patchified - mean)/std` (bn-norm applied), not raw patchified — guards the symmetry with `decode_packed_latents`' bn-unnorm.
- `test_image_encode_ndim_guard`: pass 3D pixels → `ValueError`.
- `test_image_encode_divisibility_guard`: pass `(1, 1023, 1024, 3)` (H not div 16) → `ValueError`.

### Tier 3 — Video encode real-model roundtrip (guards #458 streaming-cache)

New `tests/inference/test_wan2_vae_encode_roundtrip.py`, marked `@pytest.mark.inference` (CI auto-skips; run locally):

- Start: `~/claude-home/fusion-mlx/start.sh start` with wan2.1 1.3B (already installed — memory: workflow-test-results).
- Black-frame determinism: `mx.zeros((1, 1, 512, 512, 3))` → `encode` → `decode` → assert roundtrip correlation ≥ 0.9 (VAE is lossy, not 1.0; but ≥0.9 proves the layout is not scrambled — #458's symptom was correlation ≈ 0).
- 7-frame streaming path: `mx.zeros((1, 7, 512, 512, 3))` → `encode` → `decode` → assert corr ≥ 0.9. Covers the 1+2N chunking (i=0 chunk=1, i=1 chunks=2,2) — the exact path where #458 lived.
- Cleanup: `start.sh stop`, delete intermediate latent temp files, keep logs only (project rule: 验证完成后清理过程数据).

### Tier 4 — Image encode real-model roundtrip

New `tests/inference/test_flux2_vae_encode_roundtrip.py`, marked `@pytest.mark.inference`:

- flux2 klein already installed (memory: e2e-workflow-test-status).
- `mx.zeros((1, 1024, 1024, 3))` → `encode` (→ packed `(1,128,64,64)`) → `decode` (consumes packed) → assert corr ≥ 0.9. This is the load-bearing symmetry test: it fails if encode skips patchify or bn-norm, because `decode_packed_latents` would unpatchify/unnorm wrong.
- Cleanup: stop engine, delete temp files, keep logs.

### Acceptance gate

Spec 1 (#653 VAE-encode) is done when: all Tier 1+2 unit tests pass, `ruff check .` clean, `black --check .` clean, and Tier 3+4 real-model roundtrips hit corr ≥ 0.9 locally (logged, not CI-gated). fusion-comfyui P6 stub internalization for `VAEEncodeForInpaint` then unblocks (tracked in spec 3 / fusion-comfyui P6).

## Global Constraints

- **Language / runtime**: Python ≥ 3.12, MLX on Apple Silicon. No PyTorch in the engine path (fusion-mlx is pure MLX/Metal).
- **Engine surface dtype**: `mx.array` only. No `numpy` on the engine surface contract — fusion-comfyui bridges `np.ndarray`↔`mx.array` at the call site, symmetric with how it consumes `decode()` output today. (`VAEUtil.encode` is mflux-internal, not surface.)
- **Thread-affinity**: every encode path runs on `get_executor("video")` (wan2) or `get_executor("image")` (flux), with output `mx.eval`'d in-executor before return. Non-negotiable — the staged API's cross-thread latent transport depends on it (issue #170, #419).
- **Code style**: 4-space-multiple indent, no docstrings, default logging on every new method (project rules + `engines/video_backends/wan2.py` convention).
- **Stage contract surface**: grows by exactly one public method (`encode`) per engine. No new `load_vae_encoder`/`unload_vae_encoder` stage pair — lifecycle stays under `load_vae`/`unload_vae`. No `public_api.py` edit.
- **Backends**: only `Wan2Backend` (video) and `ImageGenEngine` (image, flux2) get encode now. Other video backends inherit the `NotImplementedError` default and override later if a downstream need appears.
- **Image latent symmetry (load-bearing)**: flux2 encode output MUST be the 4D packed, bn-normalized form `(1,128,H/16,W/16)` — patchify + bn-norm applied after `vae.encode`. `decode()`/`denoise()` consume this packed form via `decode_packed_latents`; a bare `vae.encode` output `(1,32,H/8,W/8)` is NOT consumable and breaks `encode→decode` roundtrip. The Tier 4 roundtrip test is the guard.
- **Version floor**: this lands as a fusion-mlx patch; fusion-comfyui consumes it via `public_api` once a release cuts the dep floor (follow-up, tracked in fusion-comfyui P6).
- **Surgical changes**: touch `engines/video_backends/base.py` (base default), `engines/video_backends/wan2.py` (override + loader + unload edit), `engines/video.py` (engine delegate), `engines/image_gen.py` (image encode), `video/wan2/stage.py` (helper), plus test files. No unrelated refactors.
