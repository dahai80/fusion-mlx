# Multi-Backend Video Generation for fusion-mlx

> Extends `docs/plans/video-generation-engine.md` (#454, LTX-2 only, COMPLETE) to a
> multi-backend architecture. User decision (2026-07-09): **all four backends**
> (WanVideo / CogVideo / LTX 2.3 / legacy LTX-Video) + **single engine + backend registry**.
> Philosophy: mlx/mlx-lm/mlx-vlm gaps → upstream issue+PR+code; everything else (mlx-video
> backends, CogVideo, legacy LTX-Video) → reference-implement directly in fusion-mlx.

## Verified current state (re-checked 2026-07-09)

- `VideoGenEngine` (`fusion_mlx/engines/video.py`) is **hardcoded** to
  `mlx_video.models.ltx_2.generate.generate_video` (signature
  `generate_video(model_repo, text_encoder_repo, prompt, pipeline=..., ...)`).
- Installed `mlx-video` 0.0.1 (github Blaizzy/mlx-video) ships exactly **two** backends:
  - `ltx_2` — covers **LTX-2 AND LTX-2.3** (config.py:141 "LTX-2.3: prompt-conditioned
    adaptive layer norm"); `PipelineType` = DISTILLED/DEV/DEV_TWO_STAGE/DEV_TWO_STAGE_HQ.
  - `wan_2` — WanVideo, T2V + I2V (`generate_video(model_dir, prompt, image=..., steps=..., guide_scale=..., scheduler=..., ...)`). **E2E-validated** in `scripts/wan_convert.py` + `scripts/wan_e2e_smoke.py`, **NOT wired** into the engine.
- **CogVideo**: no MLX implementation exists anywhere (web search empty). Direct port required.
- **Legacy LTX-Video (0.9.x, Lightricks/LTX-Video)**: NOT in mlx-video (only the `ltx_2` line). Direct port required; mlx-video `ltx_2/` MLX code (vae/transformer/rope/samplers) is the closest reference (same family).
- `engine_type` is a string dispatch chain (`engine_pool.py:1507-1552`); `"video_gen"` already exists. `"video"` executor = `max_workers=1` (`engine_core.py:59`).
- `ImageGenEngine` delegates to external `mflux` → **no reusable MLX VAE/DiT/scheduler primitives inside fusion-mlx**. `diffusers` NOT installed (architecture reference only); `transformers 5.0.0` available (T5 loadable).
- Feature 2 (MLLM continuous batching): `fusion_mlx/pool/priority_scheduler.py` — `PriorityScheduler` with REALTIME/BATCH, preemption, chunked prefill (`prefill_chunk_size=512`). Baseline works.
- Feature 3 (vision cache): `fusion_mlx/cache/vision_feature_cache.py` `VisionFeatureSSDCache` keyed by `(model_name, image_hash)`; `vlm.py:580` hashes video frames individually → frames auto-benefit. Baseline works.

## Architecture (single engine + backend registry)

New package `fusion_mlx/engines/video_backends/`:

```
video_backends/
  __init__.py            # BACKENDS registry + resolve_backend()
  base.py                # VideoBackend ABC, VideoGenParams, VideoConstraints
  ltx2.py                # LTX2Backend  (wraps mlx_video ltx_2)  [refactor of existing]
  wan2.py                # Wan2Backend  (wraps mlx_video wan_2)  [NEW]
  ltx_video_legacy/      # direct MLX port of LTX-Video 0.9.x   [NEW, large]
  cogvideo/              # direct MLX port of CogVideoX          [NEW, largest]
```

`VideoBackend` contract (no docstrings per repo rule; logged):
- class attr `name: str`, `supports_i2v: bool`
- `@classmethod detect(cls, model_path: str) -> bool` — claim this model dir?
- `start(self, model_path, **kwargs) -> None` — load/resolve (run on io executor)
- `stop(self) -> None` — free weights, mx.clear_cache
- `generate(self, params: VideoGenParams) -> list[bytes]` — produce mp4 bytes (run on video executor, 600s)
- `constraints(self) -> VideoConstraints` — `num_frames_rule` (callable/str), `dim_divisibility`, `max_n`, `supports_i2v`

`VideoGenEngine` becomes a thin delegator:
- `__init__(model_name, **kwargs)`: resolve `self._backend` from `kwargs.get("backend")` or auto-detect via `resolve_backend(model_name)` (first `detect()` True wins; order = LTX2, Wan2, CogVideo, LTXVideoLegacy). Log which backend was chosen.
- `start()/stop()/get_stats()` delegate.
- `generate(...)`: build `VideoGenParams`, validate against `self._backend.constraints()` (raise → 422 at API), call `self._backend.generate()`. Keeps the activity-tracking + executor + tempfile + seed-multiplication logic currently in `video.py`.

## Phasing (each phase independently shippable + verified; checkpoint after each)

### Phase 0 — Registry + refactor LTX-2 into LTX2Backend (no behavior change)
- Add `video_backends/` package + `base.py` + `ltx2.py` (move existing `_generate_one`/`_resolve` logic verbatim into `LTX2Backend`).
- `VideoGenEngine` delegates to `LTX2Backend`. Existing API/tests unchanged.
- Verify: existing `test_video_gen_engine.py` + `test_videos_routes.py` green (mock mlx_video ltx_2). No behavior change.

### Phase 1 — Wan2Backend + API extension (backend/pipeline/I2V/params)
- `wan2.py`: `Wan2Backend` wraps `mlx_video.models.wan_2.generate.generate_video`. `start()` verifies local converted dir exists (NO `get_model_path` — Wan uses a local dir, not an HF repo). `generate()` maps `VideoGenParams` → wan signature (`steps`, `guide_scale`, `scheduler`, `shift`, `image` for I2V). Constraints: `num_frames = 4n+1`, dims via wan `_best_output_size`.
- `videos_routes.py`: extend `VideoGenerateRequest` with optional `backend`, `pipeline`, `image` (I2V: URL/path or b64), `negative_prompt`, `num_inference_steps`, `cfg_scale`, `guide_scale`. **Constraint validation becomes backend-aware** (move the hardcoded LTX `_validate_ltx_constraints` into `LTX2Backend.constraints()`; each backend owns its rules). Backward compat: bare LTX-2 request still works.
- Discovery: extend backend auto-detect — config.json `architectures`/`model_type` markers (CogVideoXTransformer3DModel → cogvideo; Wan config → wan2; LTX version field → ltx2 vs legacy). `backend` API param + admin override = escape hatch (mirrors #454 touchpoint 9).
- Tests: extend `test_video_gen_engine.py` with wan_2 stub + multi-backend resolution; `test_videos_routes.py` per-backend constraints + I2V path. New `test_video_backends.py`.
- Docs: `README.md` video section + `docs/plans/video-generation-multibackend.md` (this file) updated.
- Verify: mocked tests green; **deferred real-E2E** (network): `scripts/wan_convert.py` → `POST /v1/videos/generate {"backend":"wan2",...}` returns mp4 (mirrors #454 deferred criterion 4).

### Phase 2 — Verify + strengthen features 2 & 3 (light)
- Feature 2: confirm `PriorityScheduler` chunked prefill engages for VLM video prefill (long vision-token sequences) without OOM/regression. Add regression test if missing (long video → chunked, no crash). If an mlx-vlm video-processing gap surfaces → **upstream issue+PR** per philosophy.
- Feature 3: confirm `VisionFeatureSSDCache` hit-rate for repeated video frames; add optional video-level composite key (`model:video_hash:frame_idx`) for cross-request within-video dedup. Measure hit/miss before/after.

### Phase 3 — LTXVideoLegacyBackend (direct MLX, reuse ltx_2 blocks)
- Reuse mlx-video `ltx_2/` MLX building blocks (vae, transformer, rope, samplers, attention) as the reference; adapt config loader + sigma schedule for LTX-Video 0.9.x weights.
- Sub-steps (checkpoint each): (a) config + weight loader, (b) VAE decode, (c) transformer forward, (d) scheduler loop, (e) T2V E2E, (f) I2V.
- If an MLX primitive is missing (e.g. a conv/attention variant) → **upstream mlx issue+PR** per philosophy.
- Tests: mocked unit tests (stub the ported MLX modules); real-E2E deferred (needs legacy LTX-Video weights).

### Phase 4 — CogVideoBackend (direct MLX, largest)
- Port from diffusers `CogVideoXPipeline` architecture (reference only — diffusers not a runtime dep): T5 text encoder (via `transformers`), CogVideoX 3D VAE (encode/decode), CogVideoX DiT transformer (3D RoPE, spatial-temporal attention), CogVideoX flow-matching scheduler.
- Sub-steps (checkpoint each): (a) T5 embeddings, (b) 3D VAE decode, (c) DiT forward, (d) flow scheduler loop, (e) T2V E2E, (f) I2V.
- Tests: mocked unit tests; real-E2E deferred (needs CogVideoX-2B/5B weights).
- This phase is a sub-project; estimate honestly as the largest single item.

## Touchpoints (verified line numbers, 2026-07-09)

| File | Change |
|---|---|
| `fusion_mlx/engines/video_backends/*` | NEW package (base, ltx2, wan2, ltx_video_legacy/, cogvideo/) |
| `fusion_mlx/engines/video.py` | Refactor `VideoGenEngine` → thin delegator over `self._backend` |
| `fusion_mlx/engines/__init__.py` | (unchanged — `VideoGenEngine` still exported) |
| `fusion_mlx/api/videos_routes.py` | Extend request (backend/pipeline/image/negative_prompt/steps/cfg/guide_scale); backend-aware constraints; I2V path |
| `fusion_mlx/pool/engine_pool.py:1539-1540` | `VideoGenEngine(model_name=entry.model_path)` — pass `backend` kwarg if discovery recorded one (add `backend` to `EngineEntry` optional, or infer from model_path at resolve time) |
| `fusion_mlx/pool/model_discovery.py:946-953` | Extend `_is_video_model`/detect to distinguish the 4 families (architectures/markers) |
| `pyproject.toml` | `[video]` optional group already has mlx-video; no new dep (direct ports use mlx+transformers only) |
| `README.md` | Video section: 4 backends, I2V, backend selection |

## Tests
- `tests/unit/test_video_gen_engine.py` — extend: wan_2 stub + backend auto-detect + explicit `backend=` override + per-backend constraint validation + I2V param passthrough.
- `tests/unit/test_videos_routes.py` — per-backend 422 rules, I2V, backend override.
- `tests/unit/test_video_backends.py` (NEW) — registry resolution order, `detect()` claims.
- `tests/unit/test_video_discovery.py` — 4-way family detection.
- Phase 2: continuous-batching video-prefill regression; vision-cache video-frame hit test.
- Phases 3-4: mocked unit tests per ported module; real-E2E deferred (weights).

## Verification (success criteria)
1. `black --check` + `ruff check` clean on all touched files.
2. Phase 0-1 mocked unit tests green; no regression in existing video/vlm/cache tests.
3. Phase 2: video-prefill continuous-batching test green; vision-cache hit-rate measured.
4. Phases 3-4: per-module mocked tests green; real-E2E deferred (network + weights), mirroring #454.
5. Each phase: checkpoint commit; macOS app repackage reminder when `fusion_mlx/` Python lands on main (standing constraint).

## Upstream policy (per user philosophy)
- mlx-video / CogVideo / legacy-LTX-Video → **direct in fusion-mlx** (`engines/video_backends/`), NOT upstream PRs.
- Any mlx / mlx-lm / mlx-vlm gap hit during features 2/3 or the direct ports → **file issue + PR + code upstream** (the user's prescribed flow), then carry the code.

## Risks
- **CogVideo + legacy-LTX ports are large** (weeks). Phased with checkpoints; each sub-step verified before next (Rule 10). Mocked tests gate each step; real-E2E deferred like #454.
- **Backend auto-detect ambiguity** (LTX-2 vs 2.3 vs legacy) → `backend` API param + admin override escape hatch.
- **Memory**: video gen is heavy; `engine_pool` memory admission already guards load; fails loudly (Rule 12).
- **mlx-video API drift**: isolated to one method per backend (`generate`), localized fix.
