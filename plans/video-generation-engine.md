# Video Generation Engine (#454) — VERIFIED PLAN

> Status: COMPLETE - implemented + mocked-tested (49 unit tests green, black+ruff clean). Branch: `feat/video-gen-engine` (from `main` @ `54cf8cbd`). mlx-video API now RUNTIME-VERIFIED (package installed + `inspect.signature`/`getsource` introspected 2026-07-07).
> Predecessors (unmerged, do NOT merge here): #431 spec-routing (`c96b65e4`), #453 video hardening (`6786fa89`), #451 mllm cleanup (`5fc1d015`).

## ✅ Environment constraint (RESOLVED 2026-07-07)

`github.com` is now reachable. `pip install git+https://github.com/Blaizzy/mlx-video.git` succeeded (package version 0.0.1, Home-page `https://github.com/Blaizzy/mlx-video`). The real API was introspected at runtime:

- **Top-level**: `mlx_video.get_model_path(repo)` (downloads/returns local HF cache path). `LTXModel`, `WanModel`, `AudioDecoder`, `models` present. There is NO top-level `load` or `generate`.
- **Generation entry**: `mlx_video.models.ltx_2.generate.generate_video(model_repo, text_encoder_repo, prompt, pipeline=PipelineType.DISTILLED, negative_prompt=..., height=512, width=512, num_frames=33, num_inference_steps=40, cfg_scale=4.0, seed=42, fps=24, output_path='output.mp4', verbose=True, ...)`.
- `generate_video` loads weights internally EVERY call via `LTXModel.from_pretrained` (no module-level cache), so `start()` front-loads only the network download via `get_model_path`; generation reads weights from the local HF cache.
- `PipelineType` enum: `DISTILLED`, `DEV`, `DEV_TWO_STAGE`, `DEV_TWO_STAGE_HQ`.
- LTX-2 constraints: `num_frames` must be `1 + 8*k` (silently adjusted otherwise); `height`/`width` divisible by 64 (distilled/two-stage) or 32 (dev) - enforced via `assert`.
- ⚠️ **PyPI `mlx-video` 0.1.0 is a DIFFERENT video-IO library** (only `load`/`normalize`/`resize`/`to_float`, reads video files to arrays, NO `generate`). `pip install mlx-video` installs the WRONG package. Use `pip install git+https://github.com/Blaizzy/mlx-video.git`.

Unit tests mock `mlx_video` via `sys.modules` (stub `get_model_path` + fake `mlx_video.models.ltx_2.generate` module with `PipelineType`/`generate_video`). Tests verify wiring + the verified API contract, NOT real generation.

Manual E2E (verification criterion 4) still DEFERRED - requires downloading an LTX-2 model (~19B), a long network transfer not performed here.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model scope | LTX-2 only | mlx-video flagship. Mirrors `ImageGenEngine` wrapping only Flux. One engine class, one load path. |
| Feature scope | T2V only (first cut) | Mirrors `images.py` first cut (t2i only). Rule 2. I2V/A2V deferred. |
| Dependency | optional dep group `[video]` | `mlx-video @ git+...` under new optional group; `start()` lazy-imports with friendly ImportError (mirrors `image_gen.py:53-60`). |
| Detection | `_is_video_model` mirrors `_is_image_model` | `configuration.json` `task=="text-to-video"` (mlx-video ships this manifest, same convention as mflux Flux `task=="text-to-image"`). No separate architecture sets (Rule 2). Admin override (touchpoint 9) = escape hatch if auto-detect misses. |
| Surgical scope | add only `video`/`video_gen` | Do NOT fix pre-existing gaps (image_gen missing from `EngineEntry.engine` union L99-107 + `get_engine` return L733-740; `image` option missing from admin HTML L309-316). Rule 3. |

## 9 touchpoints (all confirmed against current code, line numbers verified 2026-07-07)

### 1. `pyproject.toml` — optional dependency
Add to `[project.optional-dependencies]` (alongside `grammar`/`mcp`/`modelscope`):
```toml
video = ["mlx-video @ git+https://github.com/Blaizzy/mlx-video.git"]
```

### 2. `fusion_mlx/pool/model_discovery.py` (verified lines)
- `ModelType` Literal (L28-37): add `"video"` after `"image"`.
- `EngineType` Literal (L38-47): add `"video_gen"` after `"image_gen"`.
- NEW `_is_video_model(path)` (after `_is_image_model` L923-939): mirror exactly, `data.get("task") == "text-to-video"`.
- `_is_model_dir` (L942-946): add `_is_video_model(path)` to the `has_image_manifest or _is_video_model(path)` condition so video dirs register.
- `detect_model_type` (L555-556): add `if _is_video_model(model_path): return "video"` right after the `_is_image_model` early-return.
- `_register_model` (L1150-1153): add `elif model_type == "video": engine_type = "video_gen"` before the `else: "batched"`.

### 3. `fusion_mlx/engines/video.py` (NEW) — `VideoGenEngine(BaseNonStreamingEngine)`
Mirror `image_gen.py:35-154` line-for-line in shape:
- `__init__(model_name, **kwargs)`: `self._model_name`/`_model_path`, `self._model = None`, `_kwargs`.
- `start()`: lazy `from mlx_video import get_model_path` inside `_resolve` run in `get_executor("io")` 180s timeout; friendly `ImportError` (github install URL, NOT PyPI) if missing. `get_model_path` front-loads the network download; `generate_video` reloads weights every call so no model object is held. **ISOLATE the mlx-video resolve into `_resolve()` - single point.** Log model_name.
- `stop()`: null `self._model`, `gc.collect()`, `mx.synchronize()` + `mx.clear_cache()` on `get_executor("video")` (mirror L81-92).
- `generate(prompt, num_frames=97, width=768, height=512, fps=24, seed=None, n=1, **kwargs) -> list[bytes]`: run in `get_executor("video")` 600s timeout; call mlx-video `generate_video(...)` writing to a `managed_tempfile_path(prefix="fusion_video_", suffix=".mp4")` mp4, read bytes, return `list[bytes]` (one mp4 per `n`; temp auto-unlinked by the context manager). `_begin_activity`/`_update_activity`/`_finish_activity`. Log prompt_len/num_frames/elapsed. **ISOLATE the mlx-video generate call into `_generate_one()` - single point.**
- `get_stats()`: `{"model_name": ..., "loaded": self._model is not None}`.

> **mlx-video call (VERIFIED 2026-07-07)**: `from mlx_video.models.ltx_2.generate import PipelineType, generate_video` + `from mlx_video import get_model_path`. `start()` calls `get_model_path(model_name)` (download only); `_generate_one()` calls `generate_video(model_repo, text_encoder_repo, prompt, pipeline=PipelineType(...), height=, width=, num_frames=, seed=, fps=, output_path=<temp>, verbose=False)`. `seed` defaults to `random.randint(0, 2**31-1)` (NOT 0) when None; `stop()` routes mx cleanup to the 2-worker `io` executor (not the contended 1-worker `video` executor) to avoid a 5s-timeout deadlock.

### 4. `fusion_mlx/engines/__init__.py` (verified L8-30)
Add `from .video import VideoGenEngine` + `"VideoGenEngine"` to `__all__`.

### 5. `fusion_mlx/engine_core.py` (verified L56-61)
`_executor_config`: add `"video": {"max_workers": 1, "prefix": "mlx-video"}`.

### 6. `fusion_mlx/pool/engine_pool.py` (verified lines)
- import (L35 area): `from ..engines.video import VideoGenEngine`.
- `EngineEntry.model_type` Literal (L62-71): add `"video"`.
- `EngineEntry.engine_type` Literal (L72-82): add `"video_gen"`.
- `EngineEntry.engine` union (L99-107): add `| VideoGenEngine`. (Leave pre-existing image_gen omission — Rule 3.)
- `_MODEL_TYPE_TO_ENGINE` (L430-439): add `"video": "video_gen"`.
- `_load_engine` factory (L1532-1533): add `elif entry.engine_type == "video_gen": engine = VideoGenEngine(model_name=entry.model_path)` after the image_gen branch.

### 7. `fusion_mlx/api/videos_routes.py` (NEW) — mirror `api/images.py` exactly
- `router = APIRouter(prefix="/v1/videos", tags=["videos"])`; `_pool` + `set_videos_context(pool)`.
- `VideoGenerateRequest`: `prompt`, `n=1 (ge1 le4)`, `num_frames=97`, `width=768`, `height=512`, `fps=24`, `seed=None`, `model=None`, `response_format="url" pattern="^(url|b64_json)$"`.
- `VideoOutput` (url, b64_json); `VideoGenerateResponse` (data, created).
- `POST /generate`: `_pool.get_engine(model_name)` → `isinstance(engine, VideoGenEngine)` (404 on mismatch) → `engine.generate(...)` → per mp4 bytes: `b64_json` or `url=f"data:video/mp4;base64,{b64}"`.

### 8. `fusion_mlx/server.py` (verified lines)
- import (L32-33 area): `from .api.videos_routes import router as videos_router, set_videos_context`.
- include (L483 area, after `images_router`): `app.include_router(videos_router)`.
- set_context (L733 area, after `set_images_context`): `set_videos_context(self.pool)`.

### 9. Admin override (escape hatch when auto-detect misses)
- `fusion_mlx/admin/models_route.py`: add `"video"` to `valid_types` (L265-274) + `"video": "video_gen"` to `type_to_engine` (L284-293).
- `fusion_mlx/admin/templates/dashboard/_modal_model_settings.html` (L316, after audio_sts): add `<option value="video">Video</option>`. (Leave pre-existing missing `image` option — Rule 3.)

## Tests (mirror existing engine/route patterns; all mock mlx_video)

- `tests/unit/test_video_gen_engine.py` - `sys.modules["mlx_video"]` patch (stub `get_model_path` + fake `mlx_video.models.ltx_2.generate` module with `PipelineType`/`generate_video`):
  - `start()` raises friendly ImportError when mlx_video absent (`sys.modules["mlx_video"]=None`) - message includes github install hint.
  - `start()` resolves via fake `get_model_path` in io executor; idempotent; `stop()` clears loaded + syncs on io executor; `get_stats()` reflects loaded state.
  - `generate()` calls fake `generate_video` (writes fake mp4 bytes to `output_path`), returns `list[bytes]`; `seed=None` -> random non-zero distinct seeds; `seed=42,n=3` -> seeds 42/43/44; temp auto-unlinked; pipeline/text_encoder_repo passed through; generate before start raises RuntimeError.
- `tests/unit/test_videos_routes.py` - `POST /v1/videos/generate` happy path (mock engine returns mp4 bytes -> b64_json + data URL), 404 on non-VideoGenEngine / `ModelNotFoundError`, 503 when pool uninitialized, 500 on engine raise, 422 on LTX constraint violations (`num_frames` not `1+8*k`, `width`/`height` not divisible by 64, n>4, bad response_format). Strict mock `generate` side_effect rejects unexpected kwargs (#12).
- `tests/unit/test_video_discovery.py` - `_is_video_model` detects `configuration.json` `task=="text-to-video"` AND requires a diffusers subdir (vae/transformer/audio_vae); a stray text-to-video manifest without subdirs returns False / falls through to llm (#5/#7). Maps to `engine_type="video_gen"` via `_register_model`.

## Verification (success criteria — what "done" looks like)

1. `black --check` + `ruff check` clean on all touched files.
2. `pytest tests/unit/test_video_gen_engine.py tests/unit/test_videos_routes.py` green (mocked).
3. `pytest tests/unit/test_video_utils.py tests/unit/test_model_discovery*.py` still green (no regression).
4. **DEFERRED** (network): `pip install -e .[video]` + LTX-2 model → `POST /v1/videos/generate {"prompt":"...","model":"<ltx-2>"}` returns mp4 b64_json. Confirm/fix the 2 isolated mlx-video calls.

## Non-goals (deferred)

- I2V / A2V; Wan2.1 / Wan2.2; video upscale; streaming progress.

## Risks

- **mlx-video API drift** (HIGHEST): the 2 isolated calls (`_load`, `_generate_one`) are researched-not-verified. Fix is localized when network returns.
- **git+ dep needs network**: optional group keeps it out of core install; `start()` gives install hint.
- **LTX-2 19B memory**: `engine_pool` memory admission already guards load; fails loudly on low-RAM (Rule 12).
- **macOS app repackage**: when this lands on `main` (touches `fusion_mlx/` Python), `.app`/`.dmg` must be rebuilt (standing constraint).

## Phase 2: Continuous Batching & Vision Cache Verification (2026-07-09)

The three features are layered: (1) video generation engine (Phase 0/1 above),
(2) MLLM continuous batching scheduler, (3) vision feature cache for video
frames. Phases 2 and 3 reuse existing fusion-mlx subsystems; this section
records the verification performed and the debt surfaced.

### Feature 2 — MLLM continuous batching scheduler: VERIFIED (existing)

The real continuous-batching scheduler lives in `fusion_mlx/scheduler/`
(`Scheduler` class, instantiated at `engine_core.py:183`). It is true
iteration-level batching via mlx-lm `BatchGenerator` with real chunked prefill
(`sched_batch.py:506`). VLM and video requests flow through this same
scheduler; they are not special-cased at the scheduling layer.

- **VLM/text batch isolation** (`sched_schedule.py:201-213`): the homogeneity
  guard classifies each request via `request_is_vlm = request.vlm_inputs_embeds
  is not None`. VLM and text-only requests use different prefill paths
  (embeddings vs token IDs), so a status mismatch defers the later request to
  the next batch (`self.waiting.appendleft(request); break`). Multiple VLM
  requests batch together; VLM+text mixes do not. Verified by inspection —
  driving `_schedule_waiting` end-to-end was ruled out: the function is 746
  lines with 3 conditional `scheduled.append` branches (VLM-MTP / normal /
  paged) behind a large SpecPrefill block and 6+ model-touching downstream
  methods; mocking all of it would pass for the wrong reasons (Rule 9).
- **`fusion_mlx/pool/priority_scheduler.py` is DEAD CODE.** `PriorityScheduler`
  is not wired into production (referenced only in `pool/__init__.py` export
  and `tests/unit/test_pool.py`). Its `step()` drains priority queues into
  `base.add_request()` then calls `base.step()` once — no continuous batching.
  Its `_check_chunked_prefill` (line 430) is a no-op stub. Not the scheduler
  the feature refers to; candidate for removal.

**Tests added** (`tests/unit/test_continuous_batching.py::TestVLMContinuousBatching`):
`test_scheduler_accepts_vlm_requests`, `test_vlm_and_text_requests_queue_together`
— VLM requests carrying `vlm_inputs_embeds`/`vlm_image_hash` are admitted into
the same waiting queue as text requests (the precondition for VLM batching).

### Feature 3 — Vision feature cache for video frames: VERIFIED (existing)

`VisionFeatureSSDCache` (`fusion_mlx/cache/vision_feature_cache.py`) is a
two-tier cache (memory LRU + SSD safetensors) keyed by
`_composite_key(model_name, image_hash)`. 19 unit tests pass.

- **Non-native video path benefits** (`engines/vlm.py:577-608`):
  `_prepare_vision_inputs` calls `compute_per_image_hashes(images)` and looks
  up each frame in the cache; on an all-hit it concatenates cached features
  and skips the vision encoder, on a miss it computes + `_split_vision_features`
  + `put` per-image. Video frames on the non-native path are extracted as PIL
  images (`vlm.py:949-983`, Path B) and flow into this same branch, so repeated
  frames across turns reuse cached features.
- **Native video models deliberately skip the cache** (`vlm.py:828-830`):
  ndarray frames have no stable PIL hash, so Qwen-style native video
  (`_is_native_video_model`, `vlm.py:925-936`, Path A) bypasses the cache by
  design. This is intended behavior, not a gap.

**Tests added** (`tests/unit/test_vision_feature_cache.py::TestVisionCacheEngineWiring`):
`test_repeat_image_hits_cache_and_skips_encoder` (second call with the same
image hits the cache — `_compute_vision_features` called once, `stats.hits >= 1`),
`test_cache_disabled_does_not_consult_cache` (disabled → model's
`get_input_embeddings` runs each call, cache untouched: 0 hits / 0 misses),
`test_different_image_misses_again` (different image → miss, encoder runs
again). Mocks are at clean seams only (mlx_vlm `prepare_inputs`,
`_compute_vision_features`, `_split_vision_features`, `_vlm_model`); the cache
and image-hash functions are real.

### Debt surfaced (pre-existing, NOT introduced by this work)

- **11 `TestVLMEngineIntegration` tests fail** (`test_vision_feature_cache.py:266+`):
  they import `from fusion_mlx.engine.vlm import VLMBatchedEngine` (singular
  `engine` — a stub/shim) instead of the real `fusion_mlx.engines.vlm`
  (plural), and assert a stale API contract (`encode_image(pixel_values,
  image_position_ids=...)`, `_image_token_count`) the real class no longer has.
  Unrelated to the three features; left as debt (Rule 3/6). Fix = correct the
  import path + update assertions to the real `_compute_vision_features`
  strategy API.
- **`PriorityScheduler` dead code** (see Feature 2): candidate for deletion.

## Phase 3 & 4: Legacy LTX-Video + CogVideoX stubs (2026-07-09)

**Decision (Option B): stub + upstream issue.** mlx-video (Blaizzy/mlx-video)
ships only `ltx_2` (covers LTX-2 and LTX-2.3) and `wan_2` (Wan2.1 / Wan2.2).
Web + GitHub-API search confirmed **no MLX port of legacy LTX-Video (0.9.x) or
CogVideoX exists** anywhere. A from-scratch port of either would be multi-kLoC,
unverifiable without multi-GB weights and GPU-class compute, and would violate
Rule 6/9/12. Per the user's flow ("遇到上游问题，先提issue，再提pr，跟着提交落地code"),
the correct move is: register a stub that raises a clear `NotImplementedError`
pointing at the upstream issue tracker and naming a real shipped alternative,
then file the upstream issues. The stubs let `resolve_backend` route the
families correctly today instead of mis-routing them to LTX2.

**Implementation** (`fusion_mlx/engines/video_backends/unimplemented.py`):
`UnimplementedBackend(VideoBackend)` base stores `_model_name`/`_loaded` in
`__init__` (so `resolve_backend`'s `cls(model_name, **kwargs)` constructs it),
`start()`/`generate()` raise `NotImplementedError` with the message
`"{family} has no MLX port and is not shipped by mlx-video; ... Request/track
upstream support: https://github.com/Blaizzy/mlx-video/issues. Use ltx2 (LTX-2 /
LTX-2.3) or wan2 (Wan2.1 / Wan2.2) instead."`, `stop()` is a no-op, and
`constraints()` returns permissive `VideoConstraints(supports_i2v=True, max_n=4,
dim_divisibility=1, num_frames_validator=None)` so `validate_params` accepts the
request and the stub - not the validator - delivers the clear error.

- `LegacyLTXBackend` (`name="ltx_video_legacy"`): `detect` matches `ltx-video` /
  `ltx_video` substrings (incl. `Lightricks/LTX-Video`).
- `CogVideoBackend` (`name="cogvideo"`): `detect` matches `cogvideo` /
  `cog_video` substrings (incl. `THUDM/CogVideoX-2b`).

**Registry** (`video_backends/__init__.py`): BACKENDS now has 4 entries
(`ltx2`, `wan2`, `ltx_video_legacy`, `cogvideo`); `_ALIASES` adds `ltx-video` /
`ltx_video` -> `ltx_video_legacy` and `cogvideox` / `cog_video` / `cogvideo-x`
-> `cogvideo`. **Critical correctness property**: `ltx-2` / `ltx-2.3` still
resolve to the real `LTX2Backend` (registered first, detected first) - the
legacy stub does NOT shadow the modern shipped backend. Unknown model names
still fall back to `LTX2Backend` (Phase 0 behavior preserved).

**Bug found + fixed during verification**: the first `resolve_backend` call on a
stub raised `TypeError: LegacyLTXBackend() takes no arguments` because
`UnimplementedBackend` inherited `object.__init__`. Added the `__init__` above.

**Tests added** (`tests/unit/test_video_backends.py`):
- `test_legacy_ltx_autodetect`, `test_cogvideo_autodetect` (name + repo-id
  detection).
- `test_legacy_does_not_shadow_modern_ltx` (ltx-2/ltx-2.3 -> LTX2, not the
  stub) - guards the critical property above.
- `test_explicit_legacy_and_cogvideo_aliases` (explicit hints resolve).
- `TestUnimplementedBackends` (parametrized over both): permissive constraints,
  `validate_params` accepts then backend raises, `stop()` no-op, `start()`
  raises with upstream URL + a real alternative name, `generate()` raises.
- `test_backends_registry_has_both` -> `test_backends_registry_has_all` (asserts
  all 4 keys; the old 2-key assertion was stale).
- Full file: 36 passed; lint clean (`black` + `ruff`).

**Deferred**: filing the 2 upstream mlx-video issues (legacy LTX-Video +
CogVideoX MLX port requests) was held - it commits the user's identity to a
public third-party tracker and the Phase 3/4 direction was confirmed while the
user was AFK. Re-raise with the user before filing.
