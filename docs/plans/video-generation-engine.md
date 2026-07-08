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
