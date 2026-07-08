# Video Generation Engine (#454) — VERIFIED PLAN

> Status: COMPLETE - implemented + mocked-tested (38 unit tests green, black+ruff clean). Branch: `feat/video-gen-engine` (from `main` @ `54cf8cbd`).
> Predecessors (unmerged, do NOT merge here): #431 spec-routing (`c96b65e4`), #453 video hardening (`6786fa89`), #451 mllm cleanup (`5fc1d015`).

## ⚠️ Environment constraint (Rule 12 — fail visibly)

`github.com` is UNREACHABLE in this env (HTTP2 framing error + port 443 timeout 75s). Cannot `pip install mlx-video` or clone the repo to inspect its API, nor download an LTX-2 model. Implications:

- **mlx-video API is researched, NOT runtime-verified.** Engine internals (touchpoint 3) written against prior GitHub-contents-API research. The single mlx-video load+generate call is isolated in one method so a fix is ~5 lines once network returns.
- **Unit tests mock `mlx_video`** via `sys.modules` patch (same pattern as `test_video_utils.py` mocks cv2). Tests are green WITHOUT the real package — but they verify the **wiring** (engine lifecycle, executor usage, activity tracking, bytes return, route dispatch, b64/data-url formatting), NOT that mlx-video accepts the args. Honest per Rule 9.
- **Manual E2E (verification criterion 4) DEFERRED** until network available.

All 9 touchpoints' STRUCTURE is confirmed against current code (post-#414/#453) — none depend on mlx-video internals. Deliverable = full structural integration + mocked-tested engine + clearly-marked single integration point.

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
- `start()`: lazy `from mlx_video import LTXModel, LTXModelConfig` (or `load`) inside `_load` run in `get_executor("io")` 180s timeout; friendly `ImportError` if missing (mirror L53-60). **ISOLATE the mlx-video load into `_load()` — single point needing runtime confirmation.** Log model_path.
- `stop()`: null `self._model`, `gc.collect()`, `mx.synchronize()` + `mx.clear_cache()` on `get_executor("video")` (mirror L81-92).
- `generate(prompt, num_frames=97, width=768, height=512, fps=24, seed=None, **kwargs) -> list[bytes]`: run in `get_executor("video")` 600s timeout; call mlx-video `generate(...)` writing to `tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)`, read bytes, unlink temp, return `list[bytes]` (one mp4 per `n`). `_begin_activity`/`_update_activity`/`_finish_activity` (mirror L112-147). Log prompt_len/num_frames/elapsed. **ISOLATE the mlx-video generate call into `_generate_one()` — single point.**
- `get_stats()`: `{"model_name": ..., "loaded": self._model is not None}`.

> **mlx-video call (researched, unverified)**: `from mlx_video import load, generate` → `model, processor = load(model_path)` in `start()`; `generate(model, processor, prompt, num_frames, height, width, fps, seed, output_path)` in `_generate_one()`. If this shape is wrong, only `_load()` + `_generate_one()` change.

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

- `tests/unit/test_video_gen_engine.py` — `sys.modules["mlx_video"]` patch (fake `load`/`generate`):
  - `start()` raises friendly ImportError when mlx_video absent (del from sys.modules) — message includes install hint.
  - `start()` loads via fake `load` in io executor; `stop()` nulls + syncs; `get_stats()` reflects loaded state.
  - `generate()` calls fake `generate` (writes fake mp4 bytes to temp path), returns `list[bytes]`, activity tracking begins/finishes, temp file unlinked.
- `tests/unit/test_videos_routes.py` — `POST /v1/videos/generate` happy path (mock engine returns mp4 bytes → b64_json + data URL), 404 on non-VideoGenEngine, validation errors (n>4, bad response_format).
- extend `tests/unit/test_model_discovery*.py` — `_is_video_model` detects `configuration.json` `task=="text-to-video"`; maps to `engine_type="video_gen"` via `_register_model`.

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
