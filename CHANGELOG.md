# Changelog

## [0.8.5] - 2026-08-05

Patch release.

- Accept HuggingFace repo ids in model resolution (#372): `/v1/chat/completions`
  (and all OpenAI/Anthropic routes) now accept HF repo ids like
  `mlx-community/Qwen3-0.6B-4bit` in addition to registry short names. The
  engine pool normalizes a repo-id request to the discovered entry whose
  `source_repo_id` matches (case-insensitive) before lookup, so clients using
  the OpenAI-standard `org/repo` form no longer hit an opaque 502 through the
  gateway; unmatched ids still raise a clear `ModelNotFoundError` (404) listing
  available short names.
- Fix stale `--lora-path` CLI help (#390): the help text and inline comment
  claimed "Runtime hot-swap is not yet supported", contradicting the
  per-request adapter hot-swap already implemented in `engine_pool`
  (`_adapter_key`/`_make_adapter_entry`, wired into the OpenAI/Anthropic routes
  via the request `adapters` field). Help text now reflects that per-request
  adapter hot-swap is supported.

## [0.8.4] - 2026-08-05

Patch release.

- Add Windows CUDA backend node (#365): optional vLLM-powered
  OpenAI-compatible server for heavy LLM inference (DeepSeek 70B / Qwen 72B
  FP8) on Windows CUDA hosts. New `fusion-mlx cuda-node` subcommand builds a
  FastAPI app embedding vLLM's `AsyncLLMEngine`, serving `/health`,
  `/v1/models`, `/v1/chat/completions`, `/v1/completions`. The node
  self-registers with the cluster via mDNS under `platform=windows-cuda` so a
  fusion-gateway can route heavy-model intents to it (gateway-side platform
  routing landed in v0.8.0). Platform detection (`FUSION_PLATFORM` env →
  `sys.platform` + CUDA probe → `mac`) surfaces a `platform` TXT record on
  every node's mDNS advertisement. vLLM is imported lazily so the package
  stays importable on Mac; starting the node without vLLM raises a clear
  `RuntimeError`. LLM-only scope (diffusion-on-CUDA tracked separately).
  See [docs/cuda-node.md](docs/cuda-node.md).

## [0.8.3] - 2026-08-05

Patch release.

- Add Stable Cascade image generation: native MLX port of the Würstchen
  3-stage pipeline (#370). Unified `StableCascadeUNet` serves both the
  prior (`switch_level=(False,)` → `UpDownBlock2d` 1×1 mapping, no
  spatial change) and decoder (`switch_level=None` → `Conv2d(k=2,s=2)`
  down / `ConvTranspose2d(k=2,s=2)` up), with a PaellaVQModel
  decode-only VQGAN and a CLIP-ViT-bigG text encoder. DDPM-Würstchen
  scheduler (cosine `_alpha_cumprod`, `linspace(1.0,0.0,steps+1)`
  timesteps). Wired into the `image_gen` engine (`stable_cascade`
  variant, auto-detected from `cascade`/`wuerstchen` model names) and
  the `/v1/images/generate` API. Validated end-to-end with real
  stabilityai weights (prior 1550/1550, decoder 1726/1726, VQGAN
  121/122, CLIP 517/517 keys). See [docs/cascade-image.md](docs/cascade-image.md).

## [0.8.2] - 2026-08-06

Patch release.

- Add SDXL image generation: native MLX port of
  `StableDiffusionXLPipeline` with CosXL (`cosxl_edit`) and SDXS variant
  support (#371). Dual text encoders (CLIP-L + OpenCLIP-G, cross-attn
  dim 2048), EulerDiscreteScheduler, AutoencoderKL. Wired into the
  `image_gen` engine (`sdxl`/`cosxl`/`sdxs` variants) and the
  `/v1/images/generate` API. Validated end-to-end with real
  `stabilityai/stable-diffusion-xl-base-1.0` weights.

## [0.8.1] - 2026-08-06

Patch release.

- Fix background fine-tune jobs crashing with `BrokenPipeError` when tqdm
  flushes the closed stderr pipe of the background service (#381). The
  `train()` call is now wrapped in `redirect_stderr(io.StringIO())` so
  tqdm writes to an in-memory buffer instead of the dead pipe.
- Fix 7 pre-existing full-suite test failures that blocked CI green on
  main (#380): url_safety shadowing in the security test fixtures and a
  vlm video load_video path mismatch. No behavior change to runtime code.
  Unit suite now 7937 pass / 0 fail locally; CI test (3.11) drops from a
  2h+ hang to ~15m.

## [0.8.0] - 2026-08-06

Stable Diffusion 3-Medium full MLX txt2img (#369). From-scratch MLX port
of the SD3-Medium MMDiT (24 joint transformer blocks, inner_dim=1536,
joint_attention_dim=4096, pooled_projection_dim=2048) + AutoencoderKL
VAE + FlowMatchEuler rectified-flow scheduler, in `fusion_mlx/image/sd3/`.
Text encoders reuse mflux T5-XXL + CLIP-L (CLIPEncoder) alongside a
custom parametrized CLIPTextModel for CLIP-G (20 heads). Wired into the
image engine: `/v1/images/generate` with `model="sd3-medium"` auto-detects
the sd3 variant, supports `negative_prompt` and `shift` (unlike Flux).
fp8 T5 (`t5xxl_fp8_e4m3fn.safetensors`, decoded via torch) is preferred
over sharded fp16 with automatic fallback. `SD3_LOCAL_DIR` +
`SD3_*_SUBFOLDER`/`_FILE` env vars allow offline/local weight resolution.
Real-model E2E validated (CLIP-L+CLIP-G+T5+MMDiT+VAE → PIL). 28 unit
tests in `test_image_gen_sd3.py`.

## [0.7.11] - 2026-08-05

Video DiT diffusion throughput optimization (#367). HunyuanVideo and
Cosmos diffusion loops ran two full-latent DiT forwards per step
(uncond + cond, standard CFG) — at production frame counts (57-121
frames) every video DiT workflow (2B/7B/13B) exceeded the 1800s budget
before reaching VAEDecode. This release fuses the uncond+cond pair into
a single batched B=2 forward (both DiTs are batch-safe along dim 0),
halving the per-step forward count for ~2x throughput with no quality
change. A single-forward shortcut skips the uncond branch entirely when
`cfg_scale <= 1.0` (useful for guidance-distilled / low-cfg workflows).
Step-level it/s INFO logging replaces the previous debug-only line so
hangs vs. slow steps are diagnosable and ComfyUI can report progress.

### Performance
- HunyuanVideo (`fusion_mlx/video/hunyuanvideo/generate.py`): CFG
  batched guidance (B=2 fused forward), `cfg<=1.0` single-forward
  shortcut, step-level it/s INFO logging.
- Cosmos (`fusion_mlx/video/cosmos/generate.py`): same batched CFG,
  single-forward shortcut, and it/s logging. Masks broadcast to B=2.
- Wan2/VACE already used batched CFG — unchanged.

### Notes
- Verified on real models (HunyuanVideo 13B DiT + Cosmos 7B DiT,
  3-step diffusion): batched vs. two-pass converge with no NaN; relative
  latent diff <1% (Hunyuan 0.86%, Cosmos 0.25%). The residual is MLX
  fp32 kernel reordering across batch sizes — within CFG guidance
  tolerance, does not accumulate across steps.

### Fixed
- `tests/unit/test_gen_acceleration_knobs.py`: update `TestImageGenKnobFlow`
  mocks for the FLUX.1 path (`mflux.models.flux.cli.flux_generate.Flux1`,
  `ModelConfig.schnell()/dev()`) introduced by #368/#375; 3 pre-existing
  failures resolved.

## [0.7.10] - 2026-08-05

Remove the dead engine-method guided-decoding branch on `/v1/responses`
(#373). The route read `engine.supports_guided_generation` and called
`engine.generate_with_schema(...)`, but no engine class defines either
symbol — the branch was dead code and a latent `AttributeError`. The
live constrained path on `/v1/responses` is the R12-4 post-generate
validation branch (`use_strict_postgen_validation`), which is buffered-
only by design (the Responses surface has no guided-streaming SSE
helper).

### Fixed
- **#373 - Dead guided-decoding branch on `/v1/responses`.** Removed the
  `use_guided` / `engine.generate_with_schema` /
  `engine.supports_guided_generation` references from
  `fusion_mlx/routes_internal/responses.py` (`_resolve_strict_context`
  and the `_non_stream` dispatch). Strict `json_schema` requests now go
  straight to the R12-4 post-generate validation + single-repair-retry
  path (the only live constrained path on this surface). Live constrained
  decoding for the chat surface runs through the grammar-compiler
  (xgrammar/llguidance) path in `openai_routes.py`, unchanged. The unused
  `validate_output_against_schema` import was dropped.

### Removed
- `tests/unit/test_responses_chat_template_kwargs.py`: deleted the
  `TestBatchedEngineGuidedHonorsEnableThinking` class (3 tests pinning
  the removed `BatchedEngine.generate_with_schema` +
  `shared_apply_chat_template` contract, previously `@pytest.mark.skip`)
  and `test_strict_via_guided_path_also_auto_disables` (asserted the
  removed `engine.generate_with_schema` call). Simplified the `_Engine`
  mock (dropped `supports_guided` / `generate_with_schema` / `guided_calls`
  / `guided_text`). 15 tests remain green.

## [0.7.9] - 2026-08-05

Fix non-streaming chat completions returning empty `content` for Qwen3
thinking models when speculative decoding finishes the request (#364).

### Fixed
- **#364 - Non-stream `content: null` on Qwen3 thinking models.** The
  EAGLE3 draft-model speculative-decode path (`spec_decode_step` in
  `scheduler/spec_decode.py`) finished the request without setting
  `out.output_text` on the final `RequestOutput`. Non-streaming reads
  `output_text` off the merged final output (`engines/batched.py` →
  `clean_special_tokens`), so `content` came back empty while
  `completion_tokens` was still billed. Streaming (which accumulates
  `new_text` per-token) was unaffected. Mirrored the `output_text`
  decode + assignment already present in the sibling `ngram_spec_step`
  and `dflash_spec_step` finish paths.
- **#364 - `enable_thinking` dropped on `/v1/responses`.** The route
  set `enable_thinking` as a top-level chat kwarg, but `engine.chat`
  only forwards `chat_template_kwargs` to the chat-template render, so
  the Qwen3 template ran in default thinking-on mode and a
  `max_tokens`-truncated response lost the visible answer. Routed
  `enable_thinking` through `chat_template_kwargs` and applied the
  shared disable-by-default (`resolve_enable_thinking_default`) on both
  the stream and non-stream paths, matching the `/v1/chat/completions`
  behavior.
- **Test debt - stale mock targets after `vllm_mlx`→`fusion_mlx` rename.**
  `test_responses_chat_template_kwargs.py::TestAdapterForwarding`
  patched `vllm_mlx.service.helpers.get_config`; corrected to
  `fusion_mlx.config.get_config` (where `_get_cfg_attr` actually
  imports it).

### Skipped
- **#373 - Guided-decoding drift.** Skipped
  `TestBatchedEngineGuidedHonorsEnableThinking` (3 tests): pins
  `BatchedEngine.generate_with_schema` + `shared_apply_chat_template`,
  which were removed when guided decoding moved to the standalone
  `api/guided.py` helper. The `/v1/responses` route still calls the
  removed engine methods; tracked separately in #373.

## [0.7.8] - 2026-08-05

Expose GRPO / logprob endpoints to support real reinforcement-learning
training (#363).

### Added
- **#363 Phase 1 - `logprob` endpoint.** `POST /admin/api/fine-tune/logprob`
  returns `{logprob, token_count, per_token}` for a prompt+completion pair
  under a teacher-forcing single forward pass. Optional `adapter_path`
  scores under a LoRA adapter via native `mlx_lm.load(adapter_path=...)`.
  Backed by `fusion_mlx.training.logprob.compute_logprob` (handles both
  bare-logits and `(logits, cache)` model return shapes).
- **#363 Phase 2 - GRPO training endpoints.**
  - `POST /admin/api/fine-tune/grpo/jobs` creates a Group Relative Policy
    Optimization training job (LoRA policy, on-demand base-model reference).
  - `GET /admin/api/fine-tune/grpo/jobs`, `GET .../{id}`,
    `POST .../{id}/cancel`, `DELETE .../{id}`, `GET .../{id}/stream` (SSE).
  - `fusion_mlx.training.grpo.GRPOTrainer`: PPO-clipped policy loss with
    group-normalized advantages, `generate_step` sampling, configurable
    reward endpoint (`POST {prompt, completions} -> {rewards}`) with a
    length-based fallback, AdamW optimizer over LoRA params.
  - `fusion_mlx.training.grpo_service.GRPOService`: job queue mirroring
    `FineTuneService` lifecycle (persisted to
    `~/.fusion-mlx/adapters/grpo_jobs.json`), load-on-demand reference model
    evicted after each step.

## [0.7.7] - 2026-08-05

Implement the missing `fusion_mlx.positioned_kv_cache` module (#360).

### Added
- **#360 - `positioned_kv_cache` module.** Implements
  `positioned_update_and_fetch(cache, keys, values, position)` — a
  non-appending write that places keys/values at an arbitrary position
  in MLX-LM `KVCache` / `QuantizedKVCache` layers (buffer grows
  step-aligned via `_grow_kv_cache`, offset advances to
  `max(offset, position+num_steps)`). Operates on plain cache layers
  without subclassing, so layers still round-trip through
  `mlx_lm.save_prompt_cache` / `load_prompt_cache`. This is the
  pre-checkpoint write contract referenced by
  `runtime/disk_kv_checkpoint.py:443`. 11 unit tests
  (`tests/unit/test_positioned_kv_cache.py`): positioned writes on
  dense + quantized caches, step-aligned growth, offset semantics,
  negative-position rejection, and save/load round-trip for both cache
  types. The `disk_kv_checkpoint.py` TODO updated to reflect the module
  now exists; the scheduler hook that drives that checkpoint path
  remains unmigrated (tracked by
  `tests/unit/test_scheduler_disk_kv_hook.py` skip-tests).

## [0.7.6] - 2026-08-05

Fine-tune queue-stuck fix (#361) + source-code TODO hygiene.

### Fixed
- **#361 - second fine-tune job stuck in `queued`.**
  `FineTuneService.start_processing()` gated on a sticky `_running` flag
  that was set on the first job and never cleared — so every subsequent
  job was enqueued but `_process_queue()` was never called and the job
  stayed `queued` forever. `_running` is now treated as a one-time
  "initialized" marker (not a concurrency gate); `start_processing()`
  always calls `_process_queue()`, and `_current_job_id is not None` is
  the sole concurrency guard (it already was). Regression tests:
  `test_second_job_processed_after_first_drains`,
  `test_start_processing_idempotent_when_running`.

### Maintenance
- **Stale TODOs removed.** Three placeholder TODOs in
  `runtime/diffusion_lane.py` and `doctor/__init__.py` referenced
  modules that now exist (`model_aliases.resolve_profile`,
  `scheduler.BackpressureError`/`SchedulerConfig`, `bench.tier_runner`)
  — replaced with factual comments.
- **#360 - `fusion_mlx.positioned_kv_cache` not implemented.** The one
  genuinely-missing module referenced by `runtime/disk_kv_checkpoint.py`
  (lines 99/130/443, for `positioned_update_and_fetch` pre-checkpoint
  writes) now has a tracking issue (#360); the in-source TODO updated to
  reference it.

## [0.7.5] - 2026-08-05

Regression-test coverage for the OCR model enumeration crash (#359).

### Tests
- **#359 - `GET /v1/ocr/models` regression tests.** The crash
  (`AttributeError: 'EnginePool' object has no attribute 'engines'` at
  `ocr_routes.py:132`) was fixed on `main` in commit `0c15ff6` by switching to
  the public accessors `get_loaded_model_ids()` + `get_entry()`, but had no
  test coverage. Added `TestListOcrModelsRoute` to `tests/unit/test_ocr_routes.py`
  (2 cases): lists only `is_ocr_model` engines via the accessor contract, and
  no crash on an empty pool. Uses `MagicMock(spec=VLMBatchedEngine)` so the
  route's `isinstance` filter is exercised.

## [0.7.4] - 2026-08-05

Cross-host gateway shared-secret authentication (#352).

### Security
- **#352 - `FUSION_ROUTE_TOKEN` shared secret.** When the `FUSION_ROUTE_TOKEN`
  env var is set, `X-Fusion-Route` is upgraded from spoofable routing provenance
  to a credential: its value must equal the token, validated with
  `hmac.compare_digest` (constant time). Missing/mismatched → `403
  invalid_route_token`. When unset (default), the header keeps its
  presence-only check from #343 — no behavior change for existing single-host
  deployments. Validation lives in `RouteGuardMiddleware` (ASGI layer), so it
  applies uniformly before any route handler. Health probes (`/`, `/health`,
  `/healthz`, `/readyz`, `/livez`, `/openapi.json`, `/docs`, `/redoc`,
  `/favicon.ico`) and `OPTIONS` preflight remain exempt. The token is enforced
  even under `FUSION_ROUTE_WARN_ONLY=true` (stricter wins), so a cross-host
  gateway can't be accidentally downgraded to warn-only. Added
  `tests/unit/test_route_guard_token.py` (8 cases).

## [0.7.3] - 2026-08-05

Packaging fix to enable the PyPI first release (#348). The five git-commit-pinned
dependencies violated PyPI's no-direct-dependency policy
(`400 Can't have direct dependency: mlx-lm @ git+https://...`), blocking upload.

### Packaging
- **#348 - PyPI direct-dependency fix.** Replaced the five `@ git+...@<commit>`
  pins with precise PyPI `==` version pins that map to the same upstream code:
  `mlx-lm==0.31.3`, `mlx-embeddings==0.1.0`, `mlx-vlm==0.5.0`,
  `dflash-mlx==0.1.7`, `mlx-audio[tts,stt,sts]==0.4.3` (audio extra). Both the
  `[project] dependencies` and `[tool.uv] override-dependencies` sections updated.
- **Supply-chain integrity migration.** Replaced `scripts/verify_git_pins.sh`
  (validated git commit SHAs) with `scripts/verify_pypi_hashes.sh`, which locks
  and verifies the SHA256 of each pinned wheel + sdist against pypi.org's
  published digests. Run with `bash scripts/verify_pypi_hashes.sh` (5/5 OK).

## [0.7.2] - 2026-08-05

Patch release shipping fixes for #355, #356, #357, a latent converter weight-loading bug, and CI lint repairs.

### Reliability
- **#355 - load admission KV headroom.** Admission projection now reserves `min(max_kv_cache_memory, 2 GiB)` for the live KV cache + activations, using the last observed post-load footprint when available. Closes the weights-only under-projection that admitted models which then OOMed under concurrent load. Tunable via `FUSION_MLX_ADMISSION_KV_HEADROOM_GB` (0 disables).
- **#357 - `/v1/models/status` route shadowing.** The status route is now registered before the `gui_compat` catch-all `/v1/models/{model_name}`, so status is reachable.
- **Converter SIM118 breakage (latent).** `migrate/converter.py` iterated a `safetensors.safe_open` handle with `for key in f:`, which raises `TypeError` (safe_open has no `__iter__`). Restored `for key in f.keys():` with `# noqa: SIM118`. Runtime weight-mapping path; no test coverage.

### Observability / Docs
- **#356 - unset `iogpu.wired_limit_mb`.** When the sysctl is unset and the Apple Metal cap is within the static ceiling, the startup log is raised from `DEBUG` -> `INFO` with the actionable `sudo sysctl iogpu.wired_limit_mb=N` hint. New README subsection documents when/how to set the sysctl, value guidance, and persistence via `/etc/sysctl.conf`.
- **Admin active_requests counting** restored to a precise count.

### CI/CD
- **Lint job runs on Python 3.13.** black's AST safety check needs runner Python >= highest `target-version` (py313). Bumped `.github/workflows/ci.yml` lint job from 3.12 -> 3.13.
- Reformat `tests/unit/test_active_models_visibility.py` (committed unformatted).

## [Unreleased]

### Security
- **Path traversal fix.** `control_video`, `control_mask`, `reference_images`, `camera_conditions` in video routes now validated with `is_safe_local_path()` / `is_safe_url_with_dns()` via new `_validate_path_param()` helper.
- **Auth on all API routes.** 5 previously unauthenticated route files now require `verify_api_key`: `bench_routes`, `analyze_routes`, `migration_routes`, `recommend_routes`, `recommend_batch_routes`.
- **Anonymous access warning.** Server startup logs a WARNING when no API key is configured, guiding operators to set `FUSION_MLX_API_KEY` for production.

### CI/CD
- **Security scanning.** New `security` CI job: `pip-audit` (dependency vulnerabilities) + `bandit` (SAST).
- **Coverage reporting.** `pytest-cov` integrated: `--cov=fusion_mlx --cov-report=xml --cov-fail-under=40`. Coverage config in `pyproject.toml`.
- **Pre-commit hooks.** `.pre-commit-config.yaml` with ruff, ruff-format, trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files.

### Reliability
- **Graceful shutdown timeout.** uvicorn `timeout_graceful_shutdown=15` seconds for in-flight request drain.

### Documentation
- **Security Architecture** section in `docs/architecture.md`: auth flow, path safety policy, SSRF protection.
- **Video Generation Pipeline** section: Wan2 backend dispatch, adapter injection, NHWC layout GOTCHA.

### Tests
- **test_url_safety.py**: 11 tests for SSRF, path traversal, null byte, file URI handling.
- **test_auth.py**: 4 new middleware auth tests (anonymous, valid key, wrong key, missing key).

## [0.5.11] - 2026-07-30

### Added
- **Wan2.1-Fun-Camera support (#254).** SimpleCameraAdapter (PixelUnshuffle + Conv2d + ResidualBlocks) for camera pose control. Auto-detection from config.json `add_control_adapter` field. `camera_conditions` parameter through API → backend → generate pipeline.
- **`/v1/node/load` endpoint (#264).** Node-level load snapshot API.

### Fixed
- **Flaky GUI integration test.** `test_memory_auto_unload` now skips when GUI server not running (socket probe + `@gui_required` decorator).

### Changed
- **Branch cleanup.** All 13 merged feature/fix/wire branches deleted (local + remote). Only `main` remains.

## [0.5.8] - 2026-07-29

### Added
- **Debt paydown #50/#71/#80** (PR#240). Removed dead ServerConfig fields (`enable_tool_logits_bias`, `enable_audio_lane`), simplified `_sync_config()`, deleted `routes_internal/completions.py` shim, fixed 7 broken test imports, xfail 4 overflow tests, autouse video pool reset fixture.
- **T5 bfloat16 default + VAE sanitize** (PR#247). Default T5 encoder to bf16 (fp16 overflows on long sequences). Wan2 VAE NaN→0 replacement before clip.
- **`/v1/migration-level` endpoint** (PR#235). Model adaptation assessment API.
- **`/v1/recommend/batch` endpoint** (PR#236). Multi-model evaluation API.
- **`/v1/quantize/layered` endpoint** (PR#237). Per-layer quantization config API.
- **`/v1/benchmarks` endpoint** (PR#238). Real performance data API.
- **`/v1/analyze` endpoint** (PR#239). Model structure analysis API.

## [0.5.7] - 2026-07-28

### Added
- **VACE-14B E2E validated.** T2V coherent 832×480 (4 steps, ~185s on M5 Max). fp8 T5→bf16 auto-detect (fp16 overflows in attention). Control branch + weight remap + generate wiring.
- **UniWorld-V1 pure-MLX backend.** Qwen2.5-VL+SigLIP2+Flux. 65 unit tests.
- **Open-Sora V2 pure-MLX backend.** Flux MMDiT 11B (DoubleStream+SingleStream+3-branch CFG I2V). 26 unit tests.
- **CogVideoX pure-MLX backend.** 2B/5B (2B=no RoPE/Conv2d patch, 5B=3D RoPE/Linear patch). 42 unit tests.
- **Logprobs cross-layer plumbing for streaming chat.** Scaffolding landed, population deferred.

### Fixed
- **fp8 T5 encoder produced 100% NaN.** fp8 weights dequantized to fp16 → q/k/v values ~4000 → attention dot products overflow fp16 ±65504 range. Auto-detect uint8/int8 weights → default to bf16.
- **Wan2 config fallback + VACE backend registration.** model_type detection edge cases.
- **Video backends mx.load + float16 cast + expand_dims.** Weight loading consistency.
- **#228: 29 pre-existing embedding test failures.** EmbeddingRequest.max_length + MLXEmbeddingModel.processor setter.
- **#229: truncated mid-think reasoning_content and incomplete status.** Non-stream truncation dropped reasoning_content.
- **Packaging cleanup.** Removed tracked wheel from packaging/_wheels (1.9G build artifacts untracked via .gitignore).

## [0.4.8] - 2026-07-13

### Added
- **`/v1/convert` + `/v1/quantize` async job API.** Conversion and
  quantization load a full model into memory and write a new artifact, so a
  synchronous endpoint would block a worker for minutes. The new endpoints
  use an async job model: `POST /v1/{convert,quantize}` ->
  `{ "job_id", "status": "queued" }`, polled via
  `GET /v1/{convert,quantize}/jobs/{job_id}` ->
  `{ "status", "progress", "output_path", "error", ... }`. Jobs run on a
  single-worker thread pool (serialized to avoid OOM) and reuse the existing
  `fusion-mlx convert` CLI pipeline as the job body, so API behavior matches
  the CLI. `/v1/quantize` requires `quant_bits` or a float `quant_mode`
  (mxfp4/nvfp4/mxfp8). (`#110`, closes `#103`)

### Fixed
- **Code injection in agent graph Python export (`/v1/agents/.../export`).**
  `_generate_python_script` f-string-interpolated untrusted graph field
  values into generated Python source: `temperature` was interpolated
  unquoted into an expression (direct code injection) and `name` / `model` /
  `system_prompt` were interpolated into quoted strings without escaping
  (quote/newline breakout). All untrusted values are now embedded as a single
  `json.dumps` config literal (a valid, escaped Python dict), and
  `temperature` is coerced to a float and clamped to `[0.0, 2.0]`. (`#109`)

## [0.4.7] - 2026-07-13

### Added
- **Agent Graph API (`/v1/agents/*`).** New CRUD + execution endpoints for
  agent workflow graphs: list/create/read/update/delete graphs, export, and
  run a graph against a loaded model. (`#106`)
- **`/v1/base` endpoint for base-binding detection.** Exposes MLX runtime
  capabilities (version, Metal availability, KV-cache support, quantization
  formats, GPU info) so ecosystem components like Fusion-Model-Hub can verify
  the base before model operations. Honest about mlx limits: `gpu_cores` and
  `metal_family` are not reported by `mx.device_info()` and stay `null` rather
  than fabricated. (`#104`)

### Fixed
- **Wan2.2 `ti2v` models failed to load (`Model type ti2v not supported`).**
  Wan2.2 ti2v ships `config.json` with `model_type="ti2v"` but no
  `configuration.json` task manifest, so `_is_video_model` returned False and
  the model was misdetected as an LLM - mlx-lm then raised on load (HTTP 500).
  The check now falls back to recognizing config.json `model_type` in
  `{t2v, i2v, ti2v}` (still gated on diffusers subdirs). (`#95`)
- **`/v1/completions` silently dropped OpenAI params and mistyped `logprobs`.**
  `CompletionRequest.logprobs` was declared `bool | None` (copy-pasted from
  `ChatCompletionRequest`), but the OpenAI legacy-completions spec - and the
  route's own `0..5` range check at `routes/completions.py:121` - treat it as
  an integer top-k. A `logprobs: 2` request was rejected as `invalid_request`
  ("Input should be a valid boolean") before the route body ran. Additionally,
  `n`, `best_of`, `echo`, `response_format`, and `stream_options` were never
  declared, so Pydantic silently dropped them and the route crashed with
  `AttributeError` once it reached `request.n` / `request.echo`. `logprobs` is
  now `int | None` and the five fields are declared with types mirroring
  `ChatCompletionRequest`. Backward-compatible: `bool` still coerces to `int`.
- **`extract_multimodal_content` dropped `video` / `video_url` / `audio_url`
  parts.** The helper only emitted `image_url` / `input_audio` branches, so
  VLM video/audio inputs were silently lost and `has_multimodal` under-reported
  multimodal requests. All three content types are now preserved and counted.
- **`/v1/completions` logprob extraction crashed on `top_k <= 0`.**
  `_extract_token_logprob` called `np.argpartition` without guarding the
  empty / zero-k case. Added an early-return that emits the sampled token
  with an empty `top_logprobs` list.
- **Memory enforcer double-counted sustained over-ceiling pressure.**
  `_is_emergency_pressure` mutated `_over_ceiling_polls` and was called twice
  per check (initial + post-hot-cache-shrink recheck), inflating the
  sustained-pressure counter and triggering premature emergency eviction.
  Split into a side-effecting `_is_emergency_pressure` + a pure
  `_evaluate_emergency_pressure`; the recheck now uses the pure variant.
- **Engine pool unload blocked the pool lock on slow teardown.** `release_engine`
  / `unload_if_idle_unpinned` performed full engine teardown under the pool
  lock. Split into `_detach_engine` (under lock) + `_settle_unloaded_engine`
  (outside lock); sync `unload_engine` now subtracts `estimated_size` from
  accounting.
- **`fusion-mlx serve --model-dir` ignored `--host` / `--port`.**
  `_serve_from_model_dir` hardcoded `0.0.0.0:8000`. Now reads `args.host` /
  `args.port` (defaulting to `0.0.0.0` / `8000`).
- **LTX-2 distilled denoise returned bf16 latents.** `denoise_distilled`
  computed in f32 then cast back to bf16, losing precision for the downstream
  VAE decode. Now returns f32 latents (mirrors the dev path).
- **`ContentPart` (api/openai_models.py) lacked `video` / `audio_url`.** Added
  the `video` and `audio_url` fields plus an `AudioURL` model, restoring parity
  with the sibling `api/models.py`.

## [0.4.5] - 2026-07-07

### Fixed
- **Performance screen Apply did not persist (macOS app).** `GET /admin/api/global-settings`
  in flat-Settings fallback mode returned hardcoded `scheduler` / `memory` / `cache` /
  `mcp` values (e.g. `max_concurrent_requests: 8`), ignoring what `POST` wrote to
  `settings.json`. The POST handler persisted correctly, but GET never read it back, so
  every Apply looked like a no-op. `_build_fallback_global_settings` now reads those
  sections from `settings.json`, matching the existing `sampling` / `server` / `model` /
  `auth` read path.
- **Service stats showed "data couldn't be read because it is missing" (macOS app).**
  `server_metrics.to_dict()` emitted `total_tokens_prompt`, but the Swift `StatsDTO`
  expects `total_prompt_tokens` (via `convertFromSnakeCase`), so decoding threw
  `keyNotFound` and the Status screen rendered the system error. Unified all four sites
  — `server_metrics.py` (writer) plus `server.py`, `routes/health.py`,
  `routes/metrics.py` (readers) — on `total_prompt_tokens`. No Swift change needed.
- **App icon refresh.** Replaced Dock icon, AppLogo (light/dark), and menubar
  (outline/filled) assets.

## [0.4.2] - 2026-07-04

### Added
- **DSpark block speculative decoding** (DeepSeek DeepSpec). New lossless spec-decode
  mode: `fusion-mlx serve <model> --enable-dspark --dspark-drafter-path <draft>
  --dspark-draft-quant-bits 8`. Self-contained server forked early from normal serve
  init. A 5-layer context-injected drafter taps the target's own hidden states, proposes
  a 7-token block per round, and the target verifies it in a single forward pass via
  distribution-preserving rejection sampling (lossless). STS-calibrated confidence head
  (ECE 0.0846 -> 0.0184, AUROC preserved; `confidence_threshold=0.0` keeps STS inert,
  pruning monotonically hurts). New `supports_dspark` profile flag in `model_aliases`.

### Benchmark
- **Qwen3-8B-bf16 on Apple M5 Max 128GB** (median of 3 end-to-end HTTP serve runs):
  baseline vanilla serve 28.39 tok/s -> DSpark serve 48.15 tok/s = **+69.6%** (target
  was +50%). Lossless; greedy output is semantically equivalent to baseline. Results
  published to bench.dpdns.org (baseline id 2, DSpark id 3). See
  `docs/dspark-benchmark.md` for the full report.

## [0.4.1] - 2026-07-04

### Fixed
- **Speculative decode corruption on hybrid recurrent models.** On Qwen3.6-27B-mxfp8
  and other hybrid architectures (48 GatedDeltaNet recurrent layers + 16 full-attention
  layers, `ArraysCache`), speculative decoding produced incoherent repetition
  (`"useruseruser"`, `"Thinking1000000..."`) and overshot `max_tokens`. Root cause: the
  batched verify forward `model([D1..DK], cache)` computes recurrent state in parallel,
  but `ArraysCache` state must update sequentially — each token depends on the prior
  state — so batching derails it into a repetition fixed-point. Both n-gram and
  draft-model spec share this verify forward, so both corrupted identically.
  Fix: a boot-level gate in `engine_core.py` probes the loaded model's cache via the
  shared `model_has_recurrent_cache()` helper and skips both spec paths when recurrent
  layers are present, falling back to coherent pure decode. Pure-attention models keep
  speculative decode unchanged. The same probe powers `enrich_model_config`, so the boot
  gate and the config gate cannot drift apart.

### Benchmark
- **Coherent decode: ~18 tok/s overall / ~20 tok/s decode** (Apple M5 Max, 128 GB).
  This is the hardware ceiling for this 26.7 GB model — raw `mlx_lm.stream_generate`
  caps at 18.4 tok/s on the same hardware.
- **v0.4.0's reported 29.8 tok/s was a corrupt artifact, not real performance.** The
  0.4.0 benchmark measured speed only (HTTP API, `max_tokens=100`, 3 runs) and never
  coherence-tested the output. Every 0.4.0 spec-enabled run returned
  `completion_tokens=199` for `max_tokens=100` — a 99-token overshoot that is the
  signature of speculative decode — and the emitted text was incoherent repetition.
  0.4.0 running *coherently* (spec disabled) was also ~18 tok/s. **There is no real
  performance regression**: 0.4.1's coherent ~18 tok/s equals 0.4.0's true coherent
  throughput; 0.4.1 additionally fixes the corruption 0.4.0 shipped silently.

### Integrated (from v0.4.1-wip migration commits)
- Rapid-MLX tool parsers, fusion-mlx model patches, oQ quantizer, telemetry, MCP, middleware.

### Tests
- New `tests/unit/test_spec_recurrent_gate.py` (7 cases) covering the recurrent-cache
  probe and config enrichment. Net +7 passing, 0 regressions versus HEAD. 89 pre-existing
  failures from the rapid-mlx merge are documented test debt (broken
  `fusion_mlx.spec_decode` test imports) and are unrelated to this fix.

### macOS app
- The app's server-launch command (`python -m fusion_mlx.cli serve`) is verified coherent
  with this fix. The Swift app code is unchanged. Release `.app` bundles embed the
  `fusion_mlx` package from the worktree, so a rebuilt app includes this fix.
