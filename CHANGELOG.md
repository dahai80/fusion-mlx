# Changelog

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
