# Changelog

## [Unreleased]

### Fixed
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
- Rapid-MLX tool parsers, omlx model patches, oQ quantizer, telemetry, MCP, middleware.

### Tests
- New `tests/unit/test_spec_recurrent_gate.py` (7 cases) covering the recurrent-cache
  probe and config enrichment. Net +7 passing, 0 regressions versus HEAD. 89 pre-existing
  failures from the rapid-mlx merge are documented test debt (broken
  `fusion_mlx.spec_decode` test imports) and are unrelated to this fix.

### macOS app
- The app's server-launch command (`python -m fusion_mlx.cli serve`) is verified coherent
  with this fix. The Swift app code is unchanged. Release `.app` bundles embed the
  `fusion_mlx` package from the worktree, so a rebuilt app includes this fix.
