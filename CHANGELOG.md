# Changelog

## [Unreleased]

## [0.8.65] - 2026-09-02

### Fixed
- **#751 — GRPO training `RuntimeError: There is no Stream(gpu, N) in current
  thread`** (regression of #430). GRPO training runs `mlx_lm.generate.generate_step`
  in a worker thread (`asyncio.to_thread`). `mlx_lm.generate` captures a
  module-level `generation_stream` at import time on the main thread, and
  `generate_step` does `with mx.stream(generation_stream)` explicitly — so the
  prior #430/PR#432 fix (which only set the *default* stream in the worker) did
  not cover it: the stale main-thread stream is not bound in the worker and
  raises. Fix: `GRPOService._execute_grpo` now rebinds
  `mlx_lm.generate.generation_stream` to a fresh worker-thread-local stream
  (resolved via `importlib.import_module`, since `import mlx_lm.generate as x`
  binds the re-exported `generate()` *function* from `mlx_lm/__init__`, not the
  submodule) and restores the original in a `finally` block so a failed job
  cannot corrupt serve-side generation. Verified end-to-end on a real model:
  GRPO job completes with finite loss and mean_reward.
- **N in `Stream(gpu, N)` is the MLX stream index, not `group_size`.** The
  reporter saw N=4 (group_size=4) and inferred N tracks GRPO config;
  reproduction saw N=8 (stream-creation order). Documented to prevent
  recurrence.

### Tests
- `tests/unit/test_grpo_route.py`: +2 regression tests
  (`test_grpo_execute_grpo_rebinds_generation_stream`,
  `test_grpo_execute_grpo_restores_stream_on_error`) verifying the module global
  is rebound to a distinct worker stream and restored on both success and
  exception paths. 11 passed, 0 failed.

## [0.8.64] - 2026-09-01

### Added
- **#738/#739/#740 — Surface A+B+C ports to ltx2, opensora, uniworld backends.**
  Completes the VAE-encode (Surface A), ControlNet (Surface B), and inpaint
  (Surface C) surfaces for the remaining video backends behind the #653
  surface framework, following the cogvideox reference implementation
  (#731) and the 6-backend replication (#732-#737).
  - **#738 ltx2:** Surface A VAE-encode, Surface B ControlNet conditioning
    in `ltx2/denoise.py`, Surface C inpaint insertion in `ltx2/generate.py`.
  - **#739 opensora:** Surface A VAE-encode, Surface B ControlNet
    conditioning, Surface C inpaint insertion in `opensora/generate.py`.
  - **#740 uniworld:** ruled not-applicable (no per-backend VAE-encode /
    ControlNet / inpaint model surface); `uniworld/backend.py` gains a
    fail-visible `encode_control` refusal matching the #736 convention.
  - Inspection-based contract tests for each backend's surface branch
    (Surface B/C), plus `test_surface_uniworld_not_applicable.py`.

- **#688 step 2-3 — MiniMax-H3 ref2va reference-video-to-video generation.**
  Completes the ref2va partition: step 1 (forwarding plumbing) landed in
  v0.8.60; this lands the actual generation path.
  - **Architecture (corrected):** issue #688 was factually wrong — there is
    no separate `transformer_ref/` or `num_refiner_layers:2` difference.
    Ref2VA/transformer is the same MiniMaxH3DiTModel arch as FL2VA, just
    different weights. The real mechanism: reference videos route through
    the Qwen3-VL `vision_tower` → vision tokens scattered into `input_ids`
    at `<|video_pad|>` positions (`masked_scatter`) → deepstack visual
    embeds injected at the first N LM layers → 3D mrope `position_ids` →
    the LM's layer-49 hidden states become vision-conditioned
    `text_embeds` → the DiT denoises pure-T2V-style (`build_t2va_packed`,
    no latent condition rows).
  - **`MiniMaxH3MultimodalTextEncoder`** (`text_encoder.py`): wraps the
    full qwen3_vl VLM (keeps `vision_tower`), reuses mlx-vlm
    `Model.get_input_embeddings` (vision_tower + masked_scatter + deepstack
    + mrope), truncates the layer loop at layer 49, injects deepstack
    visual embeds, returns raw hidden states (no final norm).
  - **`_load_ref2va_frames`** + **`_encode_prompt_ref2va`** (`generate.py`):
    load reference video frames (mp4 via imageio, image via PIL), build
    Qwen3-VL chat-template content with `<|video_pad|>` placeholders, run
    `Qwen3VLProcessor` to expand + tokenize, then call the multimodal
    encoder for vision-conditioned `text_embeds`.
  - **`generate_video` ref2va branch**: when `reference_images` is set,
    loads the multimodal TE + `Qwen3VLProcessor` (staged — releases TE
    before DiT+VAE load to fit 137G physical RAM), encodes prompt with
    reference videos, then runs `generate_t2va_video`. Mutual-exclusion
    guards refuse ref2va combined with fl2va keyframes or joint audio
    (incompatible `text_embeds` conditioning).
  - **Backend** (`minimax_h3.py`): removed the `NotImplementedError`
    rejection of `reference_images`; now does path-safety then forwards to
    `generate_video`.
  - **Inspection-based contract tests** (`test_minimax_h3_i2v_guard.py`):
    assert ref2va branch present, old gate removed, empty-list
    fail-visible, multimodal encoder signature, mutual-exclusion, path
    safety.
  - No upstream block: mlx-vlm 0.5.0 has full `qwen3_vl`
    (VisionModel/LanguageModel/Model/processing_qwen3_vl) — reused, not
    blind-ported.

## [0.8.63] - 2026-09-01

### Added
- **#731-#737 — Surface B+C ports to 6 video backends.** Replicated the
  #653 pipeline surfaces to cogvideox (reference, #731), cosmos (#732),
  hunyuanvideo (#733), ltx_video_legacy (#734), ltx2_5 (#735),
  minimax_h3 (#736), and svd (#737).
  - **Surface C (inpaint-mask re-composite):** `apply_inpaint_mask(latents,
    init_latent, inpaint_mask)` inserted after each `scheduler.step` +
    `mx.eval` in every backend's denoise loop. DiT-agnostic, latent-space
    only; all-None default is bit-identical T2V passthrough. Ship on all
    7 backends.
  - **Surface B (ControlNet residual threading):** `controlnet_image`
    threaded through `VideoGenParams` + each backend `generate()` into
    `generate_video`/`denoise`. Backends without a per-backend ControlNet
    model (all 7 here — the shared adapter is Wan2-arch only) raise
    `RuntimeError` fail-visible when `controlnet_image` is set, refusing
    silent T2V degrade. No dead residual-injection plumbing.
  - `encode_control()` override on all 7 backends: raises if
    `controlnet_image` set, else logs pure-T2V and returns None.
  - `VideoGenParams` gains `inpaint_mask` and `init_latent` fields.
  - 12 new inspection-based contract tests (Surface B `encode_control` +
    Surface C `apply_inpaint_mask` insertion point per backend). 47 surface
    tests pass, 895 backend regression tests pass.

## [0.8.62] - 2026-09-01

### Added
- **#746 — FineTuneConfig passthrough of `weight_decay`, `max_grad_norm`, and
  `lora_target_modules` to mlx-lm 0.31.3.** Three SFT hyperparameters
  previously accepted by the API but silently dropped before reaching the
  training loop are now wired through end to end:
  - `weight_decay` is forwarded to the optimizer constructor for AdamW, SGD,
    Muon, and Adafactor (which accept it natively). `mlx.optimizers.Adam` has
    no `weight_decay` arg, so a non-zero `weight_decay` with the `adam`
    optimizer now fails visibly in `validate()` (Rule 12) rather than being
    silently ignored.
  - `max_grad_norm` applies global L2 gradient-norm clipping. mlx-lm 0.31.3's
    trainer calls `optimizer.update(model, grad)` inside the compiled step
    with no clip hook, so a thin `_GradClipOptimizer` wrapper overrides
    `update` to clip the gradient tree before delegating. Its `state`
    property proxies to the wrapped optimizer so the trainer's compiled
    `state` list captures the real optimizer state.
  - `lora_target_modules` restricts LoRA adapters to modules whose
    class-name basename is in the set (e.g. `["q_proj", "v_proj"]`). Resolved
    post-load into `config["keys"]` (full module paths), since mlx-lm 0.31.3
    gates adapters by `keys` and has no `--lora-target-modules` CLI flag.
    `None` preserves prior behavior (all linears in the top `lora_layers`).
  Defaults keep prior behavior; full fine-tuning rejects
  `lora_target_modules` (no adapters) and `max_grad_norm <= 0`.

## [0.8.61] - 2026-09-01

### Fixed
- **#741 — SkyReels ControlNet per-step residual failure now fails visibly.**
  The per-step ControlNet residual computation in
  `SkyReelsBasePipeline._denoise_sample` was wrapped in a bare
  `except Exception` that swallowed any failure, set `cn_residuals = None`,
  and ran the DiT forward with no conditioning — silently degrading a
  ControlNet request to T2V. This violated Rule 12 (fail visibly) and the
  same spec §4 invariant enforced for Wan2 `encode_control` in #653 (C1).
  The try/except is removed: `modify_denoise_step`/`get_residuals`
  exceptions now propagate and abort the run, and a `None` residual raises
  `RuntimeError` refusing silent T2V degradation. A caller asking for
  ControlNet conditioning no longer silently gets an unconditioned video.

## [0.8.60] - 2026-09-01

### Changed
- **#688 — MiniMax-H3 ref2va forwarding plumbing (step 1 of 3).**
  `generate_video` and `MiniMaxH3Backend.generate` now accept a
  `reference_images` parameter and forward it through the engine layer,
  making the previously-silent drop of reference images explicit. ref2va
  (reference-image-to-video) generation is **not yet implemented**: it
  requires a separate `transformer_ref` checkpoint (~67GB) and a
  reverse-engineered refiner forward (issue #688 steps 2-3, multi-session).
  Both `generate_video` (direct SDK path) and the backend route (before
  model load, fail-fast) raise `NotImplementedError` with an `issue #688`
  pointer when `reference_images` is provided, so the drop is fail-visible
  rather than silent. No behavior change for t2va/fl2va paths.

## [0.8.59] - 2026-09-01

### Fixed
- **#656 — watermark embed/verify patch (follow-up to v0.8.58).** Broad
  final review found four load-bearing defects that made the v0.8.58
  watermark feature non-functional on real models (all masked by 1D-only
  unit tests + CI-gated real-model tests):
  - **CRITICAL — signature used resolved filesystem path, not logical
    model id.** `compute_signature` was seeded from the resolved
    `model_path`, so embed→redistribute→verify-under-a-different-path
    produced a different signature and verification failed. Fix: sign
    over `request.model` (the caller's logical id), stable across
    relocation. Regression test
    `test_embed_verify_signature_stable_across_relocation`.
  - **2D+ weight tensors raised IndexError.** `embed_bits`/`extract_bits`
    used `np.flatnonzero` (flat indices) but indexed `weights[idx]`
    (axis-0), which only works for 1D arrays. Real 2D/4D weights crashed.
    Fix: flatten via `np.ascontiguousarray(weights).ravel()`, operate on
    the flat buffer, `out.reshape(orig_shape)` on embed.
  - **Carrier positions differed between embed and verify.**
    `generator.choice(eligible, size=k)` selects different positions for
    different sample sizes, so embed (k=payload_bits) and verify
    (k=array_size) read/wrote different slots → verify always failed.
    Fix: `_carrier_order` = a full deterministic
    `generator.permutation(eligible)` independent of count; embed/extract
    slice the same `[:k]` prefix.
  - **Verify returned 500 instead of `verified: false` on tampered
    weights.** Zeroed/quantized-below-epsilon carriers left zero eligible
    slots → `extract_bits` raised `ValueError` → 500. A tamper-evident
    verify must return `verified: false` on corruption. Fix: catch the
    `ValueError` in `_run_verify` and return `verified: false` with a
    corruption reason (fail-visible, Rule 12).

## [0.8.58] - 2026-09-01

### Added
- **#656 — weight-tensor watermark embed/verify endpoints.**
  - `POST /v1/watermark/embed` — embed a secret-seeded LSB spread-spectrum
    watermark into model weight tensors; returns a Hub-aligned signature
    (`sha256(secret:model::payload)[:32]`). Admin + hub-source gated.
    Reuses the convert/merge single-worker executor.
  - `POST /v1/watermark/verify` — extract + verify the embedded payload
    from tensors. Returns `verified`, `payload`, `signature`, `mismatch_rate`.
  - `FMH_WATERMARK_SECRET` env var (shared with Fusion-Model-Hub); route
    503s on default/empty secret.
  - `fusion_mlx.watermark.lsb` pure-numpy core (embed_bits/extract_bits/
    payload_to_bits/bits_to_payload/compute_signature).

### Changed
- Quantized (int) weight tensors are config-driven skipped during
  watermark embed/verify.

## [0.8.57] - 2026-08-31

### Added
- **#653 — VAE-encode / ControlNet / inpaint-mask engine surfaces.**
  - Surface A: `encode(pixels) -> 5D latent` wired on 9 video backends (Wan2,
    SkyReels, ltx_video_legacy, svd, cosmos, hunyuanvideo, cogvideox, minimax_h3,
    ltx2_5). Numpy-bridge + worker-thread `mx.eval` (#630 stream invariant).
    `ltx2` / `opensora` / `uniworld` deferred to follow-up issues.
  - Surface B: `encode_control(controlnet_image=, control_type=, controlnet_strength=)`
    builds a `ControlNet` adapter + control latent on `ControlState`; `denoise(control=...)`
    injects per-step residuals into the DiT block loop (Wan2 + SkyReels).
  - Surface C: `denoise(..., inpaint_mask=, init_latent=)` re-composites frozen regions
    after each denoise step (`mask*latents + (1-mask)*init`). Wan2 + SkyReels default path.
  - `VideoGenEngine.denoise` + `VideoBackend.denoise` thread `inpaint_mask`/`init_latent`
    (backward-compatible defaults `None`).
  - Real-model e2e tests gated behind `FUSION_MLX_REAL_MODEL_TESTS`
    (`tests/unit/test_653_real_model.py`): SkyReels VAE roundtrip, Wan2 ControlNet steer,
    Wan2 + SkyReels inpaint frozen-region preservation.
  - 10 follow-up issues filed for Surface B+C denoise-loop ports on the other backends.

### Changed
- `WanModel.__call__` gains `controlnet_residuals`/`controlnet_stride` kwargs (R1).
- `run_denoise` (Wan2 stage.py) gains `inpaint_mask`/`init_latent` + re-composite after `sched.step`.

## [0.8.56] - 2026-08-31

Patch release — audio speech route contract wiring (F2/F3/F4) + r11_b debt rescue.

### Fixed
- **Audio speech `format` fold + `response_format` validation (#724).** `AudioSpeechRequest` lived on an orphaned `api/models.py` copy with dead `model="kokoro"` / `voice="af_heart"` defaults that the live route never imported. The route now uses the live `audio_models.AudioSpeechRequest`, which validates `response_format` against `_ALLOWED_AUDIO_FORMATS` and folds the legacy `format` alias. Empty/whitespace `input` raises an OpenAI envelope 400 with `param="input"` (not a bare-string `detail`).
- **`voice="default"` never resolved (#725).** `_resolve_default_voice_literal` was defined but never called. Wired into `create_speech` so literal `voice="default"` resolves to the registry `default_voice` for the resolved model, assigned back onto `request.voice` so BOTH streaming and non-streaming paths see the resolved value. No voice allowlist (non-`"default"` passes through) — settled F3 contract.
- **Audio aliases advertised `[]` capabilities (#726).** `/v1/models` listing now shows `[_AUDIO_TYPE_TO_CAPABILITY[type]]` for audio aliases (was `[]`); single-id cards keep `["text"]`; added `/v1/models/{model_id}` retrieve route.
- **Mixed-case TTS alias 404 (#727).** Added case-insensitive TTS alias fallback (`_TTS_MODEL_ALIASES_LOWER`) so mixed-case HF repo names (`Kokoro-82M-bf16` / `KOKORO-82M-8BIT`) resolve instead of 404.

### Test Debt
- `test_audio_r11_b_bundle.py` permanently un-quarantined (32P/0F). Pool-seam rebuild (mirror `test_audio_tts.py`) replaced the stale `audio_route._tts_engine=None` singleton with live `_get_engine_pool()` injection. Pins 7 live contracts (F2-F7).
- `test_audio_r7_c_bundle.py` (#728), `test_audio_r8_a_bundle.py` (#729) stay quarantined with doc verdicts — 33F/31F all `REMOVED-ARCH/CONTRACT` (Rule 9: false coverage). 5 rescuable contracts migrated to `r11_b` as F5/F6/F7.
- `test_api_models.py` regression fixed — import `AudioSpeechRequest` from `audio_models`; re-pinned defaults to the live contract.

## [0.8.55] - 2026-08-31

Patch release — API-key override visibility (#705) + settings.json hardening.

### Fixed
- **Silent cli/env api_key override of `settings.json` (#705, #721).** `Server.__init__` resolved the effective api_key priority (cli `--api-key` > `FUSION_MLX_API_KEY` env > `settings.json auth.api_key`) silently — a cli/env key that differed from the configured settings.json key produced no warning, so clients sending the settings.json key got 401 with no startup signal of the cause. `_resolve_effective_api_key` now emits a `WARNING` when the resolved source is cli or env AND a settings.json key is set AND they differ, naming only the winning source (never the key value). 4 tests added in `tests/unit/test_api_key_priority.py`.
- **`settings.json` file mode tightened to `0o600` (#720).** The plaintext api_key in `settings.json` was written with the default umask mode (often `0o644`, world/group-readable). Save now applies `0o600` (owner read/write only), and load re-applies `0o600` to an existing file if the current mode is broader.

## [0.8.54] - 2026-08-30

Patch release — H3 VideoVAE qk_norm parity fix.

### Fixed
- **H3 VideoVAE attention qk_norm** (#716, closes #715). The MLX port of MiniMax-H3 `VideoVAE` skipped the official `qk_norm` (weightless RMSNorm on query/key before RoPE) declared in `config.json` (`qk_norm_type="rms_norm"`, `qk_norm_affine=false`). This was a silent numerical divergence from the trained model — attention logit scale shifted. Fix adds config-driven `vit_qk_norm_type` / `vit_qk_norm_affine` wired through `TransformerBlock` → `ViT3DDecoder` → `Attention`, applied to q and k before RoPE. `affine=false` means no learnable weights — state dict unchanged. 3 TDD tests added (`tests/unit/test_h3_vae_qk_norm.py`).

## [0.8.53] - 2026-08-30

Patch release — tool-call logits schema extraction crash fix (F-140) + test-debt rescue.

### Fixed
- **Tool-call logits schema extraction crash on malformed tool shapes (F-140).** `_extract_param_schemas` and `validate_param_value` (`api/tool_logits.py`) ran bare `.get()`/`.items()` on tool definitions, so a malformed shape (`parameters: null`/list/scalar, non-dict `properties`, tool or `function` field not a dict) raised `AttributeError` — surfacing as HTTP 500 instead of a clean per-tool skip. Added `isinstance(_, dict)` guards: a malformed tool is skipped (sibling tools still extracted), and a non-dict param schema is treated as no-constraint (value passes). Mirrors the F-031 narrow case plus the full ≥7 known crash family.

### Tests
- Rescued `tests/unit/test_tool_logits_schema_guards.py` from quarantine (was 26f/2p → 28p). Pins the F-140 helper contract: malformed shapes return an empty schema map (never raise), mixed-shape lists keep well-formed siblings, non-dict schema passes validation.

## [0.8.52] - 2026-08-30

Patch release — streaming `/v1/responses` cross-path parity fix (#707).

### Fixed
- **Streaming `/v1/responses` reasoning-item status under `max_output_tokens` (#707).** `_stream_responses` (`routes_internal/responses.py`) hardcoded per-item `"status": "completed"` on the terminal reasoning-item done event, diverging from the non-stream adapter (`build_responses_response`), which flips to `"incomplete"` when `finish_reason == "length"` AND no downstream output shipped (message body stripped or tool_calls present). The stream now mirrors the adapter: reasoning is `"incomplete"` only on a reasoning-only budget cutoff, else `"completed"`.
- **`reasoning_tokens` dropped from stream usage (#707 secondary).** The flat `GenerationOutput` production path never read `completion_tokens_details`, so streaming `usage.output_tokens_details.reasoning_tokens` was silently dropped even when the engine reported it. Now threaded from all three chunk-parse branches (flat / dict / choices-attr) into the terminal `ResponsesUsage`.
- **`response.in_progress` event payload shape.** The streaming `in_progress` event omitted `object`/`model`/`created_at`; now mirrors the `created` event fields.
- **`response.output_text.done` event.** Never emitted on the stream path. Now fires after the last text delta and before any `output_item.done` per the OpenAI Responses SSE contract, gated on a message having shipped.

### Tests
- Added `tests/unit/test_responses_stream_reasoning_status.py` — pins the cross-path parity (reasoning-only-length → `incomplete`, message-then-length → `completed`, reasoning-only-stop → `completed`, usage `reasoning_tokens` echoed). `test_responses_sse_event_order.py`'s two pre-existing failures (`test_response_in_progress_payload_shape`, `test_full_spec_event_order`) now green.

## [0.8.51] - 2026-08-30

Patch release — `--rate-limit 0` disable fix on the model-dir serve path (#692) + Qwen-Image / Qwen-Image-Edit variant registration (#689).

### Added
- **Qwen-Image DiT image variants (#689).** Registered `qwen_image` and `qwen_image_edit` in `ImageGenEngine.VARIANT_MAP`, backed by the vendored `mflux.models.qwen` MMDiT (txt2img `QwenImage` + multimodal-edit `QwenImageEdit`). `_infer_variant` routes HF repo ids (`Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512`, `mlx-community/Qwen-Image-2512-4bit`) to the txt2img variant and `*-edit*` ids to the edit variant (which requires `image_paths`). The edit check runs before the image check because both contain the `qwen-image` substring.

### Fixed
- **`--rate-limit 0` leaked the default limiter on the `serve --model-dir` path (#692).** The module-level `RateLimiter` (`middleware/auth.py`) defaults to `enabled=True` at 60 rpm. `#637` wired `configure_rate_limiter(args.rate_limit, enabled=args.rate_limit > 0)` into `_serve_audio_mode` and `_stage_server_config`, but the third serve path — `_serve_from_model_dir` — built the app directly and never called `configure_rate_limiter`, so the module default leaked and throttled bursty workloads despite the documented-disable flag. The model-dir path now calls `configure_rate_limiter(args.rate_limit, enabled=args.rate_limit > 0)` after the `server._api_key` staging and before `create_app`, matching the `#636` ordering (the limiter is read during app construction the same way the API key is). Regression-pinned in `test_serve_model_dir_uds.py`.

## [0.8.50] - 2026-08-29

Patch release — test-debt rescue: `test_model_auto_config.py` re-joined the active suite (+205 collected tests). No prod code changed.

### Tests
- **`test_model_auto_config.py` rescued** (removed from `tests/unit/debt_modules.txt`, test-debt 124 → 123). All 6 prior failures were stale tests referencing removed/renamed prod symbols, fixed test-only per the migration rule (prod not modified to satisfy quarantined tests):
  - `test_deepseek_v4` asserted `reasoning_parser='deepseek_r1'` for V4/V4-Flash. Prod intentionally returns `reasoning_parser=None` per #893 (codex MED): `_MODEL_PATTERNS` regex `deepseek.*v4` → `ModelConfig(reasoning_parser=None)`, and `_deepseek_template_family` returns `None` for V4. The honest minimal fix is to NOT speculate about V4/V5 reasoning format. Rewrote to assert `None`.
  - `test_table_for_dflash_alias_surfaces_opt_in_flag` asserted `detect_model_config('mlx-community/Qwen3.5-27B-8bit').supports_dflash is True`. `ModelConfig.supports_dflash` is a vestigial dead v1 field, never set `True` in prod. The real opt-in flag is `AliasProfile.supports_dflash2` (from `aliases.json` `supports_dflash2`), covered by `test_dflash_eligibility.py`. False-coverage test asserting dead behavior deleted per Rule 9 (#674 / PR #686 precedent).
  - `test_no_warn_on_v3_alias_with_v3_parser` / `test_warn_on_v3_alias_with_v31_parser` used the dead alias `deepseek-r1-8b-4bit` (renamed to -7b/-14b/-32b; alias removed). Swapped to the working alias `deepseek-v3-4bit` (resolves to `mlx-community/DeepSeek-V3-0324-4bit`, a V3-template checkpoint). Assertions updated: `deepseek_v3` + `deepseek_r1_0528` in-spec on the V3 family; `V3.0` warning string on cross-sub-family `deepseek_v31`.
- File now 205 passed / 0 failed. Active suite: 13092 → 13097 collected (+5 net; the 205-test file replaces a quarantined no-collect slot, but some parametrize rows were trimmed in the stale-test cleanup). Full suite **12451 passed / 0 failed**.

## [0.8.49] - 2026-08-29

Patch release — test-debt rescue: 7 quarantined test files re-joined the active suite (+160 collected tests), plus a cors fixture fix that resolves a latent full-suite failure. No prod code changed.

### Fixed
- **CORS fixture class-identity poison (#641).** `tests/unit/test_cors_env_configurable.py`'s `fresh_app` fixture used `importlib.reload(server_mod)` to reset CORS module state. `reload` re-executes `fusion_mlx.server` and creates a NEW `Server` class object, so `public_api.Server` (bound once at import) was no longer identity-equal to the reloaded copy — broke `test_public_api_reexports_match_internal` (`assert public_api.Server is Server`) only when run in the full suite. Dormant while `test_cors_env_configurable` was quarantined; the rescue activated the polluter. Replaced reload with `monkeypatch.setattr` of `app` + `_cors_mounted=False` + `_cors_origins=None` (auto-reverts after each test → clean per-test state, no class-identity poison). Removed now-unused `import importlib`.

### Tests
- **7 quarantined files rescued** (removed from `tests/unit/debt_modules.txt`): `test_audio_sts`, `test_audio_stt`, `test_audio_tts` (3 audio, Gap B fixtures, no cross-module 429), `test_cors_env_configurable`, `test_r12_m3_responses_stream_leading_items_order`, `test_routes_models_effective_parsers`, `test_v4_multi_session`. All verified genuine — import real prod symbols, not conftest no-op shims (Rule 9). `test_dense_sampler_fastpath` deliberately kept quarantined (passes against the conftest shim, not prod; honors #674 / PR #686). Active suite: 12932 → 13092 collected (+160); full suite 12246 passed / 0 failed.

## [0.8.48] - 2026-08-29

Patch release — staged I2V / VACE / camera conditioning API on `Wan2Backend` (#652). Extends the issue #410 sequential-offload stage API beyond pure T2V so the fusion-comfyui Phase-2 "Transparent Staged Default" covers every Wan2 video path, not just text-to-video. The staged `denoise`/`decode` previously handled only pure-noise T2V; I2V-14B channel-concat, TI2V-5B mask-blend, VACE control latents, and Fun-Camera paths went through the monolith `generate_video(params)`. Now they are stage-encodable.

### Added
- **`Wan2Backend.encode_control(...)` (#652).** Encodes I2V/VACE/camera conditioning up front into a `ControlState` dataclass (`fusion_mlx/video/wan2/stage.py:ControlState`) that the staged `denoise` threads into `run_denoise` bit-exactly mirroring `generate.py`'s per-step conditioning. Dispatches on `model_type`: `vace` → `_prepare_vace_control_latents` → `control_hidden_states`; `i2v` (14B) → VAE-encode first frame + mask → channel-concat `y_i2v` (20ch when `in_dim - vae_z_dim == vae_z_dim + 4`, else 16ch); `ti2v` (5B) → `preprocess_image` + VAE encode → `z_img` + `build_i2v_mask`; camera → reshape `camera_conditions` → `y_camera`. Pure T2V (`image=None`, no camera) returns `None` — the pure-noise path is untouched. Camera skips the VAE-encoder gate; VACE / i2v-channel-concat / i2v-mask-blend require an explicit `load_vae_encoder()` first and raise `RuntimeError("vae_encoder is unloaded; call load_vae_encoder().")` otherwise, matching the existing `dit is unloaded` / `vae is unloaded` stage-gate contract.
- **`Wan2Backend.load_vae_encoder()` / `unload_vae_encoder()` (#652).** Stage entry/exit for the VAE encoder used by `encode_control`. Lazy-loads via the shared `_load_vae_encoder_stage` helper (single weight-loading path, Rule 7). The `vae_encoder` stage flag follows the existing inject-on-load / pop-on-unload convention — it is NOT pre-declared in `_stage_flags`'s base dict.
- **`run_denoise` flat conditioning kwargs (#652).** Conditioning is lifted into `run_denoise` as flat kwargs (`control_hidden_states`, `control_scales`, `y_camera`, `y_i2v`, `z_img`, `i2v_mask`, `i2v_mask_tokens`, `is_i2v_mask_blend`, `is_i2v_channel_concat`); it builds the `ControlState` internally. `denoise(control=None)` stays the T2V pure-noise path.

### Tests
- **Staged I2V/VACE/camera unit suite (#652).** `tests/unit/test_wan2_stage_api.py`: 32 tests covering the 3 new methods + `denoise(control=...)` threading. Dispatch tests for all paths (T2V→`None`, VACE→`control_hidden_states`, i2v-14B→`y_i2v`, ti2v-5B→`i2v_mask`+`z_img`, camera→`y_camera`), the no-VAE-encoder gate (`RuntimeError`), and `run_denoise` kwarg threading (VACE `control_hidden_states`, ti2v `is_i2v_mask_blend`/`i2v_mask`/`z_img`). All fake/mock — no model load. Real-model bit-exact acceptance (staged == monolith, same seed, per variant) gated behind `FUSION_MLX_REAL_MODEL_TESTS` per the #630 convention (4 variants on disk: I2V-14B-480P, TI2V-5B-q8, VACE-14B, Fun-Camera-1.3B).

### Conventions
- `vae_encoder` stage flag is inject-on-load / pop-on-unload (NOT pre-declared), matching the main-branch `unload_vae` `pop` convention. `stop()` and `__init__` base `_stage_flags` dict stays 3 keys (`text_encoder`/`dit`/`vae`).

## [0.8.47] - 2026-08-28

Patch release — distributed KV-cache export/import endpoints (#650). Upstream primitive for fusion-multi-node P3-28 cross-node KV migration: serialize a shard's live KV tensors to base64 `.npy`, restore them on a peer node, resume decode from the prefix without recomputing.

### Added
- **`POST /distributed/kv_cache/export` (#650).** Serializes a shard's live KV-cache key/value tensors (`cache[i].state` → base64 `.npy`) for the shard's `[start, end)` layer slice, returning one entry per layer plus `seq_len` (the cache offset = number of cached tokens). Optional `layer_range` `[start, end)` subset is clamped to the shard's own slice. Reuses the `serialize_activation` b64-`.npy` transport already used by `pipeline_step` activations (bit-exact per mlx dtype, including `bfloat16`). Fails visibly (`400`) if the shard has no active KV cache (no `decode_step` prefill yet) or `seq_len == 0`, on malformed/out-of-slice `layer_range`, and `404` for unknown `shard_id`; oversized payloads hit the `_MAX_ACTIVATION_BYTES` ceiling (256 MiB default, env `FUSION_DIST_MAX_ACTIVATION_BYTES`).
- **`POST /distributed/kv_cache/import` (#650).** Restores previously-exported KV tensors into a loaded model's KV cache via `cache[layer].state = (k, v)` (sets `.offset = seq_len`), then `decode_step` continues as if the prefix had been computed locally. Lazy-inits the full-model-length cache list (`[KVCache() for _ in range(num_layers)]`, same pattern as `decode_step`) if none exists. Fails visibly (`400`) if any layer is outside the shard's slice, if a layer's tensor length `!= seq_len`, if the post-import offset check fails, on empty `layers`, and `404` for unknown `shard_id`.

### Tests
- **KV-cache export/import unit + real-model round-trip (#650).** `tests/unit/test_distributed_kv_cache.py`: 14 mock-based validation/route/round-trip tests (no model load — fake `KVCache` stand-ins with a `.state` property mirroring the real class) covering empty-cache/unknown-shard/bad-`layer_range` rejection, serialized-layer + `seq_len` return, `layer_range` subset, import slice/`seq_len`/empty validation, lazy-init + tensor restore, export→import tensor-equal round-trip, and the two routes (400/404 + 200 round-trip via `TestClient`). Plus one gated real-model acceptance test (`@pytest.mark.real_model` + `FUSION_MLX_REAL_MODEL_TESTS`): prefill → token1; baseline decode token2 (no reset); reset → export fails visibly; re-prefill → export; reset → import; decode same token → MUST equal the baseline token (KV state restored bit-exactly). All pass.

## [0.8.46] - 2026-08-28

Patch release — MiniMax-H3 `last_frame_image` engine-forwarding fix (#687).

### Fixed
- **#687 — `VideoGenEngine.generate` dropped `last_frame_image`.** The engine-layer `VideoGenParams(...)` construction in `fusion_mlx/engines/video.py` forwarded `image=`, `audio=`, `reference_images=`, `quantize=` but omitted `last_frame_image=`, so engine-layer callers (ComfyUI fusion-comfyui plugin, SDK clients using `VideoGenEngine.generate`) could not drive H3 l2va/fl2va last-frame-anchored generation — `params.last_frame_image` stayed `None` and the backend always ran first-frame-only. The HTTP `/v1/videos` route was unaffected (it sets `gen_kwargs["last_frame_image"]` directly). Added `last_frame_image=kwargs.get("last_frame_image")` to the `VideoGenParams(...)` call, mirroring the existing `audio=` / `quantize=` forwarding. `None` default preserves backwards compatibility for all other backends.

### Tests
- **Engine-forwarding regression for `last_frame_image` (#687).** Added `TestEngineForwardsLastFrameImage` to `test_video_quantize_plumbing.py` (mirrors the `quantize` forwarding test): asserts the engine forwards the kwarg into `VideoGenParams`, defaults to `None` when unset, and forwards both first-frame `image=` and last-frame `last_frame_image=` together (fl2va joint). Backend-side coverage already existed (`test_minimax_h3_backend.py`, `test_videos_routes.py`); the gap was the engine layer, which would have caught #687.

## [0.8.45] - 2026-08-28

Patch release — CORS env-var hardening (#675). Ports the four security-adjacent CORS features not landed in #641 and resolves the five tracking `xfail` tests in `test_cors_env_configurable.py`.

### Added
- **`FUSION_MLX_CORS_MAX_AGE` (#675).** Preflight `Access-Control-Max-Age` is now env-configurable (seconds). Malformed/empty values log a WARNING and fall back to a 3600 s default (replaces Starlette's silent 600 s), so preflight results cache longer and OPTIONS traffic drops.
- **`FUSION_MLX_CORS_ALLOW_HEADERS` (#675).** Response `Access-Control-Allow-Headers` is now env-configurable. Env unset → path-appropriate default (see F-091 below). Env present + non-empty → parsed list. Env present + empty → WARNING + fallback.
- **`FUSION_MLX_CORS_ALLOW_CREDENTIALS` opt-in (#675).** Credentials are now opt-in via env (`true`/`1`/`yes`/`on`). Default `False`.

### Changed
- **Credentials default reversed to `False` (#675, reverses #641).** `#641` set `allow_credentials=bool(_cors_origins)`, so any explicit origin auto-enabled `Access-Control-Allow-Credentials: true`. The documented default is now `False`; operators who need cookies must set `FUSION_MLX_CORS_ALLOW_CREDENTIALS=true`. Wildcard `["*"]` origins force `False` per the fetch spec. **Migration:** set `FUSION_MLX_CORS_ALLOW_CREDENTIALS=true` if you relied on the old auto-enable.
- **F-091 header narrowing on the env-driven path (#675).** When origins come from `FUSION_MLX_CORS_ALLOW_ORIGINS` (env), the default `Access-Control-Allow-Headers` narrows from wide-open `["*"]` to `content-type, authorization, x-rapid-mlx-internal`. The legacy `--cors-origins` CLI path keeps `["*"]` (back-compat). **Migration:** env-path operators sending custom headers (`OpenAI-Organization`, `X-Requested-With`, …) must allowlist them via `FUSION_MLX_CORS_ALLOW_HEADERS`.
- **Empty `FUSION_MLX_CORS_ALLOW_METHODS` now warns (#675).** An env value that parses to an empty list (e.g. `" , ,, "`) now logs a WARNING naming the env var and falls back to `POST,GET,OPTIONS` instead of silently broadening.

### Tests
- **CORS env-config suite fully green (#675).** The five `xfail(strict=False)` tests tracking unported Rapid-MLX features (`test_malformed_max_age_falls_back_to_default`, `test_empty_methods_env_warns_and_falls_back`, `test_empty_headers_env_warns_and_falls_back`, `test_credentials_default_false_with_explicit_origin`, `test_env_origins_path_applies_f091_narrowing`) now pass against real env-parse logic. `test_wildcard_logs_warning_and_works` updated to the #675 wildcard+credentials-False contract. `test_cors_env_configurable.py` now 18 pass / 0 fail / 0 xfail.

## [0.8.44] - 2026-08-28

Patch release — TTS `streaming_interval` small-value validation fix + audio server-fixture rebuild rescuing ~75 quarantined xfails.

### Fixed
- **`streaming_interval` rejected too eagerly by Pydantic.** `AudioSpeechRequest.streaming_interval` was constrained `ge=0.1`, which rejected any value below 0.1 with a 422 before the handler ran — defeating the handler's own `MIN_NATIVE_TTS_STREAMING_INTERVAL_SECONDS = 0.01` 400 guard. Relaxed the field to `gt=0.0` (only 0/negative rejected) so the handler owns small-value validation and returns a clean 400 with a diagnostic `detail` mentioning `streaming_interval`. Values `>= 0.01` now pass through to native TTS streaming as intended.

### Tests
- **Audio server-fixture rebuild (Gap B).** `server_audio_client` / `server_tts_client` / `server_sts_client` fixtures rewritten to mount the audio router on a fresh `FastAPI()` app and inject the mock pool via the designed `_get_engine_pool` test seam (previously patched the lazy-built `_server_state` dict, which is not an attr-bag in prod). Inline alias-resolution test bodies patched `_resolve_model` directly. All FastAPI app constructions in the audio suite now override `verify_api_key` + `check_rate_limit` dependencies to `lambda: None`, eliminating a cross-module 429 rate-limit flake (the module-level `RateLimiter` singleton accumulates across ~100 audio tests in one process). 2 TTS `language` tests marked `xfail(strict=False)` — `language` is not in the OpenAI `/v1/audio/speech` spec; TTS omits it by design (mlx-audio `lang_code` is a separate feature). Audio suite now 112 pass / 0 fail / 6 skip / 7 xfail when run together.

## [0.8.43] - 2026-08-28

Patch release — pure-memory-mode 500 fix + test-suite flake stabilization.

### Fixed
- **#681 — `AttributeError` 500 in `preload_matched_blocks` under `hot_cache_only` (pure-memory) mode.** In `hot_cache_only` mode `_cache_dir` is `None` but `save_block` still populates `_index` before the pure-memory early-return; `preload_matched_blocks` then walked the index, called `_get_file_path` (returns `None`), and an unguarded `.exists()` raised `AttributeError` → HTTP 500 on every chat request reaching preload with cold blocks. Guarded `file_path is None` before `.exists()` (#682).
- **Intermittent full-suite flakes.** `test_hard_limit_above_ceiling` compared `hard_limit` and `ceiling` from two separate live memory queries that drifted under load; rewritten to derive all values from a single `_get_ceiling_breakdown()` snapshot and assert the exact invariant. `test_telemetry_cli` help-listing tests used a raw `subprocess.run(timeout=15)` that exceeded 15s under full-suite CPU contention; routed through the file's `_run_cli` helper (30s timeout) (#683).

## [0.8.42] - 2026-08-28

Patch release — VAE-encode public surface, STT/VAE fail-visible guards, launchd TTS timeout, pool status fields, CORS test re-alignment.

### Added
- **VAE-encode surface on `ImageGenEngine` + `public_api` (#653, fixes #670).** `engine.encode(image)` exposes VAE image→latents with batch-norm normalization and packed-channel reshape; round-trip verified against a real Flux2 VAE (#671).
- **`in_use` + TTL fields in pool status (#647).** `GET /v1/models` status now reports per-model `in_use` and idle TTL so clients can decide reuse/eviction (#666).

### Fixed
- **#668 — STT/STS transcription exceptions swallowed before 500.** Audio routes logged only after the handler returned, so a transcription/STS failure produced a bare 500 with no server-side trace. Exceptions now logged before the 500 (#679).
- **#669 — WanVAE.encode accepted unsupported frame counts.** `WanVAE.encode` silently produced corrupt latents for frame counts it cannot handle; it now rejects them up front (fail visible) (#680).
- **#672 — CORS test contract drift.** `test_cors_env_configurable.py` re-aligned to the #641 three-state CORS contract (rescued from quarantine, 13 pass / 5 xfail tracking unported Rapid-MLX features) (#678).
- **#673 — admin UI dead `max_context_window_policy` field.** Dropped the stale policy field from the admin global-settings shape (#678).
- **launchd TTS timeout (#677).** The generated plist now sets `FUSION_TTS_TIMEOUT=600` so long-form TTS generation is not killed by the default 30s launchd wait.

### Tests
- **VLM visual-grounding contract lock (#654, #667).** Regression test pins the VLM image-grounding output shape.

## [0.8.41] - 2026-08-27

Patch release — Hub↔MLX API contract alignment (#646) + client-disconnect metrics (#645).

### Changed
- `POST /v1/models/{model_id}/load` and `/unload` now accept slash-bearing HF repo ids (e.g. `mlx-community/Llama-3.2`) via URL-encoding or raw slash; `/` in the id maps to the registered hyphen id (#646).
- Quantize job terminal status changed from `done` to `completed` (#646).
- gui_compat pool fallbacks (`_resolve_pool_model`/`_unload_pool_model`) now apply the same slash→hyphen resolve as the main load/unload handler, so slash-bearing ids resolve against the pool when the gui router shadows the main route (#646).

### Added
- `source_path` accepted as an alias for `model` on `POST /v1/quantize` (#646).
- `POST /v1/quantize/layered` and its job-status routes now mounted (were written but unreachable) (#646).
- `fusion_mlx_requests_cancelled_total` Prometheus counter for client-disconnected requests, ticked from the live streaming and `/v1/responses` non-stream disconnect handlers (#645).

### Removed
- Dead `_disconnect_guard` streaming wrapper, `_force_abort_request`, and always-no-op telemetry recorder stubs — 0 production callers; streaming routes handle disconnect inline (#645).

## [Unreleased]

## [0.8.38] - 2026-08-25

Patch release — HTTP auth infrastructure fix.

- API key priority CLI/env over settings.json (#632, PR #632):
  `Server.__init__` unconditionally ran
  `if self.settings.api_key: set_api_key(self.settings.api_key)`,
  overwriting the CLI/env-resolved key with the `settings.json` key and
  leaving `self.settings.api_key` (read by the `/v1` middleware via
  `global_settings_getter`) as the `settings.json` value. An operator
  launching behind a gateway with `--api-key <X>` (or
  `FUSION_MLX_API_KEY`) while `settings.json` held a different
  `auth.api_key` got 401 "Invalid API key" on every `/v1/*` request
  because the middleware enforced the `settings.json` key, not the
  operator's key. Fix: resolve the effective key once at startup with
  the documented priority `--api-key` > `FUSION_MLX_API_KEY` env >
  `settings.json auth.api_key` (new `_resolve_effective_api_key`
  helper) and sync it to all three read paths (`self.settings.api_key`,
  `set_api_key()`, `cfg.api_key`). Verified live: operator key 200
  (was 401), settings.json key now 401. 5 new unit tests + 1
  `@real_model` live-server test; 444 auth tests pass, 0 regressions.

## [0.8.14] - 2026-08-11

Patch release — Wan2.1-Fun-Camera control_adapter fixes: Conv2d weight
layout, post-patchify token-space injection, camera_conditions log
crash, and i2v channel-concat in_dim gating with stale-config probing.

- Wan2.1-Fun-Camera Conv2d weight layout (#451, closes #451, PR #452):
  `WanModel.sanitize()` passed `control_adapter.conv.weight` through
  without transposing from PyTorch `(out, in, kh, kw)` to MLX
  `(out, kh, kw, in)`. Camera pose conditioning produced garbage
  features. Fix: transpose 4D `control_adapter.*` weights `(0, 2, 3, 1)`.
- Camera adapter post-patchify injection (#453, closes #453, PR #454):
  `WanModel.__call__` injected `control_adapter(y_camera)` into the raw
  latent `x_list` BEFORE `_patchify` (latent space `[C,F,H/8,W/8]`), but
  the adapter output is post-patchify token space `[B,dim,F,H/16,W/16]`
  — a shape-incompatible add. Fix: defer injection to after `_patchify`
  in token space `[B,L,dim]` (transpose + reshape), matching upstream
  ComfyUI `x = patch_embedding(x) + control_adapter(camera_conditions)`.
- camera_conditions log crash (#455, closes #455, PR #454): the generate
  log line used `bool(camera_conditions)`, which raises
  `ValueError: Only length-1 arrays can be converted to Python scalars`
  once camera_conditions is an `mx.array`. Fix: `camera_conditions is not None`.
- i2v channel-concat in_dim gating (#456, closes #456, PR #454): the i2v
  channel-concat path always built `y_i2v = [mask(4), z_video(16)] = 20ch`,
  correct only for in_dim=36 (Wan2.2-14B). For in_dim=32 (Wan2.1-14B,
  Fun-Camera-1.3B) the patch_embedding Conv3d expects 32 channels but
  received 36 → `addmm` shape error (input 128 vs weight 144). Fix: gate
  mask concatenation on `extra_channels == vae_z_dim + 4` (20ch for
  Wan2.2-14B; 16ch video-only for in_dim=32).
- Stale config.json in_dim probing (#456, PR #454): Wan2.1-14B dirs can
  hold an i2v checkpoint with in_dim=36 (from patch_embedding.weight) but
  config.json says 32. `correct_in_dim()` probes the DiT safetensors for
  `patch_embedding.weight` and overrides config.in_dim when it disagrees
  with the weights, mirroring upstream ComfyUI. Wired into all three
  config-load sites (generate.py, wan2 backend, stage.py).

## [0.8.13] - 2026-08-10

Patch release — Flux.2-klein 9B config misclassification fix, image
route alias-resolution 404 fix, flux-2/kokoro aliases, SD3 fp16
quant-reload guard, and TTS audio/speech traceback logging.

- Flux.2-klein 9B config misclassification (#449, closes #449): the
  `_infer_flux2_config` matcher tested the substring "4b" before "9b",
  so quantized model ids like `flux2-klein-9b-4bit` matched "4b" (from
  "4bit") and picked the 4b config (24 heads). The 4b config's
  inner_dim=3072 mismatches the 9b weights' inner_dim=4096, breaking
  the transformer reshape with
  `Cannot reshape array of size 4194304 into shape (1,1024,24,128)`.
  Fix: reorder checks so "9b"/"kv"/"base" are tested before "4b".
  Regression test added. Real-model verified (valid 1024x1024 PNG).
- Image route alias-resolution 404 (#446, closes #446): image
  generation requests with a short alias hit a 404 because the image
  route resolved the model id against the wrong registry. Wired alias
  resolution through the shared `resolve_model_id` path so aliases
  like `flux-2` and `kokoro` resolve before the route lookup.
- flux-2 / kokoro aliases (#447, closes #447): added `flux-2` ->
  `flux2-klein-9b-4bit` and `kokoro` -> `Qwen3-TTS-12Hz-1.7B-Base-8bit`
  to `model-config.json` aliases.
- SD3 fp16 quant-reload guard (#435, closes #435):
  `SD3Pipeline._load_transformer_and_vae` ran `nn.quantize` then
  unconditionally reloaded `load_transformer`, which overwrote uint32
  quantized weights with fp16 checkpoint tensors and broke
  `quantized_matmul` on SD3-Medium. Fix: only reload when the
  checkpoint has quant metadata (.scales/.biases/.qweight); fp16
  checkpoints skip the reload.
- TTS audio/speech traceback logging (#450, closes #450): the
  `/v1/audio/speech` endpoint swallowed engine-load, streaming, and
  synthesize exceptions into a bare 500 with no traceback, making
  failures undiagnosable. Added `logger.exception` to all three
  swallowed except blocks. (The 500 itself no longer reproduces after
  the alias work; the logging is for future diagnosability.)

## [0.8.12] - 2026-08-08

Patch release — GGUF load guard, Wan2 staged VAE decode cross-thread
fix, and a DPO logprobs TypeError fix.

- GGUF load guard (#423): mlx-lm/mlx-vlm/mlx-embeddings have no GGUF
  load path (mx.save_gguf is one-way export); loading a .gguf file or
  GGUF-only dir crashed with an opaque error. New shared guard
  `fusion_mlx/engine/gguf_guard.py` (`assert_not_gguf`) is called
  before every load entry (LLM/VLM/Embedding/Reranker ×3) and raises a
  clear `GGUFLoadError` pointing at `mlx-community` repos or
  `POST /v1/convert`. Non-GGUF targets are a no-op. 10 unit tests.
  Closes the "GGUF 加载桥" audit action item; duplicate-engine debt
  tracked in #422.
- Wan2 staged VAE decode Stream fix (#419, closes #418): the Phase-2
  staged pipeline raised `There is no Stream(gpu, N)` at VAE decode
  because a lazy denoised latent crossed threads. Fix: `mx.eval` the
  5D batch projection on the executor thread before returning; slice
  `latent[0]` inside `_decode` on the executor thread. Real-model e2e
  (Wan2.1-T2V-1.3B) passes, sequential offload confirmed.
- DPO logprobs TypeError fix (#421, closes #420): `_ref_logprobs_pair`
  called `mx.sum` on a Python list; convert to `mx.array` first.

## [0.8.11] - 2026-08-08

Patch release — Wan2 video backend pipeline stage API (#410/#416),
and VL cold-start crash fix verified (#413 closed).

- Wan2 pipeline stage API (#410, PR #416): refactors the Wan2 video
  backend into a staged pipeline so `Wan2Backend` exposes the issue #170
  stage contract (10 methods: `load_text_encoder`/`encode_text`/
  `unload_text_encoder`, `load_dit`/`denoise`/`unload_dit`,
  `load_vae`/`decode`/`decode_tiled`/`unload_vae`). New
  `fusion_mlx/video/wan2/stage.py` shares the weight-loading path with
  `generate.py`. Unblocks Fusion-ComfyUI Phase-2 sequential-offload flow
  (load → use → unload each component so only one heavy model holds
  memory at a time). Text-to-video scope; I2V/VACE stay on the
  monolithic `generate()` path. 20 new unit tests, full suite 8110
  passed 0 failed.
- VL cold-start crash (#413): closed as fixed. Verified on main
  (post-v0.8.10) that cold-start first VL request against
  `Qwen2.5-VL-7B-Instruct-4bit` returns HTTP 200 with no
  `There is no Stream(gpu, N)` error. Root cause (#411/#414, MLX 0.31.3+
  cross-stream weight binding) fixed by binding VLM load to a dedicated
  single-worker MLX executor.

## [0.8.10] - 2026-08-07

Patch release — preference-alignment training, QLoRA quantized-base
fine-tuning, speculative-decoding benchmark, and a VLM cross-stream
load fix.

- DPO/ORPO preference training (#399): new `DPOConfig`/`DPOTrainer`
  pipeline with DPO reference-model loss and ORPO odds-ratio (no ref
  model) variants, exposed via `POST /admin/api/fine-tune/dpo/jobs` and
  `POST /admin/api/fine-tune/orpo/jobs` (+ list/status/cancel routes).
  `DPOService` runs the job queue; saved adapters reuse the existing
  `tree_flatten` adapter path. 15 unit tests cover config validation,
  service lifecycle, and both loss paths.
- QLoRA fine-tune (#402): `FineTuneConfig` gains `quantize_base` /
  `quant_bits` and a `qlora` `fine_tune_type`. Quantizes the frozen base
  weights (4/8-bit) so larger models fit in VRAM during LoRA training,
  then dequantizes for adapter merge/save.
- Speculative-decoding benchmark (#388): `scripts/bench_spec_decode_388.py`
  streaming bench + `BENCHMARK_SPEC_388.md` report. EAGLE3 on
  Llama-3.1-8B-Instruct (dense KV-cache) = 1.445× @ 63.4% acceptance on
  short generation, degrading to 1.033× @ 47.3% on long generation. A
  generic draft model was a 0.78× slowdown (24.6% acceptance) — an
  EAGLE3-trained draft is required, not a generic LM. The PRD target
  Qwen3.6-27B-mxfp8 is hybrid GDN (recurrent `ArraysCache`), so
  draft-model/EAGLE3 spec is auto-gated off by design (`spec_eligible =
  not model_has_recurrent_cache(model)`); only N-gram/suffix/prompt-lookup
  spec runs there — use DFlash for that model.
- VLM cross-stream load fix (#411): MLX 0.31.3+ binds model weights to the
  stream of the thread that first touches them. The VLM on-demand load ran
  on `get_executor("io")` (a 2-worker pool without MLX stream init), so
  weights bound to an io-worker thread's stream while vision-prep + prefill
  ran on the engine's `mlx_executor` → cross-stream access raised
  `There is no Stream(gpu, 0) in current thread` on every VL request when a
  text model was already warm. Load now runs on a dedicated single-worker
  `ThreadPoolExecutor` with `initializer=_init_mlx_step_thread`, reusing
  the engine's owning thread/stream so load + prefill share one stream.
- CI: fixed ruff I001 in `engines/vlm.py` (stdlib import wedged mid
  first-party block) that blocked PR #414's lint job.

## [0.8.9] - 2026-08-07

Patch release.

- LoRA allowed-dirs startup race (#394): `FUSION_LORA_ALLOWED_DIRS` hit a
  startup-order race — `EnginePool` cached an empty allowed-list at init
  (constructed before `server.py`'s auto-add of `~/.fusion-mlx/adapters`),
  then rejected all per-request adapters. `server.py` also joins the env
  with `:` while the historical contract documented `,`, so a colon-joined
  list parsed as one literal segment. `_resolve_allowed_adapter_dirs()`
  now accepts both `,` and `:`; `_validate_adapter_path()` re-resolves from
  env on each call (falls back to the cached list only when env is empty),
  so the late auto-add takes effect.
- Standalone route-guard default (#398): since v0.7.0 `route_guard`
  enforces `X-Fusion-Route` by default; the gateway injects it, but
  `start.sh` is the standalone launcher (loopback, no gateway) — so every
  local /v1/* call was rejected. `preflight()` now defaults
  `FUSION_ROUTE_WARN_ONLY=true` when unset. Gateway deployments override
  with `=false`; pre-existing values are never clobbered.
- Rescued quarantined `test_admin_auth.py` back into the CI gate: fixed
  stale assertions (bound-name patch target for `TestCheckUpdate`,
  loopback-only `skip_api_key_verification` for `TestSkipAdminAuth`,
  renamed `fusionmlx_admin_session` cookie for `TestSessionCookieName`).

## [0.8.8] - 2026-08-07

Patch release.

- Fine-tune models-list route (#397): add static route
  `GET /admin/api/fine-tune/jobs/models` aliasing the existing
  `list_finetunable_models` handler, registered before the parameterized
  `/api/fine-tune/jobs/{job_id}` route. Previously the parametric route
  captured `job_id=="models"` and returned `404 Job not found: models`,
  breaking fusion-trainer's model enumeration. The legacy
  `/api/fine-tune/models` path is unchanged (backward compatible). Same
  payload: `[{model_id, model_type, model_path, loaded}]`.

## [0.8.7] - 2026-08-07

Patch release.

- Incremental `tool_call_delta` streaming (#385): wire the dormant
  `extract_tool_calls_streaming` into the streaming chat generator so tool
  calls emit as `delta.tool_calls` SSE mid-generation (OpenAI standard)
  instead of waiting for `gen.finished` all-at-once. Opt-in only when
  `request.tools` is present; non-tool streams are byte-identical
  (backward compatible). Reuses existing tool parsers via
  `_resolve_streaming_tool_parser` (profile/registry -> `"auto"` fallback);
  dedups emitted calls by `index`. The `gen.finished` finalize fallback is
  preserved (skips re-emit when already streamed; emits all-at-once when no
  inline detection fired).
- Prefix cache session-agnostic documentation (#386): document that the
  prefix cache is keyed purely by token-prefix chain hash (no `session_id`
  dimension), so forked sessions share prefix KV automatically. Closed by
  design; no code change.

## [0.8.6] - 2026-08-05

Patch release.

- In-place LoRA swap (#389): keep a single base engine resident and swap the
  LoRA adapter onto it in place via mlx_lm's `LoRALinear` machinery
  (`load_adapters` / `remove_lora_layers`) instead of reloading the full base
  model per adapter. Correct for 4-bit / 8-bit quantized bases (low-rank arrays
  added beside the quantized linear, never fused into packed weights) and
  allocates no second base copy. A per-base `asyncio.Lock` serializes the
  apply → infer → restore window; bare-base requests wait for any in-flight
  swap to restore. Opt-in via `FUSION_LORA_INPLACE_SWAP=1` (default OFF
  preserves the existing per-adapter derived-engine behavior). Switch latency
  ~7 ms apply / ~1 ms restore on Qwen3-0.6B-4bit. See
  [docs/lora-inplace-swap.md](docs/lora-inplace-swap.md).

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
