# #653 VAE-encode / ControlNet / Inpaint-mask Engine + public_api Surfaces — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship three engine surfaces on fusion-mlx video backends — (A) VAE `encode(pixels)->latent` on all 11 real backends, (B) ControlNet denoise-time conditioning on Wan2 (WanControlnet adapter) + SkyReels (own adapter, config-driven), (C) frozen-region inpaint mask on Wan2 + SkyReels — exposing them through the already-public `VideoGenEngine` class.

**Architecture:** Approach B (approved spec). Surfaces stay orthogonal: Surface C (inpaint mask) is a `denoise(..., inpaint_mask=, init_latent=)` sibling param re-composited via a neutral helper after each `sched.step`, never touching `ControlState`. Surface B (ControlNet) threads a `ControlNet` adapter instance + preprocessed control latent through `ControlState` (Wan2) or engine-layer config fields (SkyReels); residuals are step-dependent, computed per-step inside the denoise loop and injected into the DiT block loop. Surface A wires each family's existing model-layer VAE `.encode()` through the backend's `load_vae_encoder`/`encode`/`unload_vae_encoder`, mirroring the Wan2 reference, with the #630 thread-portability invariant (numpy-bridge on caller thread + executor + `mx.eval` on worker thread).

**Tech Stack:** Python 3.11-3.13, MLX (`mlx.core`), FastAPI, NumPy bridge for cross-thread stream ownership. Tests: pytest, `FUSION_MLX_REAL_MODEL_TESTS` gate for model-load tests. Lint: black + ruff.

**Spec:** `docs/superpowers/specs/2026-08-31-653-vae-controlnet-inpaint-surfaces-design.md`

## Global Constraints

- 4-space-multiple indentation, no docstrings, default logging on every new code path, clean test data after verification.
- Code/commits/PRs written normal (not caveman); commit messages use conventional prefixes (feat/fix/test/docs/chore).
- Push to `origin` remote (NOT `github`); tag THEN release; PyPI skipped (no token).
- Real-model tests gated behind `FUSION_MLX_REAL_MODEL_TESTS`; never `mx.clear_streams()` in tests (cross-test MLX stream pollution).
- NEVER silently average two patterns (Rule 7): WanControlnet (Wan2) and SkyReels `create_adapter("controlnet")` (SkyReels) stay distinct mechanisms — uniform ENGINE surface only.
- Fail visibly (Rule 12): wrong-ndim pixels, missing encoder, mask shape mismatch, adapter model mismatch all raise with context, never silent clamp/garbage.
- Thread-portability invariant (#630): `encode` MUST numpy-bridge on the caller thread, rebuild `mx.array` inside the executor, `mx.eval` on the worker thread. Skipping it → `RuntimeError "no Stream(gpu, N) in current thread"`.
- Use `.venv/bin/python -m pytest` (not bare `pytest` — rtk proxy swallows grep). `python` not in PATH outside venv.
- black needs `--fast`; NEVER black/ruff `debt_modules.txt`.

## Plan rulings (recorded before execution)

- **Ruling R1 — Wan2 DiT model-layer injection site.** Spec Section 1 Surface B says "`run_denoise` calls `adapter.compute_residuals(...)` each step and adds residuals to the DiT hidden states" but does not file-list the DiT model. Verified: `WanModel.__call__` (wan_2.py:331) has NO `controlnet_residuals`/`controlnet_stride` kwargs; its block loop (wan_2.py:560) injects only VACE `vace_hints`. To realize "adds residuals to the DiT hidden states" we add `controlnet_residuals: list | None = None` + `controlnet_stride: int = 4` kwargs to `WanModel.__call__` and inject in the block loop, mirroring the EXISTING SkyReels DiT pattern (pipelines/__init__.py:716-724 `dit(..., controlnet_residuals=cn_residuals, controlnet_stride=cn_stride)`). This is the model-layer injection site the spec described at high level — not a deviation. Cost if wrong: residuals land in wrong tokens / wrong stride → garbled control; caught by real-model ControlNet test (frozen-region assertion analog).

- **Ruling R2 — latent-shape reconciliation for WanControlnet in Wan2.** `ControlNet.compute_residuals` expects `hidden_states: [B, C_vae, H, W]` (B-first, NHWC-ish, single-frame convention built for SkyReels). Wan2's `run_denoise` operates on `latents` 4D `(z_dim, t_latent, h_latent, w_latent)` (C-first, temporal). `run_denoise` must reshape `latents` to the convention `compute_residuals` expects before the call, and reshape residuals back to Wan2 token space for DiT injection. This reconciliation is an explicit step in the Surface B Wan2 task (Task 5), with the exact reshape. Flagged as the riskiest task; if it proves infeasible, that is a systematic-debugging moment + ruling, NOT a reason to silently drop Surface B Wan2.

- **Ruling R3 — Surface A batching.** The 9 denoise-less backends' Surface A are identical-shape mechanical edits (same 3 methods, family-specific VAE path only). Per the subagent batching rule, ONE dispatch implements all 9; SkyReels Surface A is a separate task (it also carries B/C-adjacent review surface); any backend whose family VAE lacks a separable encoder uses the #670 full-VAE-load fallback inside the same batch task.

- **Ruling R4 — default denoise path only for v1.** Surface C inserts into Wan2 `run_denoise` (stage.py:497 after `sched.step`) and SkyReels `_denoise_sample` default loop (pipelines/__init__.py:743). SkyReels async/speculative paths (`_denoise_sample_async` 780, `_denoise_sample_speculative` 898/1501) are gated OFF by default (#177/#180) — mask insertion there is a follow-up if those paths ship as default, filed in Task (follow-up issues).

---

## File structure

### Files created
- `fusion_mlx/engines/video_backends/_inpaint.py` — neutral helper `apply_inpaint_mask(latents, init_latent, mask) -> mx.array` + `patch_downsample_mask(mask, vae_stride, patch_size, t_latent, h_latent, w_latent) -> mx.array`. Pure MLX ops, no backend dependency.
- `tests/unit/test_inpaint_mask_helper.py` — unit tests for both helpers.
- `tests/unit/test_controlnet_wire.py` — Surface B Wan2 unit tests (fake adapter).
- `tests/unit/test_skyreels_controlnet_wire.py` — Surface B SkyReels unit tests (config-driven).
- `tests/unit/test_vae_encode_<backend>.py` — per-backend Surface A unit tests (fake VAE). One file per backend (or one batched file `test_vae_encode_surfaces.py` covering all 10 non-Wan2 + SkyReels).

### Files modified — Surface C
- `fusion_mlx/engines/video.py` — `denoise` signature gains `inpaint_mask`, `init_latent`; forwarded to backend.
- `fusion_mlx/engines/video_backends/base.py` — abstract `denoise` signature gains the two params (default `None`, backward-compatible).
- `fusion_mlx/engines/video_backends/wan2.py` — `denoise` accepts + threads the two params into `run_denoise`.
- `fusion_mlx/video/wan2/stage.py` — `run_denoise` loop inserts `apply_inpaint_mask` after `sched.step` (line 497).
- `fusion_mlx/engines/video_backends/skyreels.py` — `denoise` accepts + threads into `_denoise_sample`.
- `fusion_mlx/video/skyreels_v3/pipelines/__init__.py` — `_denoise_sample` gains the two kwargs; insert after line 743.

### Files modified — Surface B
- `fusion_mlx/video/wan2/stage.py` — `ControlState` gains `controlnet_adapter`, `controlnet_latent`; `run_denoise` computes per-step residuals + threads to DiT.
- `fusion_mlx/video/wan2/wan_2.py` — `WanModel.__call__` gains `controlnet_residuals`/`controlnet_stride` kwargs + block-loop injection (Ruling R1).
- `fusion_mlx/engines/video_backends/wan2.py` — `encode_control` gains `controlnet_image` path (build adapter, preprocess, store on ControlState).
- `fusion_mlx/video/adapters/controlnet.py` — pin/verify `compute_residuals` contract (tests-first); no prod change unless signature gap found.
- `fusion_mlx/engines/video_backends/skyreels.py` — `encode_control(controlnet_image=, control_type=, controlnet_strength=)` sets `pipeline.config` fields.

### Files modified — Surface A (per backend, mechanical)
`skyreels.py`, `cogvideox.py`, `cosmos.py`, `hunyuanvideo.py`, `ltx_video_legacy.py`, `ltx2_5.py`, `ltx2.py`, `minimax_h3.py`, `opensora.py`, `svd.py` — each gains `load_vae_encoder`/`encode`/`unload_vae_encoder` mirroring Wan2.

### Docs
- `README.md` — document the three surfaces + `VideoGenEngine` method signatures.
- `CHANGELOG.md` — entry under the patch release.

---

## Task 1: Inpaint mask helper + tests (Surface C foundation)

**Files:**
- Create: `fusion_mlx/engines/video_backends/_inpaint.py`
- Create: `tests/unit/test_inpaint_mask_helper.py`

**Interfaces:**
- Consumes: nothing (pure MLX).
- Produces: `apply_inpaint_mask(latents, init_latent, mask) -> mx.array` and `patch_downsample_mask(mask, vae_stride, patch_size, t_latent, h_latent, w_latent) -> mx.array` — used by Tasks 3, 4.

<!-- Task 1 steps: fill -->

---

## Task 2: Surface C engine + base signature threading

**Files:**
- Modify: `fusion_mlx/engines/video.py` (`VideoGenEngine.denoise`)
- Modify: `fusion_mlx/engines/video_backends/base.py` (abstract `denoise`)

**Interfaces:**
- Consumes: Task 1 `apply_inpaint_mask` (signature only — engine/base don't call it, backends do).
- Produces: `VideoGenEngine.denoise(..., inpaint_mask=None, init_latent=None)` and `VideoBackend.denoise(..., inpaint_mask=None, init_latent=None)` — backward-compatible defaults.

<!-- Task 2 steps: fill -->

---

## Task 3: Surface C Wan2 — run_denoise mask insertion

**Files:**
- Modify: `fusion_mlx/video/wan2/stage.py` (`run_denoise`, insert after line 497)
- Modify: `fusion_mlx/engines/video_backends/wan2.py` (`Wan2Backend.denoise` threading)

**Interfaces:**
- Consumes: Task 1 `apply_inpaint_mask`; Task 2 `denoise(..., inpaint_mask=, init_latent=)`.
- Produces: Wan2 denoise loop re-composites frozen region per step.

<!-- Task 3 steps: fill -->

---

## Task 4: Surface C SkyReels — _denoise_sample mask insertion

**Files:**
- Modify: `fusion_mlx/video/skyreels_v3/pipelines/__init__.py` (`_denoise_sample`, insert after line 743)
- Modify: `fusion_mlx/engines/video_backends/skyreels.py` (`SkyReelsBackend.denoise` threading)

**Interfaces:**
- Consumes: Task 1 `apply_inpaint_mask`; Task 2 `denoise(..., inpaint_mask=, init_latent=)`.
- Produces: SkyReels denoise loop re-composites frozen region per step.

<!-- Task 4 steps: fill -->

---

## Task 5: Surface B Wan2 — ControlState adapter fields + DiT residual injection (RULING R1/R2, riskiest)

**Files:**
- Modify: `fusion_mlx/video/wan2/stage.py` (`ControlState` dataclass + `run_denoise` residual threading)
- Modify: `fusion_mlx/video/wan2/wan_2.py` (`WanModel.__call__` gains `controlnet_residuals`/`controlnet_stride` + block-loop injection)

**Interfaces:**
- Consumes: `fusion_mlx/video/adapters/controlnet.py` `ControlNet.compute_residuals(hidden_states, t, context, control_states, seq_lens=, grid_sizes=) -> list[mx.array]`.
- Produces: `ControlState.controlnet_adapter: ControlNet | None`, `ControlState.controlnet_latent: mx.array | None`; `WanModel.__call__(..., controlnet_residuals=None, controlnet_stride=4)`.

<!-- Task 5 steps: fill -->

---

## Task 6: Surface B Wan2 — encode_control controlnet_image path

**Files:**
- Modify: `fusion_mlx/engines/video_backends/wan2.py` (`Wan2Backend.encode_control`)
- Modify: `fusion_mlx/video/adapters/controlnet.py` (pin contract via tests; prod change only if signature gap)

**Interfaces:**
- Consumes: Task 5 `ControlState.controlnet_adapter`/`controlnet_latent`; `ControlNet(scale=, image=, config={})`, `.load()`, `.encode_control(image_path, control_type)`.
- Produces: `Wan2Backend.encode_control(controlnet_image=, control_type=, controlnet_strength=)` builds adapter + control latent, stores on ControlState; routes ControlNet path ONLY when `controlnet_image` set and NOT VACE/i2v/camera (Rule 7 single-path).

<!-- Task 6 steps: fill -->

---

## Task 7: Surface B SkyReels — encode_control config plumbing

**Files:**
- Modify: `fusion_mlx/engines/video_backends/skyreels.py` (`SkyReelsBackend.encode_control`)

**Interfaces:**
- Consumes: SkyReels `pipeline.config` (read by `_denoise_sample` at pipelines/__init__.py:607-609); NO ControlState on SkyReels.
- Produces: `SkyReelsBackend.encode_control(controlnet_image=, control_type=, controlnet_strength=)` sets `pipeline.config.controlnet_image`/`control_type`/`controlnet_strength`.

<!-- Task 7 steps: fill -->

---

## Task 8: Surface A — SkyReels VAE encode

**Files:**
- Modify: `fusion_mlx/engines/video_backends/skyreels.py` (`load_vae_encoder`/`encode`/`unload_vae_encoder`)
- Create: `tests/unit/test_vae_encode_skyreels.py`

**Interfaces:**
- Consumes: SkyReels family VAE `.encode()` (model layer, verified to exist).
- Produces: `SkyReelsBackend.encode(pixels) -> mx.array` (5D latent), mirroring Wan2.

<!-- Task 8 steps: fill -->

---

## Task 9: Surface A — batch the 9 denoise-less backends (RULING R3)

**Files:**
- Modify: `cogvideox.py`, `cosmos.py`, `hunyuanvideo.py`, `ltx_video_legacy.py`, `ltx2_5.py`, `ltx2.py`, `minimax_h3.py`, `opensora.py`, `svd.py`
- Create: `tests/unit/test_vae_encode_surfaces.py` (batched, fake VAE per backend)

**Interfaces:**
- Consumes: each family VAE `.encode()` (verified to exist for all); #670 full-VAE-load fallback where a family ships decoder-only.
- Produces: each backend's `load_vae_encoder`/`encode`/`unload_vae_encoder`.

<!-- Task 9 steps: fill -->

---

## Task 10: Real-model tests (gated FUSION_MLX_REAL_MODEL_TESTS)

**Files:**
- Create: `tests/unit/test_653_real_model.py`

**Interfaces:**
- Consumes: all surfaces (A encode, B ControlNet, C inpaint) on Wan2 + SkyReels.
- Produces: VAE encode roundtrip, Wan2 ControlNet e2e, Wan2 + SkyReels inpaint e2e.

<!-- Task 10 steps: fill -->

---

## Task 11: File 10 follow-up issues (denoise-port for Surface B+C on the other backends)

**Files:**
- GitHub issues (one per family): cogvideox, cosmos, hunyuanvideo, ltx_video_legacy, ltx2_5, ltx2, minimax_h3, opensora, svd, uniworld(stub).

<!-- Task 11 steps: fill -->

---

## Task 12: Lint + full-suite sweep + README/CHANGELOG + PR + merge + release

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `fusion_mlx/_version.py` (shell printf — Write/Edit no-op).

<!-- Task 12 steps: fill -->

---

## Self-review

<!-- fill after tasks complete: spec coverage, placeholder scan, type consistency -->

---

## Status / Resume Notes (handoff 2026-08-31)

This plan is a **skeleton, not execution-ready.** Resume by completing it per the writing-plans skill, then execute.

**Commit state:**
- Design spec: COMMITTED `d1c8fef` — `docs/superpowers/specs/2026-08-31-653-vae-controlnet-inpaint-surfaces-design.md`.
- This plan file: UNCOMMITTED as of handoff (committed alongside these notes — see git log for `docs(plans): ...`).
- Branch: `feat/api-stable-layer` (NOT main). Remote: `origin` (git@github.com:dahai80/fusion-mlx.git).

**What is DONE:**
- Required plan header (Goal/Architecture/Tech Stack/Spec path/Global Constraints).
- 4 Plan Rulings (R1–R4) recorded verbatim above — these are the load-bearing decisions; re-read before touching Tasks 5/9/4.
- File-structure section (Files created / Files modified per surface / Docs).
- 12 Task stubs, each with **Files** + **Interfaces (Consumes/Produces)** filled.

**What is NOT done (the resume work):**
- Every Task 1–12 body is `<!-- Task N steps: fill -->` placeholder. Per writing-plans "No Placeholders", each task needs bite-sized steps with exact code: write failing test → run to fail → implement → run to pass → commit.
- Self-review section unfilled.
- No code written, no tests, no PR, no release. v0.8.56 is the latest release; #653 lands as a future patch (likely v0.8.57).

**Gathered signatures (load-bearing, verified this session — cite, don't re-derive):**
- `WanModel.__call__` at `fusion_mlx/video/wan2/wan_2.py:331` — NO `controlnet_residuals`/`controlnet_stride` kwargs today. Block loop at `wan_2.py:560` injects only VACE `vace_hints`. Surface B Wan2 requires adding these kwargs + block-loop injection (Ruling R1).
- SkyReels DiT residual injection precedent: `fusion_mlx/video/skyreels_v3/pipelines/__init__.py:716-724` — `dit(..., controlnet_residuals=cn_residuals, controlnet_stride=cn_stride)`. Mirror this in Wan2.
- `ControlNet.compute_residuals(hidden_states, t, context, control_states, seq_lens=None, grid_sizes=None) -> list[mx.array] | None` at `fusion_mlx/video/adapters/controlnet.py`. Expects `hidden_states: [B, C_vae, H, W]` (B-first, single-frame) — Wan2 latents are 4D C-first `(z_dim, t_latent, h_latent, w_latent)`, needs reshape reconciliation (Ruling R2, riskiest task = Task 5).
- Wan2 Surface C insertion point: `fusion_mlx/video/wan2/stage.py:497` (after `sched.step`). Precedent re-composite pattern at `stage.py:499-507` (`latents = (1.0 - control.i2v_mask) * control.z_img + control.i2v_mask * latents`).
- SkyReels Surface C insertion point: `fusion_mlx/video/skyreels_v3/pipelines/__init__.py:743` (`scheduler.step(...).prev_sample`), before flicker smooth. Default path only (Ruling R4); async `_denoise_sample_async` :780 + speculative `_denoise_sample_speculative` :898/:1501 gated OFF (#177/#180) — follow-up.
- SkyReels ControlNet config read site: `pipelines/__init__.py:607-609` (pipeline.config fields, NO ControlState on SkyReels).
- Mask semantics: `apply_inpaint_mask(latents, init, mask) = mask*latents + (1-mask)*init` — mask=1 reactive/denoised, mask=0 frozen/init. Consistent with existing `i2v_mask_blend`.

**Exact resume steps:**
1. Re-read this plan + the committed spec (`docs/superpowers/specs/2026-08-31-653-vae-controlnet-inpaint-surfaces-design.md`).
2. Fill Task 1–12 bodies with exact code (TDD: failing test → fail → implement → pass → commit). No placeholders.
3. Run self-review (spec coverage / placeholder scan / type consistency) — fill the Self-review section.
4. Commit the completed plan.
5. Execute via `superpowers:subagent-driven-development`: fresh subagent per task + task review + broad final review. Rulings R1–R4 govern conflicts; the spec is the binding authority.
6. Task 11 = file 10 follow-up issues for denoise-less backends (cogvideox/cosmos/hunyuanvideo/ltx_video_legacy/ltx2_5/ltx2/minimax_h3/opensora/svd/uniworld).
7. Task 12 = lint (black --fast, ruff; NEVER debt_modules.txt) + full-suite sweep (`.venv/bin/python -m pytest`, NOT bare pytest) + README/CHANGELOG + PR → merge to main → tag THEN release. PyPI skipped (no token).
8. Real-model tests (Task 10) gated behind `FUSION_MLX_REAL_MODEL_TESTS`; start/stop fusion-mlx via `~/claude-home/fusion-mlx/start.sh start|stop`; download models via https://hf-mirror.com. Never `mx.clear_streams()` in tests (#630 stream invariant).

**Resuming session: do NOT ask the user questions to re-confirm scope.** Surface A on all 11 backends, Surface B+C on Wan2+SkyReels only — locked. The 10 follow-up issues cover the rest. Proceed straight to filling task bodies.
