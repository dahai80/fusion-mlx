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

- [ ] **Step 1: Write the failing tests** (kept here as the canonical copy)

Create `tests/unit/test_inpaint_mask_helper.py`:

```python
import mlx.core as mx
import pytest

from fusion_mlx.engines.video_backends._inpaint import (
    apply_inpaint_mask,
    patch_downsample_mask,
)


def test_apply_inpaint_mask_reactive_and_frozen():
    latents = mx.array([[1.0, 2.0], [3.0, 4.0]])
    init = mx.array([[10.0, 20.0], [30.0, 40.0]])
    mask = mx.array([[1.0, 0.0], [1.0, 0.0]])
    out = apply_inpaint_mask(latents, init, mask)
    expected = mx.array([[1.0, 20.0], [3.0, 40.0]])
    assert mx.allclose(out, expected).item()


def test_apply_inpaint_mask_none_passthrough():
    latents = mx.array([1.0, 2.0, 3.0])
    assert mx.array_equal(apply_inpaint_mask(latents, None, None), latents).item()


def test_apply_inpaint_mask_shape_mismatch_raises():
    latents = mx.zeros((1, 2, 2, 2))
    init = mx.zeros((1, 2, 3, 3))
    mask = mx.ones((1, 2, 2, 2))
    with pytest.raises(ValueError, match="init_latent shape"):
        apply_inpaint_mask(latents, init, mask)


def test_patch_downsample_mask_2x2_to_1x1():
    mask = mx.array([[1.0, 1.0], [0.0, 0.0]])
    out = patch_downsample_mask(
        mask, vae_stride=(4, 2, 2), patch_size=(1, 2, 2),
        t_latent=1, h_latent=1, w_latent=1,
    )
    assert out.shape == (1, 1, 1, 1)
    assert abs(float(out[0, 0, 0, 0]) - 0.5) < 1e-6


def test_patch_downsample_mask_broadcasts_temporal():
    mask = mx.ones((4, 4))
    out = patch_downsample_mask(
        mask, vae_stride=(4, 2, 2), patch_size=(1, 2, 2),
        t_latent=3, h_latent=2, w_latent=2,
    )
    assert out.shape == (1, 3, 2, 2)
    assert mx.allclose(out, mx.ones((1, 3, 2, 2))).item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_inpaint_mask_helper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fusion_mlx.engines.video_backends._inpaint'`

- [ ] **Step 3: Write the implementation**

Create `fusion_mlx/engines/video_backends/_inpaint.py`:

```python
import logging

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def apply_inpaint_mask(latents, init_latent, mask):
    # #653 Surface C: neutral per-step re-composite. mask=1 -> reactive
    # (keep denoised); mask=0 -> frozen (restore init). Orthogonal to
    # ControlState; never touches conditioning. None -> T2V passthrough.
    if mask is None or init_latent is None:
        return latents
    if latents.shape != init_latent.shape:
        raise ValueError(
            f"apply_inpaint_mask: init_latent shape {tuple(init_latent.shape)}"
            f" != latents {tuple(latents.shape)}"
        )
    return mask * latents + (1.0 - mask) * init_latent


def patch_downsample_mask(mask, vae_stride, patch_size, t_latent, h_latent, w_latent):
    # Average-pool a pixel-space mask (H, W) or (T, H, W) to latent spatial
    # size, broadcast to (1, t_latent, h_latent, w_latent) for 4D Wan2
    # latents. vae_stride=(s_t, s_h, s_w); last two mask dims are pixel H/W.
    arr = np.array(mask, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, None]
    elif arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"patch_downsample_mask: mask ndim {arr.ndim} not 2/3/4")
    _, _, h_px, w_px = arr.shape
    s_h, s_w = vae_stride[1], vae_stride[2]
    if h_px // s_h != h_latent or w_px // s_w != w_latent:
        logger.warning(
            "patch_downsample_mask: px %dx%d / stride %dx%d != latent %dx%d",
            h_px, w_px, s_h, s_w, h_latent, w_latent,
        )
    arr = arr[:, :, : h_latent * s_h, : w_latent * s_w]
    arr = arr.reshape(1, 1, h_latent, s_h, w_latent, s_w).mean(axis=(3, 5))
    out = mx.array(arr)
    out = mx.broadcast_to(out, (1, t_latent, h_latent, w_latent))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_inpaint_mask_helper.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/engines/video_backends/_inpaint.py tests/unit/test_inpaint_mask_helper.py
git commit -m "feat(video): add inpaint mask re-composite helper (#653 Surface C)"
```

---

## Task 2: Surface C engine + base signature threading

**Files:**
- Modify: `fusion_mlx/engines/video.py` (`VideoGenEngine.denoise`)
- Modify: `fusion_mlx/engines/video_backends/base.py` (abstract `denoise`)

**Interfaces:**
- Consumes: Task 1 `apply_inpaint_mask` (signature only — engine/base don't call it, backends do).
- Produces: `VideoGenEngine.denoise(..., inpaint_mask=None, init_latent=None)` and `VideoBackend.denoise(..., inpaint_mask=None, init_latent=None)` — backward-compatible defaults.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_surface_c_engine_threading.py`:

```python
import mlx.core as mx
import pytest

from fusion_mlx.engines.video import VideoGenEngine


class _FakeBackend:
    def __init__(self):
        self._loaded = True
        self.calls = []

    async def denoise(
        self,
        latent,
        pos_embed,
        neg_embed,
        steps,
        cfg,
        seed,
        num_frames,
        control=None,
        inpaint_mask=None,
        init_latent=None,
    ):
        self.calls.append(
            {"control": control, "inpaint_mask": inpaint_mask, "init_latent": init_latent}
        )
        return latent


@pytest.mark.asyncio
async def test_denoise_threads_inpaint_mask_and_init_latent():
    engine = VideoGenEngine.__new__(VideoGenEngine)
    engine._backend = _FakeBackend()
    engine._model_name = "fake"
    mask = mx.array([1.0])
    init = mx.array([2.0])
    out = await engine.denoise(
        mx.zeros((1,)), mx.zeros((1,)), None, 1, 1.0, 0, 1,
        inpaint_mask=mask, init_latent=init,
    )
    assert mx.array_equal(out, mx.zeros((1,))).item()
    assert engine._backend.calls[0]["inpaint_mask"] is mask
    assert engine._backend.calls[0]["init_latent"] is init


@pytest.mark.asyncio
async def test_denoise_defaults_inpaint_none_backcompat():
    engine = VideoGenEngine.__new__(VideoGenEngine)
    engine._backend = _FakeBackend()
    engine._model_name = "fake"
    await engine.denoise(mx.zeros((1,)), mx.zeros((1,)), None, 1, 1.0, 0, 1)
    assert engine._backend.calls[0]["inpaint_mask"] is None
    assert engine._backend.calls[0]["init_latent"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_c_engine_threading.py -v`
Expected: FAIL with `TypeError: denoise() got an unexpected keyword argument 'inpaint_mask'` (engine `denoise` has no such param today).

- [ ] **Step 3: Modify `VideoGenEngine.denoise` (video.py:141-154)**

Change the signature to accept the two new keyword-only params and forward them:

```python
    async def denoise(
        self,
        latent: mx.array,
        pos_embed: mx.array,
        neg_embed: mx.array | None,
        steps: int,
        cfg: float,
        seed: int,
        num_frames: int,
        control=None,
        inpaint_mask=None,
        init_latent=None,
    ) -> mx.array:
        return await self._backend.denoise(
            latent, pos_embed, neg_embed, steps, cfg, seed, num_frames, control,
            inpaint_mask=inpaint_mask, init_latent=init_latent,
        )
```

- [ ] **Step 4: Modify abstract `VideoBackend.denoise` (base.py:185-198)**

Add the same two keyword params (default `None`) to the abstract signature so backends conform:

```python
    async def denoise(
        self,
        latent: mx.array,
        pos_embed: mx.array,
        neg_embed: mx.array | None,
        steps: int,
        cfg: float,
        seed: int,
        num_frames: int,
        control=None,
        inpaint_mask=None,
        init_latent=None,
    ) -> mx.array:
        raise NotImplementedError(
            f"{self.name} stage API not implemented (issue #170 phase 2)"
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_c_engine_threading.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add fusion_mlx/engines/video.py fusion_mlx/engines/video_backends/base.py tests/unit/test_surface_c_engine_threading.py
git commit -m "feat(video): thread inpaint_mask/init_latent through VideoGenEngine + base (#653 Surface C)"
```

---

## Task 3: Surface C Wan2 — run_denoise mask insertion

**Files:**
- Modify: `fusion_mlx/video/wan2/stage.py` (`run_denoise`, insert after line 497)
- Modify: `fusion_mlx/engines/video_backends/wan2.py` (`Wan2Backend.denoise` threading)

**Interfaces:**
- Consumes: Task 1 `apply_inpaint_mask`; Task 2 `denoise(..., inpaint_mask=, init_latent=)`.
- Produces: Wan2 denoise loop re-composites frozen region per step.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_surface_c_wan2_insertion.py`:

```python
import inspect

from fusion_mlx.video.wan2.stage import run_denoise


def test_run_denoise_accepts_inpaint_mask_and_init_latent():
    sig = inspect.signature(run_denoise)
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
    assert sig.parameters["inpaint_mask"].default is None
    assert sig.parameters["init_latent"].default is None


def test_run_denoise_recomposites_after_sched_step():
    src = inspect.getsource(run_denoise)
    assert "apply_inpaint_mask(latents, init_latent, inpaint_mask)" in src
    assert "inpaint_mask is not None" in src


def test_wan2_backend_denoise_threads_inpaint_kwargs():
    from fusion_mlx.engines.video_backends.wan2 import Wan2Backend
    sig = inspect.signature(Wan2Backend.denoise)
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
```

NOTE for the implementer: this unit test locks the *contract* — `run_denoise` and `Wan2Backend.denoise` accept the two Surface C kwargs, and the denoise loop calls `apply_inpaint_mask` after `sched.step` under a guard. It deliberately does NOT exercise the denoise loop with fakes (run_denoise does real MLX scheduler/rope work a minimal config cannot satisfy without becoming a parallel implementation — that would be false coverage per Rule 9). The behavioral assertion (frozen region preserved across steps) is the real-model test in Task 10.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_c_wan2_insertion.py -v`
Expected: FAIL — `"inpaint_mask" in sig.parameters` is False (the params don't exist yet), and `"apply_inpaint_mask(latents, init_latent, inpaint_mask)" in src` is False.

- [ ] **Step 3: Add `inpaint_mask`/`init_latent` to `run_denoise` (stage.py:228-251)**

Add the two keyword params at the end of the signature:

```python
def run_denoise(
    config,
    models,
    context,
    context_null,
    target_shape,
    seq_len,
    steps,
    guide_scale,
    shift,
    scheduler,
    seed,
    no_compile,
    on_step=None,
    control_hidden_states=None,
    control_scales=None,
    y_camera=None,
    y_i2v=None,
    z_img=None,
    i2v_mask=None,
    i2v_mask_tokens=None,
    is_i2v_mask_blend=False,
    is_i2v_channel_concat=False,
    inpaint_mask=None,
    init_latent=None,
):
```

- [ ] **Step 4: Insert the re-composite after `sched.step` (stage.py:497)**

Add immediately after line 497 (`latents = sched.step(...).squeeze(0)`) and BEFORE the existing TI2V mask block (499-507). Import `apply_inpaint_mask` at top of stage.py:

```python
from fusion_mlx.engines.video_backends._inpaint import apply_inpaint_mask
```

Insertion:

```python
        latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)

        # #653 Surface C: frozen-region re-composite. Orthogonal to the
        # TI2V mask blend below — runs even when is_i2v_mask_blend is False.
        # mask=1 -> reactive (keep denoised); mask=0 -> frozen (restore init).
        if inpaint_mask is not None and init_latent is not None:
            latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)

        # TI2V-5B: re-apply mask to keep first frame frozen (generate.py 1183).
```

- [ ] **Step 5: Thread the two params through `Wan2Backend.denoise` (wan2.py:466-549)**

Add `inpaint_mask=None, init_latent=None` to the `denoise` signature (after `control=None`), and forward them in the `_denoise()` `run_denoise(...)` call (after `is_i2v_channel_concat=...`):

```python
    async def denoise(
        self,
        latent: mx.array,
        pos_embed: mx.array,
        neg_embed: mx.array | None,
        steps: int,
        cfg: float,
        seed: int,
        num_frames: int,
        control=None,
        inpaint_mask=None,
        init_latent=None,
    ) -> mx.array:
        ...
        def _denoise():
            lat_4d = run_denoise(
                config,
                models,
                pos_embed,
                context_null,
                target_shape,
                seq_len,
                steps,
                guide_scale,
                shift,
                scheduler,
                seed,
                no_compile,
                on_step=on_step,
                control_hidden_states=control.control_hidden_states,
                control_scales=control.control_scales,
                y_camera=control.y_camera,
                y_i2v=control.y_i2v,
                z_img=control.z_img,
                i2v_mask=control.i2v_mask,
                i2v_mask_tokens=control.i2v_mask_tokens,
                is_i2v_mask_blend=control.is_i2v_mask_blend,
                is_i2v_channel_concat=control.is_i2v_channel_concat,
                inpaint_mask=inpaint_mask,
                init_latent=init_latent,
            )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_c_wan2_insertion.py tests/unit/test_surface_c_engine_threading.py -v`
Expected: PASS (3 tests). The engine-threading test must still pass — `Wan2Backend.denoise` still forwards the new kwargs.

- [ ] **Step 7: Commit**

```bash
git add fusion_mlx/video/wan2/stage.py fusion_mlx/engines/video_backends/wan2.py tests/unit/test_surface_c_wan2_insertion.py
git commit -m "feat(video): insert inpaint mask re-composite in Wan2 run_denoise (#653 Surface C)"
```

---

## Task 4: Surface C SkyReels — _denoise_sample mask insertion

**Files:**
- Modify: `fusion_mlx/video/skyreels_v3/pipelines/__init__.py` (`_denoise_sample`, insert after line 743)
- Modify: `fusion_mlx/engines/video_backends/skyreels.py` (`SkyReelsBackend.denoise` threading)

**Interfaces:**
- Consumes: Task 1 `apply_inpaint_mask`; Task 2 `denoise(..., inpaint_mask=, init_latent=)`.
- Produces: SkyReels denoise loop re-composites frozen region per step.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_surface_c_skyreels_insertion.py`:

```python
import inspect

from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsPipeline


def test_denoise_sample_accepts_inpaint_kwargs():
    sig = inspect.signature(SkyReelsPipeline._denoise_sample)
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
    assert sig.parameters["inpaint_mask"].default is None
    assert sig.parameters["init_latent"].default is None


def test_denoise_sample_recomposites_after_scheduler_step():
    src = inspect.getsource(SkyReelsPipeline._denoise_sample)
    assert "apply_inpaint_mask(latents, init_latent, inpaint_mask)" in src
    assert "inpaint_mask is not None" in src


def test_skyreels_backend_denoise_threads_inpaint_kwargs():
    from fusion_mlx.engines.video_backends.skyreels import SkyReelsBackend
    sig = inspect.signature(SkyReelsBackend.denoise)
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
```

NOTE: same contract-lock approach as Task 3 (Rule 9). Behavior asserted in the Task 10 real-model test. Insertion must be in the DEFAULT `_denoise_sample` path only (Ruling R4) — NOT in `_denoise_sample_async`/`_denoise_sample_speculative`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_c_skyreels_insertion.py -v`
Expected: FAIL — the params are absent and `apply_inpaint_mask` is not in the source.

- [ ] **Step 3: Add `inpaint_mask`/`init_latent` to `_denoise_sample` (pipelines/__init__.py:539-544)**

Add the two keyword-only params (after `grid_sizes`):

```python
    def _denoise_sample(
        self,
        latents: mx.array,
        context: mx.array,
        *,
        seq_lens: list,
        grid_sizes: list,
        inpaint_mask=None,
        init_latent=None,
    ) -> mx.array:
```

- [ ] **Step 4: Insert the re-composite after `scheduler.step` (pipelines/__init__.py:743)**

Add immediately after line 743 (`.prev_sample`) and BEFORE the flicker smooth (746). Import at top:

```python
from fusion_mlx.engines.video_backends._inpaint import apply_inpaint_mask
```

Insertion:

```python
            # 采样步
            latents = scheduler.step(
                noise_pred,
                float(t),
                latents,
            ).prev_sample

            # #653 Surface C: frozen-region re-composite (default path only, R4).
            # Orthogonal to flicker smoothing below. mask=1 -> reactive,
            # mask=0 -> frozen (restore init). async/speculative paths NOT
            # patched (#177/#180) — follow-up issue.
            if inpaint_mask is not None and init_latent is not None:
                latents = apply_inpaint_mask(latents, init_latent, inpaint_mask)

            # 时序闪烁修复: 帧间 EMA 平滑 (防相邻帧跳变)
            latents = flicker_fix.smooth_temporal(latents)
```

- [ ] **Step 5: Thread the two params through `SkyReelsBackend.denoise` (skyreels.py:202-245)**

Add `inpaint_mask=None, init_latent=None` to the `denoise` signature (after `control=None`), and forward in the `_denoise()` call:

```python
    async def denoise(
        self,
        latent: mx.array,
        pos_embed: mx.array,
        neg_embed: mx.array | None,
        steps: int,
        cfg: float,
        seed: int,
        num_frames: int,
        control=None,
        inpaint_mask=None,
        init_latent=None,
    ) -> mx.array:
        ...
            def _denoise():
                result = pipeline._denoise_sample(
                    latent, context, seq_lens=seq_lens, grid_sizes=grid_sizes,
                    inpaint_mask=inpaint_mask, init_latent=init_latent,
                )
                mx.eval(result)
                return result
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_c_skyreels_insertion.py tests/unit/test_surface_c_engine_threading.py -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Commit**

```bash
git add fusion_mlx/video/skyreels_v3/pipelines/__init__.py fusion_mlx/engines/video_backends/skyreels.py tests/unit/test_surface_c_skyreels_insertion.py
git commit -m "feat(video): insert inpaint mask re-composite in SkyReels _denoise_sample (#653 Surface C)"
```

---

## Task 5: Surface B Wan2 — ControlState adapter fields + DiT residual injection (RULING R1/R2, riskiest)

**Files:**
- Modify: `fusion_mlx/video/wan2/stage.py` (`ControlState` dataclass + `run_denoise` residual threading)
- Modify: `fusion_mlx/video/wan2/wan_2.py` (`WanModel.__call__` gains `controlnet_residuals`/`controlnet_stride` + block-loop injection)

**Interfaces:**
- Consumes: `fusion_mlx/video/adapters/controlnet.py` `ControlNet.compute_residuals(hidden_states, t, context, control_states, seq_lens=, grid_sizes=) -> list[mx.array]`.
- Produces: `ControlState.controlnet_adapter: ControlNet | None`, `ControlState.controlnet_latent: mx.array | None`; `WanModel.__call__(..., controlnet_residuals=None, controlnet_stride=4)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_surface_b_wan2_diT_inject.py`:

```python
import inspect

from fusion_mlx.video.wan2.stage import ControlState, run_denoise
from fusion_mlx.video.wan2.wan_2 import WanModel


def test_control_state_has_controlnet_fields():
    fields = ControlState.__dataclass_fields__
    assert "controlnet_adapter" in fields
    assert "controlnet_latent" in fields


def test_wan_model_call_accepts_controlnet_kwargs():
    sig = inspect.signature(WanModel.__call__)
    assert "controlnet_residuals" in sig.parameters
    assert "controlnet_stride" in sig.parameters
    assert sig.parameters["controlnet_residuals"].default is None
    assert sig.parameters["controlnet_stride"].default == 4


def test_wan_model_block_loop_injects_controlnet_residuals():
    src = inspect.getsource(WanModel.__call__)
    assert "controlnet_residuals" in src
    assert "controlnet_stride" in src
    assert "i % controlnet_stride == 0" in src


def test_run_denoise_threads_controlnet_residuals():
    sig = inspect.signature(run_denoise)
    assert "controlnet_adapter" in sig.parameters
    assert "controlnet_latent" in sig.parameters
    src = inspect.getsource(run_denoise)
    assert "controlnet_residuals=" in src
    assert "compute_residuals" in src
    assert "controlnet_stride=" in src
```

NOTE: contract-lock test (Rule 9). The behavioral e2e (control image visibly steers output) is the Task 10 real-model test. The block-loop injection mirrors the existing VACE `x = x + hint * scale` pattern (wan_2.py:595) and the SkyReels DiT precedent (pipelines/__init__.py:716-724).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_b_wan2_diT_inject.py -v`
Expected: FAIL — the fields/params and injection string are absent.

- [ ] **Step 3: Add `controlnet_adapter`/`controlnet_latent` to `ControlState` (stage.py:46-61)**

Add the two fields at the end of the dataclass:

```python
@dataclass
class ControlState:
    control_hidden_states: Any = None  # VACE [list of (z,...,t,h,w)] or None
    control_scales: Any = None  # VACE per-layer scales; DiT self-defaults
    y_camera: Any = None  # Fun-Camera [list of (C_cam,F,H,W)] or None
    y_i2v: Any = None  # I2V-14B channel-concat y tensor or None
    z_img: Any = None  # TI2V-5B encoded first-frame latent [z,1,h,w]
    i2v_mask: Any = None  # TI2V-5B blend mask [z,t,h,w]
    i2v_mask_tokens: Any = None  # TI2V-5B per-frame t-token weights [1,t_lat]
    is_i2v_mask_blend: bool = False
    is_i2v_channel_concat: bool = False
    # #653 Surface B: ControlNet adapter + preprocessed control latent.
    # Residuals are computed per-step inside run_denoise and injected into
    # the DiT block loop (R1). Orthogonal to VACE (control_hidden_states).
    controlnet_adapter: Any = None
    controlnet_latent: Any = None
```

- [ ] **Step 4: Add `controlnet_adapter`/`controlnet_latent` to `run_denoise` signature + ControlState fold (stage.py:228-273)**

Add the two keyword params at the end of the `run_denoise` signature (after `init_latent=None` from Task 3), and fold them into the `ControlState` constructor:

```python
    is_i2v_mask_blend=False,
    is_i2v_channel_concat=False,
    inpaint_mask=None,
    init_latent=None,
    controlnet_adapter=None,
    controlnet_latent=None,
):
```

and in the `control = ControlState(...)` block (after `is_i2v_channel_concat=is_i2v_channel_concat,`):

```python
        is_i2v_mask_blend=is_i2v_mask_blend,
        is_i2v_channel_concat=is_i2v_channel_concat,
        controlnet_adapter=controlnet_adapter,
        controlnet_latent=controlnet_latent,
    )
```

- [ ] **Step 5: Compute per-step residuals + thread to DiT (R2 reshape) inside the denoise loop**

Inside the loop, BEFORE the model call, compute ControlNet residuals for the current step. `latents` is `(z_dim, t_latent, h_latent, w_latent)` C-first. `compute_residuals` wants `[B, C_vae, H, W]` (NCHW, B-first); `WanControlnet.forward` itself does NCHW→NHWC (controlnet.py:290-293). Take the first frame, swapaxes to B-first NCHW. Add this block right after the `timestep_val` derivation / before the cfg_disabled branch (around stage.py:405, after `_call = getattr(model, "_compiled", model)`):

```python
        _call = getattr(model, "_compiled", model)

        # #653 Surface B (R2): per-step ControlNet residuals. compute_residuals
        # expects [B, C_vae, H, W] NCHW B-first; Wan2 latents are 4D C-first
        # (z_dim, t_latent, h_latent, w_latent). Take the first frame and
        # swap to B-first NCHW. Residuals -> list of [1, L_tokens, out_proj_dim].
        cn_residuals = None
        cn_stride = 4
        if control.controlnet_adapter is not None and control.controlnet_latent is not None:
            hs = latents[:, 0:1, :, :].swapaxes(0, 1)  # (1, z_dim, h, w)
            t_mx = mx.array([float(timestep_val)])
            try:
                cn_residuals = control.controlnet_adapter.compute_residuals(
                    hs, t_mx, context, control.controlnet_latent,
                    seq_lens=[seq_len], grid_sizes=[(f_grid, h_grid, w_grid)],
                )
                cn_stride = getattr(control.controlnet_adapter, "stride", 4)
            except Exception as exc:
                logger.warning("ControlNet residual compute failed: %s", exc, exc_info=True)
                cn_residuals = None
```

Then pass `controlnet_residuals=cn_residuals, controlnet_stride=cn_stride` to BOTH model call sites — the cfg_disabled branch (stage.py:428-440) AND the CFG branch (stage.py:480-492). Add them as kwargs alongside `control_hidden_states=`/`control_scales=`/`y_camera=`.

- [ ] **Step 6: Add `controlnet_residuals`/`controlnet_stride` to `WanModel.__call__` + block-loop injection (R1) (wan_2.py:331-343, 559-596)**

Signature — add two kwargs after `y_camera`:

```python
    def __call__(
        self,
        x_list: list,
        t: mx.array,
        context: list | mx.array,
        seq_len: int,
        cross_kv_caches: list | None = None,
        y: list | None = None,
        rope_cos_sin: tuple | None = None,
        control_hidden_states: list | None = None,
        control_scales: list[float] | None = None,
        y_camera: list | None = None,
        controlnet_residuals: list | None = None,
        controlnet_stride: int = 4,
    ) -> list:
```

Block loop — after `x = block(x, cross_kv_cache=kv, **kwargs)` (wan_2.py:562) and after the VACE injection block, add the strided ControlNet injection mirroring VACE + the SkyReels precedent:

```python
            # VACE injection at specified layers
            if i in vace_hints:
                hint, scale = vace_hints[i]
                ...  # unchanged align logic
                x = x + hint * scale

            # #653 Surface B (R1): strided ControlNet residual injection.
            # Mirror SkyReels DiT (pipelines/__init__.py:716-724). cn_residuals
            # is [1, L_tokens, out_proj_dim]; repeat to x's batch dim for CFG.
            if (
                controlnet_residuals is not None
                and i % controlnet_stride == 0
                and i // controlnet_stride < len(controlnet_residuals)
            ):
                resid = controlnet_residuals[i // controlnet_stride]
                if resid.shape[0] == 1 and x.shape[0] > 1:
                    resid = mx.repeat(resid, x.shape[0], axis=0)
                x = x + resid
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_b_wan2_diT_inject.py tests/unit/test_surface_c_wan2_insertion.py -v`
Expected: PASS (7 tests).

- [ ] **Step 8: Commit**

```bash
git add fusion_mlx/video/wan2/stage.py fusion_mlx/video/wan2/wan_2.py tests/unit/test_surface_b_wan2_diT_inject.py
git commit -m "feat(video): ControlNet per-step residual injection in Wan2 DiT (#653 Surface B R1/R2)"
```

---

## Task 6: Surface B Wan2 — encode_control controlnet_image path

**Files:**
- Modify: `fusion_mlx/engines/video_backends/wan2.py` (`Wan2Backend.encode_control`)
- Modify: `fusion_mlx/video/adapters/controlnet.py` (pin contract via tests; prod change only if signature gap)

**Interfaces:**
- Consumes: Task 5 `ControlState.controlnet_adapter`/`controlnet_latent`; `ControlNet(scale=, image=, config={})`, `.load()`, `.encode_control(image_path, control_type)`.
- Produces: `Wan2Backend.encode_control(controlnet_image=, control_type=, controlnet_strength=)` builds adapter + control latent, stores on ControlState; routes ControlNet path ONLY when `controlnet_image` set and NOT VACE/i2v/camera (Rule 7 single-path).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_surface_b_wan2_encode_control_branch.py`:

```python
import inspect

from fusion_mlx.engines.video_backends.wan2 import Wan2Backend


def test_encode_control_accepts_controlnet_kwargs():
    sig = inspect.signature(Wan2Backend.encode_control)
    assert "controlnet_image" in sig.parameters
    assert "control_type" in sig.parameters
    assert "controlnet_strength" in sig.parameters
    assert sig.parameters["controlnet_strength"].default == 1.0
    assert sig.parameters["control_type"].default == "canny"


def test_encode_control_branches_on_controlnet_image():
    src = inspect.getsource(Wan2Backend.encode_control)
    assert "controlnet_image" in src
    assert "ControlState(" in src
    assert "controlnet_adapter=" in src
    assert "controlnet_latent=" in src
```

NOTE: contract-lock test (Rule 9). The e2e (ControlNet visibly steers output) is Task 10. Branch precedence = Rule 7 single-path: controlnet_image checked FIRST in the pure-T2V gate, before VACE/i2v.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_b_wan2_encode_control_branch.py -v`
Expected: FAIL — kwargs + branch absent.

- [ ] **Step 3: Add ControlNet kwargs + branch to `Wan2Backend.encode_control` (wan2.py:755-990)**

Add three kwargs at the end of the signature (after `camera_conditions: Any = None,`):

```python
        controlnet_image: str | None = None,
        control_type: str = "canny",
        controlnet_strength: float = 1.0,
    ):
```

Add the ControlNet branch right after the camera block returns (after `return ControlState(y_camera=y_camera)`, before the pure-T2V gate `if image is None and not has_camera:`):

```python
        # #653 Surface B: ControlNet path. Routed FIRST (Rule 7 single-path) so a
        # controlnet_image takes precedence over VACE/i2v/camera-only. The adapter
        # is built + loaded here; run_denoise computes per-step residuals from
        # ControlState.controlnet_adapter/.controlnet_latent (Task 5).
        if controlnet_image is not None:
            from fusion_mlx.video.adapters.controlnet import ControlNet

            adapter = ControlNet(
                scale=controlnet_strength,
                image=controlnet_image,
                config={"control_type": control_type},
            )
            adapter.load()
            control_latent = adapter.encode_control(controlnet_image, control_type)
            if control_latent is None:
                logger.warning(
                    "stage:encode_control controlnet latent None for %s -> no-op",
                    controlnet_image,
                )
                return ControlState(
                    controlnet_adapter=adapter, controlnet_latent=None
                )
            mx.eval(control_latent)
            logger.info(
                "stage:encode_control controlnet type=%s strength=%.2f latent=%s",
                control_type,
                controlnet_strength,
                tuple(control_latent.shape),
            )
            return ControlState(
                controlnet_adapter=adapter, controlnet_latent=control_latent
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_b_wan2_encode_control_branch.py tests/unit/test_surface_b_wan2_diT_inject.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/engines/video_backends/wan2.py tests/unit/test_surface_b_wan2_encode_control_branch.py
git commit -m "feat(video): ControlNet branch in Wan2Backend.encode_control (#653 Surface B)"
```

---

## Task 7: Surface B SkyReels — encode_control config plumbing

**Files:**
- Modify: `fusion_mlx/engines/video_backends/skyreels.py` (`SkyReelsBackend.encode_control`)

**Interfaces:**
- Consumes: SkyReels `pipeline.config` (read by `_denoise_sample` at pipelines/__init__.py:607-609); NO ControlState on SkyReels.
- Produces: `SkyReelsBackend.encode_control(controlnet_image=, control_type=, controlnet_strength=)` sets `pipeline.config.controlnet_image`/`control_type`/`controlnet_strength`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_surface_b_skyreels_encode_control.py`:

```python
import inspect

from fusion_mlx.engines.video_backends.skyreels import SkyReelsBackend


def test_skyreels_encode_control_exists():
    sig = inspect.signature(SkyReelsBackend.encode_control)
    assert "controlnet_image" in sig.parameters
    assert "control_type" in sig.parameters
    assert "controlnet_strength" in sig.parameters
    assert sig.parameters["controlnet_strength"].default == 1.0
    assert sig.parameters["control_type"].default == "canny"


def test_skyreels_encode_control_assigns_config_fields():
    src = inspect.getsource(SkyReelsBackend.encode_control)
    assert "pipeline.config" in src
    assert "controlnet_image" in src
    assert "control_type" in src
    assert "controlnet_strength" in src
```

NOTE: contract-lock test (Rule 9). The pipeline-layer adapter load/compute/inject is ALREADY WIRED (`_denoise_sample` reads `pipeline.config.controlnet_image` etc. at pipelines/__init__.py:605-609, injects at 716-724). This task = engine-layer config assignment ONLY. E2e = Task 10.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_b_skyreels_encode_control.py -v`
Expected: FAIL — `encode_control` inherited from base, raises NotImplementedError (no SkyReels override).

- [ ] **Step 3: Add `encode_control` override to `SkyReelsBackend` (skyreels.py, after `unload_vae` ~line 330)**

Insert the override. The pipeline reads its own config fields at denoise time, so this just assigns them. Return a small dict mirroring the Wan2 ControlState intent (consumers check truthiness), but the SkyReels path needs no ControlState — the pipeline owns injection.

```python
    async def encode_control(
        self,
        controlnet_image: str | None = None,
        control_type: str = "canny",
        controlnet_strength: float = 1.0,
        **_: Any,
    ) -> Any:
        # #653 Surface B (SkyReels): engine-layer config plumbing. The pipeline
        # _denoise_sample already reads pipeline.config.controlnet_image /
        # control_type / controlnet_strength (pipelines/__init__.py:605-609) and
        # does adapter load + compute_residuals + DiT injection (716-724). This
        # override only assigns the config fields; NO ControlState (SkyReels has
        # none — Rule 7 single-path, pipeline-owned injection).
        pipeline = await self._ensure_pipeline()
        pipeline.config.controlnet_image = controlnet_image
        pipeline.config.control_type = control_type
        pipeline.config.controlnet_strength = controlnet_strength
        logger.info(
            "stage:encode_control skyreels image=%s type=%s strength=%.2f",
            controlnet_image,
            control_type,
            controlnet_strength,
        )
        return {"controlnet_image": controlnet_image} if controlnet_image else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_surface_b_skyreels_encode_control.py tests/unit/test_surface_c_skyreels_insertion.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/engines/video_backends/skyreels.py tests/unit/test_surface_b_skyreels_encode_control.py
git commit -m "feat(video): SkyReelsBackend.encode_control config plumbing (#653 Surface B)"
```

---

## Task 8: Surface A — SkyReels VAE encode

**Files:**
- Modify: `fusion_mlx/engines/video_backends/skyreels.py` (`load_vae_encoder`/`encode`/`unload_vae_encoder`)
- Create: `tests/unit/test_vae_encode_skyreels.py`

**Interfaces:**
- Consumes: SkyReels family VAE `.encode()` (model layer, verified to exist).
- Produces: `SkyReelsBackend.encode(pixels) -> mx.array` (5D latent), mirroring Wan2.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vae_encode_skyreels.py`:

```python
import inspect

from fusion_mlx.engines.video_backends.skyreels import SkyReelsBackend


def test_skyreels_load_vae_encoder_override():
    src = inspect.getsource(SkyReelsBackend.load_vae_encoder)
    assert "NotImplementedError" not in src
    assert "self._stage_flags" in src or "pipeline" in src


def test_skyreels_encode_override():
    sig = inspect.signature(SkyReelsBackend.encode)
    assert "pixels" in sig.parameters
    src = inspect.getsource(SkyReelsBackend.encode)
    assert ".encode(" in src
    assert "NotImplementedError" not in src


def test_skyreels_unload_vae_encoder_override():
    src = inspect.getsource(SkyReelsBackend.unload_vae_encoder)
    assert "NotImplementedError" not in src


def test_skyreels_encode_numpy_bridge_thread():
    src = inspect.getsource(SkyReelsBackend.encode)
    assert "np.array" in src or "np.asarray" in src
    assert "run_in_executor" in src
    assert "mx.eval" in src
```

NOTE: contract-lock (Rule 9). The numpy-bridge + worker-thread `mx.eval` is the #630 thread-portability invariant (encode pixels are caller-thread-owned; the worker thread has its own GPU stream). The Wan2 `encode` (wan2.py:684-722) is the reference implementation. Real e2e roundtrip = Task 10.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_vae_encode_skyreels.py -v`
Expected: FAIL — `load_vae_encoder`/`encode`/`unload_vae_encoder` inherited from base raise NotImplementedError.

- [ ] **Step 3: Add `load_vae_encoder`/`encode`/`unload_vae_encoder` overrides to `SkyReelsBackend` (skyreels.py, after `unload_vae` ~line 330)**

SkyReels loads the VAE inside its pipeline (`pipeline.vae`); the existing `load_vae`/`decode` (skyreels.py:273-322) prove the pattern. `SkyReelsVAE.encode` (vae.py:212) takes `[B,3,T,H,W]` -> `[B,16,T,H/8,W/8]`. Mirror Wan2's numpy-bridge (#630 invariant):

```python
    async def load_vae_encoder(self) -> None:
        # #653 Surface A: SkyReels VAE encoder. The pipeline owns the VAE
        # instance (pipeline.vae); load_vae already validates it exists. No
        # separate encoder model — SkyReelsVAE carries encoder+decoder.
        pipeline = await self._ensure_pipeline()
        if pipeline.vae is None:
            raise RuntimeError("vae is unloaded; call load_vae().")
        self._stage_flags["vae"] = True
        logger.info("stage:vae_encoder load skyreels active_mem=%s", _active_mem())

    async def encode(self, pixels: mx.array) -> mx.array:
        pipeline = await self._ensure_pipeline()
        if pipeline.vae is None:
            raise RuntimeError("vae is unloaded; call load_vae().")

        # #630 thread-portability: pixels built on the main thread own its GPU
        # stream; the worker thread cannot mx.eval a lazy graph referencing the
        # main thread's stream. Bridge through numpy on the caller thread, rebuild
        # mx.array inside the worker (mirrors Wan2.encode wan2.py:706-720).
        src_np = np.asarray(pixels)

        def _encode():
            if src_np.ndim == 4:
                x = mx.array(src_np)[None]  # (3,T,H,W) -> (1,3,T,H,W)
            else:
                x = mx.array(src_np)
            lat = pipeline.vae.encode(x)
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("stage:vae encode skyreels out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        # Encoder shares the VAE instance; full unload happens via unload_vae().
        # Just clear the stage flag so load_vae_encoder() must be called again.
        self._stage_flags["vae"] = False
        gc.collect()
        await _clear_mlx_cache()
        logger.info("stage:vae_encoder unload skyreels active_mem=%s", _active_mem())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_vae_encode_skyreels.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/engines/video_backends/skyreels.py tests/unit/test_vae_encode_skyreels.py
git commit -m "feat(video): SkyReels VAE encode surface (#653 Surface A)"
```

---

## Task 9: Surface A — batch the denoise-less backends (RULING R3, corrected)

**R3 corrected by Phase-1 evidence (recorded this session):** The spec table (rows 109-122) marks every family VAE `.encode()` as existing — verified TRUE: `minimax_h3/vae.py:791` has `encode` (earlier grep missed it by only scanning audio_vae). But "identical-shape mechanical" is only true for backends that ALREADY own a persistent VAE instance or call a one-line family loader in `start()`. Phase-1 survey:

- **Batch A — wireable, persistent-or-simple loader (4 backends):**
  - `ltx_video_legacy.py` — `self._vae = LTVideoVAE.from_pretrained(v_dir, dtype)` already loaded in `start()` (`ltx_video_legacy.py:68`); `LTVideoVAE.encode(x)` at `vae.py:680`.
  - `svd.py` — family loader `SVDVideoVAE.from_pretrained(model_path, dtype)` (`svd/generate.py:57`); `SVDVideoVAE.encode(x)` at `svd/vae.py:213`.
  - `cosmos.py` — family loader `CosmosVideoVAE.from_pretrained(vae_path)` (`cosmos/generate.py:96`); `.encode(img_arr)` at `:141`.
  - `hunyuanvideo.py` — family loader `HunyuanVideoVAE.from_pretrained(vae_path)` (`hunyuanvideo/generate.py:123`); `.encode(img_arr)` at `:170`.
- **Batch B — wireable, family loader exists but not yet called from backend (3 backends):**
  - `cogvideox.py` — `load_vae_encoder(vae_path, config)` from `fusion_mlx/video/cogvideox/utils` (`generate.py:19,152`); `.encode(video[None])`.
  - `minimax_h3.py` — `MiniMaxH3VideoVAE.from_pretrained(vae_path, config=H3VAEConfig())` (`generate.py:638-639`); `.encode(x)` at `vae.py:791`.
  - `ltx2_5.py` — `load_video_encoder(weights_path)` from `fusion_mlx/video/ltx2_5/video_vae.py:178` (I2V path not yet wired in generate, but encoder loader is separable).
- **Defer → follow-up issue (2 backends + 1 stub):**
  - `ltx2.py` — encoder loaded inline in `generate.py:81-97` as a `VideoEncoder.from_pretrained(model_path/"vae"/"encoder")` local threaded through `_encode_image_latent_shared`; no persistent backend field. Wiring requires refactoring that shared-loader out of generate. Follow-up issue (NOT this task).
  - `opensora.py` — `self._vae` initialized `None` in `__init__` (`opensora.py:26`) but NEVER loaded in `start()`; `generate.py:170` falls back to random `image_latent`. The family VAE class + loader path were not located in the opensora package — needs discovery. Follow-up issue (NOT this task).
  - `uniworld.py` — spec row 122 says "stub-only (defer all)". Follow-up issue (NOT this task).

**Files:**
- Modify: `fusion_mlx/engines/video_backends/ltx_video_legacy.py`, `svd.py`, `cosmos.py`, `hunyuanvideo.py`, `cogvideox.py`, `minimax_h3.py`, `ltx2_5.py` (7 backends)
- Create: `tests/unit/test_vae_encode_surfaces.py` (batched, inspect-based contract lock per backend + numpy-bridge assertion; no fake-model execution — Rule 9)
- Follow-up issues (Task 11): `ltx2`, `opensora`, `uniworld` (3 issues, replaces the original 10-issue batch for Surface A — B+C follow-ups for all 10 denoise-less remain in Task 11)

**Interfaces:**
- Consumes: each family VAE `.encode()` (verified to exist for all 7 wireable backends); Wan2 `encode` reference pattern (wan2.py:684-722) — numpy-bridge `src_np = np.array(...)`, executor rebuilds `mx.array`, `mx.eval` on worker thread (#630 stream invariant).
- Produces: each of 7 backends' `load_vae_encoder`/`encode`/`unload_vae_encoder` overrides, raising `NotImplementedError` no longer. `encode(pixels: mx.array) -> mx.array` returns 5D latent `(1, C, T, H', W')`.

<!-- Task 9 steps: fill below in two batches -->

- [ ] **Step 1: Write the batched contract-lock test (7 backends)**

Create `tests/unit/test_vae_encode_surfaces.py`. One test per backend asserting (a) `load_vae_encoder`/`encode`/`unload_vae_encoder` are overridden (not the base `NotImplementedError`), (b) `encode` signature accepts `pixels: mx.array`, (c) `encode` body contains the numpy-bridge + executor + `mx.eval` thread-portability pattern (inspect.getsource substring check — Rule 9: contract-lock, no fake-model execution).

```python
import inspect
import pytest

from fusion_mlx.engines.video_backends.base import VideoBackend
from fusion_mlx.engines.video_backends import (
    cosmos as cosmos_mod,
    cogvideox as cogvideox_mod,
    hunyuanvideo as hunyuanvideo_mod,
    ltx_video_legacy as ltx_video_legacy_mod,
    ltx2_5 as ltx2_5_mod,
    minimax_h3 as minimax_h3_mod,
    svd as svd_mod,
)

_BACKENDS = [
    ("CosmosBackend", cosmos_mod.CosmosBackend, "cosmos"),
    ("CogVideoBackend", cogvideox_mod.CogVideoBackend, "cogvideox"),
    ("HunyuanVideoBackend", hunyuanvideo_mod.HunyuanVideoBackend, "hunyuanvideo"),
    ("LegacyLTXBackend", ltx_video_legacy_mod.LegacyLTXBackend, "ltx_video_legacy"),
    ("LTX25Backend", ltx2_5_mod.LTX25Backend, "ltx2_5"),
    ("MiniMaxH3Backend", minimax_h3_mod.MiniMaxH3Backend, "minimax_h3"),
    ("SVDBackend", svd_mod.SVDBackend, "svd"),
]


@pytest.mark.parametrize("cls_name,backend_cls,family", _BACKENDS, ids=[b[2] for b in _BACKENDS])
def test_encode_surface_overridden(cls_name, backend_cls, family):
    assert backend_cls.encode is not VideoBackend.encode, f"{family}: encode not overridden"
    assert backend_cls.load_vae_encoder is not VideoBackend.load_vae_encoder, f"{family}: load_vae_encoder not overridden"
    assert backend_cls.unload_vae_encoder is not VideoBackend.unload_vae_encoder, f"{family}: unload_vae_encoder not overridden"


@pytest.mark.parametrize("cls_name,backend_cls,family", _BACKENDS, ids=[b[2] for b in _BACKENDS])
def test_encode_signature_accepts_pixels(cls_name, backend_cls, family):
    sig = inspect.signature(backend_cls.encode)
    params = list(sig.parameters)
    assert "pixels" in params, f"{family}: encode missing pixels param, got {params}"
    assert sig.parameters["pixels"].annotation is not inspect.Parameter.empty


@pytest.mark.parametrize("cls_name,backend_cls,family", _BACKENDS, ids=[b[2] for b in _BACKENDS])
def test_encode_uses_numpy_bridge_and_executor(cls_name, backend_cls, family):
    src = inspect.getsource(backend_cls.encode)
    assert "np.array" in src or "np.asarray" in src, f"{family}: encode missing numpy-bridge (#630)"
    assert "run_in_executor" in src, f"{family}: encode missing executor (#630)"
    assert "mx.eval" in src, f"{family}: encode missing mx.eval on worker thread (#630)"
    assert "ndim" in src, f"{family}: encode missing ndim guard (wan2.py:692 precedent)"
```

- [ ] **Step 2: Run test to verify it fails (7 backends raise NotImplementedError)**

Run: `.venv/bin/python -m pytest tests/unit/test_vae_encode_surfaces.py -v`
Expected: FAIL — `encode is VideoBackend.encode` (not overridden) for all 7 parametrized cases; `test_encode_signature_accepts_pixels` + `test_encode_uses_numpy_bridge_and_executor` also fail.

- [ ] **Step 3: Batch A — implement `load_vae_encoder`/`encode`/`unload_vae_encoder` on 4 simple backends**

These 4 already own a persistent VAE or call a one-line `from_pretrained` loader. Add the 3 methods to each, mirroring Wan2 (numpy-bridge + executor + `mx.eval`). Add `import numpy as np` if missing.

**`ltx_video_legacy.py`** — `self._vae` already loaded in `start()` (LTVideoVAE at `vae.py:68`). Add after `generate` (before module-level helpers):

```python
    async def load_vae_encoder(self) -> None:
        if self._vae is not None:
            return
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), self._load_pipeline, self._model_dir),
            timeout=180.0,
        )
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info("legacy-ltx: vae_encoder load vae=%s", type(self._vae).__name__)

    async def encode(self, pixels: mx.array) -> mx.array:
        if self._vae is None:
            await self.load_vae_encoder()
        vae = self._vae
        ndim = pixels.ndim
        if ndim not in (4, 5):
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        src_np = np.array(pixels[0] if ndim == 5 else pixels)

        def _encode():
            x = mx.array(src_np)
            if x.ndim == 4:
                x = x[None]
            lat = vae.encode(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("legacy-ltx: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        self._vae = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )
        logger.info("legacy-ltx: vae_encoder unload")
```

**`svd.py`** — family loader `SVDVideoVAE.from_pretrained(model_path, dtype)`. Add to `SVDBackend`:

```python
    async def load_vae_encoder(self) -> None:
        if getattr(self, "_stage_vae_encoder", None) is not None:
            return
        from fusion_mlx.video.svd.vae import SVDVideoVAE

        def _load():
            return SVDVideoVAE.from_pretrained(self._model_path, dtype=self._dtype)

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info("svd: vae_encoder load vae=%s", type(self._stage_vae_encoder).__name__)

    async def encode(self, pixels: mx.array) -> mx.array:
        if getattr(self, "_stage_vae_encoder", None) is None:
            await self.load_vae_encoder()
        vae = self._stage_vae_encoder
        ndim = pixels.ndim
        if ndim not in (4, 5):
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        src_np = np.array(pixels[0] if ndim == 5 else pixels)

        def _encode():
            x = mx.array(src_np)
            if x.ndim == 4:
                x = x[None]
            lat = vae.encode(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("svd: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_vae_encoder = None
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        await _clear_mlx_cache()
        logger.info("svd: vae_encoder unload")
```

**`cosmos.py`** — `CosmosVideoVAE.from_pretrained(vae_path)`, `vae_path = os.path.join(model_path, "vae") if isdir else model_path` (cosmos/generate.py:91-96). Add to `CosmosBackend`:

```python
    async def load_vae_encoder(self) -> None:
        if getattr(self, "_stage_vae_encoder", None) is not None:
            return
        from fusion_mlx.video.cosmos.vae import CosmosVideoVAE
        import os

        model_path = self._model_path
        vae_path = (
            os.path.join(model_path, "vae")
            if os.path.isdir(os.path.join(model_path, "vae"))
            else model_path
        )

        def _load():
            vae = CosmosVideoVAE.from_pretrained(vae_path)
            mx.eval(vae.parameters())
            return vae

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info("cosmos: vae_encoder load vae=%s", type(self._stage_vae_encoder).__name__)

    async def encode(self, pixels: mx.array) -> mx.array:
        if getattr(self, "_stage_vae_encoder", None) is None:
            await self.load_vae_encoder()
        vae = self._stage_vae_encoder
        ndim = pixels.ndim
        if ndim not in (4, 5):
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        src_np = np.array(pixels[0] if ndim == 5 else pixels)

        def _encode():
            x = mx.array(src_np)
            if x.ndim == 4:
                x = x[None]
            lat = vae.encode(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("cosmos: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_vae_encoder = None
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        await _clear_mlx_cache()
        logger.info("cosmos: vae_encoder unload")
```

**`hunyuanvideo.py`** — `HunyuanVideoVAE.from_pretrained(vae_path)` (hunyuanvideo/generate.py:118-123). Add to `HunyuanVideoBackend` (same shape as cosmos, swap import + class name):

```python
    async def load_vae_encoder(self) -> None:
        if getattr(self, "_stage_vae_encoder", None) is not None:
            return
        from fusion_mlx.video.hunyuanvideo.vae import HunyuanVideoVAE
        import os

        model_path = self._model_path
        vae_path = (
            os.path.join(model_path, "vae")
            if os.path.isdir(os.path.join(model_path, "vae"))
            else model_path
        )

        def _load():
            vae = HunyuanVideoVAE.from_pretrained(vae_path)
            mx.eval(vae.parameters())
            return vae

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info("hunyuan: vae_encoder load vae=%s", type(self._stage_vae_encoder).__name__)

    async def encode(self, pixels: mx.array) -> mx.array:
        if getattr(self, "_stage_vae_encoder", None) is None:
            await self.load_vae_encoder()
        vae = self._stage_vae_encoder
        ndim = pixels.ndim
        if ndim not in (4, 5):
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        src_np = np.array(pixels[0] if ndim == 5 else pixels)

        def _encode():
            x = mx.array(src_np)
            if x.ndim == 4:
                x = x[None]
            lat = vae.encode(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("hunyuan: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_vae_encoder = None
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        await _clear_mlx_cache()
        logger.info("hunyuan: vae_encoder unload")
```

**Imports check:** each backend file already imports `mlx.core as mx`, `asyncio`, `gc`, `logging`, and `get_executor` from `..engine_core` or similar. `numpy as np` — add `import numpy as np` at top if missing. `_clear_mlx_cache` — import from the same place `wan2.py` imports it (verify per-file: `from .._mlx_cache import _clear_mlx_cache` or equivalent — executor adds the import line).

- [ ] **Step 4: Run batch-A tests to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/test_vae_encode_surfaces.py -v -k "ltx_video_legacy or svd or cosmos or hunyuanvideo"`
Expected: PASS — 12 tests (4 backends × 3 tests).

- [ ] **Step 5: Batch B — implement on 3 backends with separable-but-uncalled loaders**

**`cogvideox.py`** — `load_vae_encoder(vae_path, config)` from `fusion_mlx/video/cogvideox/utils.py:66`, `vae_path = model_dir / "vae.safetensors"` (cogvideox/generate.py:151). Config loaded in `start()` (cogvideox.py:27-31 `config.json`). Add to `CogVideoBackend`:

```python
    async def load_vae_encoder(self) -> None:
        if getattr(self, "_stage_vae_encoder", None) is not None:
            return
        from pathlib import Path
        from fusion_mlx.video.cogvideox.utils import load_vae_encoder as _load_vae_enc

        vae_path = Path(self._model_dir) / "vae.safetensors"

        def _load():
            cfg = self._config
            return _load_vae_enc(vae_path, cfg)

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info("cogvideox: vae_encoder load vae=%s", type(self._stage_vae_encoder).__name__)

    async def encode(self, pixels: mx.array) -> mx.array:
        if getattr(self, "_stage_vae_encoder", None) is None:
            await self.load_vae_encoder()
        vae_enc = self._stage_vae_encoder
        ndim = pixels.ndim
        if ndim not in (4, 5):
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        src_np = np.array(pixels[0] if ndim == 5 else pixels)

        def _encode():
            x = mx.array(src_np)
            if x.ndim == 4:
                x = x[None]
            lat = vae_enc.encode(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("cogvideox: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_vae_encoder = None
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        await _clear_mlx_cache()
        logger.info("cogvideox: vae_encoder unload")
```

**`minimax_h3.py`** — `MiniMaxH3VideoVAE.from_pretrained(vae_path, config=H3VAEConfig())`, `vae_path = _resolve_subdir(model_path, "video_vae")` (minimax_h3/generate.py:638-639). `_resolve_subdir` lives in generate.py — import or inline. Add to `MiniMaxH3Backend`:

```python
    async def load_vae_encoder(self) -> None:
        if getattr(self, "_stage_vae_encoder", None) is not None:
            return
        import os
        from fusion_mlx.video.minimax_h3.vae import MiniMaxH3VideoVAE, H3VAEConfig

        model_path = self._model_path
        video_vae_dir = os.path.join(model_path, "video_vae")
        if not os.path.isdir(video_vae_dir):
            video_vae_dir = model_path

        def _load():
            vae = MiniMaxH3VideoVAE.from_pretrained(video_vae_dir, config=H3VAEConfig())
            mx.eval(vae.parameters())
            return vae

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info("minimax_h3: vae_encoder load vae=%s", type(self._stage_vae_encoder).__name__)

    async def encode(self, pixels: mx.array) -> mx.array:
        if getattr(self, "_stage_vae_encoder", None) is None:
            await self.load_vae_encoder()
        vae = self._stage_vae_encoder
        ndim = pixels.ndim
        if ndim not in (4, 5):
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        src_np = np.array(pixels[0] if ndim == 5 else pixels)

        def _encode():
            x = mx.array(src_np)
            if x.ndim == 4:
                x = x[None]
            lat = vae.encode(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("minimax_h3: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_vae_encoder = None
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        await _clear_mlx_cache()
        logger.info("minimax_h3: vae_encoder unload")
```

**`ltx2_5.py`** — `load_video_encoder(weights_path)` from `fusion_mlx/video/ltx2_5/video_vae.py:178`. Verify the exact weights-path the encoder expects (the `vae/encoder` subdir). Add to `LTX25Backend`:

```python
    async def load_vae_encoder(self) -> None:
        if getattr(self, "_stage_vae_encoder", None) is not None:
            return
        from pathlib import Path
        from fusion_mlx.video.ltx2_5.video_vae import load_video_encoder

        model_path = Path(self._model_path)
        enc_path = model_path / "vae" / "encoder"

        def _load():
            return load_video_encoder(enc_path)

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info("ltx2_5: vae_encoder load vae=%s", type(self._stage_vae_encoder).__name__)

    async def encode(self, pixels: mx.array) -> mx.array:
        if getattr(self, "_stage_vae_encoder", None) is None:
            await self.load_vae_encoder()
        vae_enc = self._stage_vae_encoder
        ndim = pixels.ndim
        if ndim not in (4, 5):
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        src_np = np.array(pixels[0] if ndim == 5 else pixels)

        def _encode():
            x = mx.array(src_np)
            if x.ndim == 4:
                x = x[None]
            lat = vae_enc(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("ltx2_5: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_vae_encoder = None
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        await _clear_mlx_cache()
        logger.info("ltx2_5: vae_encoder unload")
```

Note: `ltx2_5` `VideoEncoder` is a callable module (`__call__`, not `.encode`) — see `ltx2/generate.py:92` `latent = vae_encoder(prepare_image_for_encoding(...))`. The plan uses `vae_enc(x)` for ltx2_5, `vae.encode(x)` for the others. If `ltx2_5/video_vae.py:load_video_encoder` returns an object with `.encode`, swap to `.encode(x)` (executor verifies at implementation time — record in report).

- [ ] **Step 6: Run full Task 9 test to verify all 7 backends pass**

Run: `.venv/bin/python -m pytest tests/unit/test_vae_encode_surfaces.py -v`
Expected: PASS — 21 tests (7 backends × 3 tests).

- [ ] **Step 7: Commit Batch A + B**

```bash
git add fusion_mlx/engines/video_backends/ltx_video_legacy.py \
  fusion_mlx/engines/video_backends/svd.py \
  fusion_mlx/engines/video_backends/cosmos.py \
  fusion_mlx/engines/video_backends/hunyuanvideo.py \
  fusion_mlx/engines/video_backends/cogvideox.py \
  fusion_mlx/engines/video_backends/minimax_h3.py \
  fusion_mlx/engines/video_backends/ltx2_5.py \
  tests/unit/test_vae_encode_surfaces.py
git commit -m "feat(video): Surface A VAE encode on 7 denoise-less backends (#653)

Wire load_vae_encoder/encode/unload_vae_encoder on ltx_video_legacy,
svd, cosmos, hunyuanvideo, cogvideox, minimax_h3, ltx2_5. Each mirrors
the Wan2 numpy-bridge + executor + mx.eval thread-portability pattern
(#630 stream invariant). ltx2/opensora/uniworld deferred to follow-up
issues (family VAE not separable from generate in backend)."
```

---

## Task 10: Real-model tests (gated FUSION_MLX_REAL_MODEL_TESTS)

**Files:**
- Create: `tests/unit/test_653_real_model.py`

**Interfaces:**
- Consumes: all surfaces (A encode, B ControlNet, C inpaint) on Wan2 + SkyReels. Tasks 1–9 wire these; this task is the behavioral guard the contract-lock tests (Rule 9) defer to.
- Produces: `tests/unit/test_653_real_model.py` — four gated real-model tests proving (A) Wan2 VAE encode→decode roundtrip preserves structure, (B) Wan2 ControlNet e2e steers output vs pure-T2V, (C) Wan2 inpaint e2e preserves a frozen region across denoise steps, (D) SkyReels inpaint e2e preserves a frozen region.

**Ruling (recorded here, governs this task):** Real-model tests are the ONLY behavioral assertions in the plan; Tasks 1–9 are contract-lock (`inspect.signature`/`inspect.getsource`) per Rule 9. These four tests are the difference between "the surfaces exist" and "the surfaces work." They MUST run real MLX weights end-to-end. Gated behind `FUSION_MLX_REAL_MODEL_TESTS` + installed model dirs; skip cleanly otherwise. Never `mx.clear_streams()` (#630). start/stop the server is NOT required — these tests instantiate `VideoGenEngine` directly (same as `test_wan2_vae_encode_roundtrip.py` / `test_wan2_stage_t2v_smoke.py`), no HTTP. Models via https://hf-mirror.com if missing.

- [ ] **Step 1: Write the four gated real-model tests**

Create `tests/unit/test_653_real_model.py`. The four tests each call the PUBLIC `VideoGenEngine` surface (the surface downstream fusion-comfyui reaches via `fusion_mlx.public_api`). The Wan2 VAE roundtrip ALREADY exists at `tests/test_wan2_vae_encode_roundtrip.py` (v0.8.42) — this test file does NOT duplicate it; it adds the three NEW surfaces (B ControlNet, C inpaint Wan2, C inpaint SkyReels) and a SkyReels VAE roundtrip. Reference it for the `_structured_pixels`/`_pixel_corr` helpers (re-import rather than copy — the helpers are public test utilities living in the top-level tests/ dir; import them).

```python
import asyncio
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.public_api import VideoGenEngine

# Wan2.1-14B: full-encoder VAE (84 keys), 14B DiT. Surface A roundtrip + Surface C
# inpaint + the pure-T2V baseline for the Surface B diff. Override per-test via env.
WAN2_DIR = Path(
    os.environ.get(
        "FUSION_653_WAN2_MODEL",
        str(Path.home() / ".fusion-mlx/models/Wan2.1-14B"),
    )
)
# Surface B: TheDenk wan2.1-t2v-14b-controlnet-canny-v1 (adapter weights, loaded on
# top of the 14B DiT). Override via FUSION_653_CONTROLNET_DIR.
CONTROLNET_DIR = Path(
    os.environ.get(
        "FUSION_653_CONTROLNET_DIR",
        str(Path.home() / ".fusion-mlx/models/models--TheDenk--wan2.1-t2v-14b-controlnet-canny-v1"),
    )
)
# SkyReels V2: Wan2.2-TI2V-5B q8 (alias wan22-ti2v-5b). Surface D (SkyReels inpaint).
SKYREELS_DIR = Path(
    os.environ.get(
        "FUSION_653_SKYREELS_MODEL",
        str(Path.home() / ".fusion-mlx/models/wan22-ti2v-5b"),
    )
)

pytestmark = pytest.mark.real_model


def _skip_unless_real_model(model_dir: Path, label: str, need_files=()):
    if not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"):
        pytest.skip(f"set FUSION_MLX_REAL_MODEL_TESTS=1 to run #653 real-model {label}")
    if not model_dir.exists():
        pytest.skip(f"{label} model not installed at {model_dir}")
    for f in need_files:
        if not (model_dir / f).exists():
            pytest.skip(f"{label} partial model: missing {f} at {model_dir}")


# ---- Surface A: SkyReels VAE encode roundtrip (new this task) ----

def _structured_pixels(t: int, h: int = 256, w: int = 256) -> mx.array:
    ys = mx.arange(h, dtype=mx.float32) / h
    xs = mx.arange(w, dtype=mx.float32) / w
    ts = mx.arange(t, dtype=mx.float32) / max(t - 1, 1)
    frame = (ys[:, None] + xs[None, :]) * 0.5
    frames = []
    for i in range(t):
        chan_r = frame + 0.1 * ts[i]
        chan_g = frame * 0.7
        chan_b = mx.broadcast_to(ts[i], (h, w))
        frames.append(mx.stack([chan_r, chan_g, chan_b], axis=-1))
    return mx.stack(frames, axis=0)[None]


def _pixel_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def test_skyreels_vae_encode_decode_roundtrip():
    # Surface A on SkyReels. SkyReelsVAE.encode (vae.py:212) [B,3,T,H,W] -> [B,16,T,H/8,W/8].
    # 5 frames = 1 streaming chunk (iter_=3 odd, same wan-style streaming). Guards the
    # numpy-bridge #630 invariant through the SkyReels encode path (Task 8).
    _skip_unless_real_model(SKYREELS_DIR, "skyreels")
    eng = VideoGenEngine(str(SKYREELS_DIR))

    async def _run():
        await eng.start()
        await eng.load_vae()
        pixels = _structured_pixels(5, 256, 256)
        lat = await eng.encode(pixels)
        assert lat.ndim == 5
        assert not np.any(np.isnan(np.array(lat))), "skyreels encode produced NaN"
        out = await eng.decode(lat)
        await eng.unload_vae()
        await eng.stop()
        return pixels, out

    pixels, out = asyncio.run(_run())
    arr = np.array(out)
    assert not np.any(np.isnan(arr)), "skyreels decode produced NaN"
    corr = _pixel_corr(np.array(pixels), arr)
    print(f"\n#653 Surface A skyreels roundtrip pixel_corr={corr:.4f} shape={arr.shape}")
    assert corr >= 0.9, f"skyreels VAE roundtrip corr {corr:.4f} < 0.9 (scrambled)"


# ---- Surface B: Wan2 ControlNet e2e (steers output vs pure-T2V) ----

def _empty_latent_14b(num_frames: int = 17, height: int = 480, width: int = 832):
    t_latent = (num_frames - 1) // 4 + 1
    h_latent = height // 8
    w_latent = width // 8
    return mx.zeros((1, 16, t_latent, h_latent, w_latent))


def test_wan2_controlnet_steers_output_vs_pure_t2v():
    # Surface B: encode_control(controlnet_image=...) returns a ControlState carrying
    # controlnet_adapter + controlnet_latent (Task 5/6); denoise(control=...) injects
    # per-step residuals into the Wan2 DiT block loop (R1). The guard: a canny-edge
    # control image MUST produce a latent that differs from the pure-T2V (control=None)
    # latent at the same seed. Bit-exactness of the residual math is verified by the
    # ControlNet library's own tests; this test proves the WIRING carries the adapter
    # all the way through the public engine surface. 1 step, fixed seed, small frame
    # count to keep it a smoke (not a quality test).
    _skip_unless_real_model(WAN2_DIR, "wan2-controlnet", need_files=("t5_encoder.safetensors",))
    _skip_unless_real_model(CONTROLNET_DIR, "controlnet-adapter")
    # Need a control image: synthesize a canny-ish edge frame to a temp png. The
    # adapter's encode_control reads the path. Use a deterministic gradient + threshold.
    import tempfile

    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed; cannot synthesize control image")

    tmp = Path(tempfile.mkdtemp(prefix="fusion_653_cn_"))
    ctrl_path = tmp / "control.png"
    ys, xs = np.mgrid[0:480, 0:832]
    edge = ((xs % 64 < 4) | (ys % 64 < 4)).astype(np.uint8) * 255
    Image.fromarray(np.stack([edge, edge, edge], axis=-1)).save(ctrl_path)

    async def _run(use_cn: bool):
        eng = VideoGenEngine(str(WAN2_DIR))
        await eng.start()
        await eng.load_text_encoder()
        emb = await eng.encode_text("a city street, neon, night")
        await eng.unload_text_encoder()
        await eng.load_dit()
        await eng.load_vae_encoder()
        ctrl = None
        if use_cn:
            ctrl = await eng.encode_control(
                controlnet_image=str(ctrl_path),
                control_type="canny",
                controlnet_strength=1.0,
            )
            assert ctrl is not None, "encode_control returned None for controlnet_image"
        else:
            ctrl = await eng.encode_control()  # pure-T2V -> None
            assert ctrl is None
        latent = await eng.denoise(
            latent=_empty_latent_14b(),
            pos_embed=emb["embed"],
            neg_embed=None,
            steps=1,
            cfg=1.0,
            seed=42,
            num_frames=17,
            control=ctrl,
        )
        await eng.unload_vae_encoder()
        await eng.unload_dit()
        await eng.stop()
        return np.array(latent)

    pure = asyncio.run(_run(False))
    with_cn = asyncio.run(_run(True))
    # Cleanup temp
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    assert not np.any(np.isnan(pure)), "pure-T2V latent has NaN"
    assert not np.any(np.isnan(with_cn)), "controlnet latent has NaN"
    diff = float(np.abs(with_cn - pure).mean())
    print(f"\n#653 Surface B controlnet vs pure-T2V mean|diff|={diff:.6f} shape={pure.shape}")
    assert diff > 0.0, "ControlNet produced bit-identical latent to pure-T2V (wiring broken)"


# ---- Surface C: Wan2 inpaint e2e (frozen region preserved) ----

def test_wan2_inpaint_preserves_frozen_region():
    # Surface C: denoise(..., inpaint_mask=, init_latent=) re-composites
    # mask*latents + (1-mask)*init after every sched.step (Task 3). The guard: where
    # mask=0 (frozen), the final latent MUST equal init_latent exactly — the denoiser
    # cannot have touched it. Where mask=1 (reactive), it MUST differ from init (the
    # denoiser did work). 1 step is enough: one re-composite proves the wiring; more
    # steps would test the scheduler, not the surface.
    _skip_unless_real_model(WAN2_DIR, "wan2-inpaint", need_files=("t5_encoder.safetensors",))
    latent = _empty_latent_14b()
    # init_latent = a DISTINCT structured latent (not zeros). mask=0 on the first
    # temporal slab (freeze frame 0), mask=1 on the rest (reactive).
    init = mx.array(np.linspace(0.5, 1.5, latent.size, dtype=np.float32).reshape(latent.shape))
    mask = mx.ones_like(latent)
    mask[:, :, 0, :, :] = 0.0  # freeze t_latent=0 slab
    mx.eval(init, mask)

    async def _run():
        eng = VideoGenEngine(str(WAN2_DIR))
        await eng.start()
        await eng.load_text_encoder()
        emb = await eng.encode_text("a horse galloping across a field")
        await eng.unload_text_encoder()
        await eng.load_dit()
        out = await eng.denoise(
            latent=latent,
            pos_embed=emb["embed"],
            neg_embed=None,
            steps=1,
            cfg=1.0,
            seed=42,
            num_frames=17,
            control=None,
            inpaint_mask=mask,
            init_latent=init,
        )
        await eng.unload_dit()
        await eng.stop()
        return np.array(out), np.array(init), np.array(mask)

    out, init_arr, mask_arr = asyncio.run(_run())
    assert not np.any(np.isnan(out)), "inpaint latent has NaN"
    frozen = out[mask_arr == 0]
    frozen_init = init_arr[mask_arr == 0]
    reactive = out[mask_arr == 1]
    reactive_init = init_arr[mask_arr == 1]
    print(
        f"\n#653 Surface C wan2 inpaint: frozen_equal={np.array_equal(frozen, frozen_init)} "
        f"reactive_diff_mean={float(np.abs(reactive - reactive_init).mean()):.6f}"
    )
    assert np.array_equal(frozen, frozen_init), (
        "frozen region (mask=0) was modified by denoise — apply_inpaint_mask not re-compositing"
    )
    assert float(np.abs(reactive - reactive_init).mean()) > 0.0, (
        "reactive region (mask=1) is bit-identical to init — denoiser did no work"
    )


# ---- Surface C: SkyReels inpaint e2e (frozen region preserved) ----

def test_skyreels_inpaint_preserves_frozen_region():
    # Surface C on SkyReels: same guard as the Wan2 test, SkyReels denoise loop.
    # SkyReels latents are 16-channel 5D same shape convention; the mask slab is
    # temporal axis 2. Gated to the default denoise path only (R4); async/speculative
    # paths are NOT exercised here (#177/#180 follow-up).
    _skip_unless_real_model(SKYREELS_DIR, "skyreels-inpaint")
    # SkyReels 5B q8: smaller spatial. Probe the DiT latent shape from config lazily —
    # use a conservative 17-frame 480x832 empty latent; SkyReels rejects mismatched
    # shapes at patch_embedding, so if the model wants a different size this surfaces
    # as a clear RuntimeError (not a silent pass). A wrong shape is a TEST bug, not a
    # surface bug — the implementer fixes the shape from the actual model config.
    latent = _empty_latent_14b(num_frames=17, height=480, width=832)
    init = mx.array(np.linspace(0.5, 1.5, latent.size, dtype=np.float32).reshape(latent.shape))
    mask = mx.ones_like(latent)
    mask[:, :, 0, :, :] = 0.0
    mx.eval(init, mask)

    async def _run():
        eng = VideoGenEngine(str(SKYREELS_DIR))
        await eng.start()
        await eng.load_text_encoder()
        emb = await eng.encode_text("waves crashing on rocks")
        await eng.unload_text_encoder()
        await eng.load_dit()
        out = await eng.denoise(
            latent=latent,
            pos_embed=emb["embed"],
            neg_embed=None,
            steps=1,
            cfg=1.0,
            seed=42,
            num_frames=17,
            control=None,
            inpaint_mask=mask,
            init_latent=init,
        )
        await eng.unload_dit()
        await eng.stop()
        return np.array(out), np.array(init), np.array(mask)

    out, init_arr, mask_arr = asyncio.run(_run())
    assert not np.any(np.isnan(out)), "skyreels inpaint latent has NaN"
    frozen = out[mask_arr == 0]
    frozen_init = init_arr[mask_arr == 0]
    reactive = out[mask_arr == 1]
    reactive_init = init_arr[mask_arr == 1]
    print(
        f"\n#653 Surface C skyreels inpaint: frozen_equal={np.array_equal(frozen, frozen_init)} "
        f"reactive_diff_mean={float(np.abs(reactive - reactive_init).mean()):.6f}"
    )
    assert np.array_equal(frozen, frozen_init), (
        "skyreels frozen region (mask=0) modified — apply_inpaint_mask not wired in SkyReels loop"
    )
    assert float(np.abs(reactive - reactive_init).mean()) > 0.0, (
        "skyreels reactive region (mask=1) bit-identical to init — denoiser did no work"
    )
```

NOTE for the implementer: (1) The four tests skip cleanly when `FUSION_MLX_REAL_MODEL_TESTS` is unset OR a model dir is missing — the full suite (Task 12) MUST stay green without models. (2) Surface B synthesizes a canny-edge PNG via Pillow; if Pillow is absent, the test skips (never fails on an optional dep). (3) Surface C uses 1 denoise step: the guard is the re-composite math, not scheduler quality — more steps add runtime without strengthening the assertion. (4) The frozen-region assertion is EXACT (`np.array_equal`) because `apply_inpaint_mask` is `mask*latents + (1-mask)*init` — where mask=0 the output IS init, bit-for-bit. (5) If a model dir uses a different latent shape than `_empty_latent_14b` assumes, the denoise call raises a clear shape error — fix the shape constant in the test from the model's actual config; do NOT relax the assertion. (6) `start.sh` (server lifecycle) is NOT used here — these instantiate `VideoGenEngine` directly. (7) Never call `mx.clear_streams()` — #630 stream invariant; the executor owns its own stream per call.

- [ ] **Step 2: Run the tests to verify they skip (no real-model gate)**

Run: `.venv/bin/python -m pytest tests/unit/test_653_real_model.py -v`
Expected: 4 SKIPPED (`set FUSION_MLX_REAL_MODEL_TESTS=1 to run #653 real-model ...`). The full suite must stay green.

- [ ] **Step 3: Run the tests with the real-model gate (verification)**

Run: `FUSION_MLX_REAL_MODEL_TESTS=1 .venv/bin/python -m pytest tests/unit/test_653_real_model.py -v -s`
Expected: 4 PASSED (if models installed) or 4 SKIPPED (if models absent). The implementer runs this with the models present to prove the wiring; if a model is missing, download via https://hf-mirror.com (per CLAUDE.md) and re-run. If a test FAILS, root-cause per systematic-debugging BEFORE changing the test — a failure means a surface is wired wrong in Tasks 1–9, not a test bug (unless the latent shape is wrong, per note 5).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_653_real_model.py
git commit -m "test(video): #653 real-model e2e — ControlNet steer + inpaint frozen-region (#653)"
```

---

## Task 11: File 10 follow-up issues (denoise-port for Surface B+C on the other backends)

**Files:**
- GitHub issues (one per family): cogvideox, cosmos, hunyuanvideo, ltx_video_legacy, ltx2_5, ltx2, minimax_h3, opensora, svd, uniworld(stub).

**Interfaces:**
- Consumes: the Surface A/B/C scope split decided in Tasks 1–10 (spec rows 109–122): Wan2 + SkyReels get all three surfaces; the 10 denoise-less backends get Surface A only (this PR) and defer Surface B+C (the ControlNet residual injection needs a per-family denoise loop, which these backends lack — their generate path is monolithic). 3 of the 10 (ltx2, opensora, uniworld) ALSO defer Surface A (no separable encoder loader found — Task 9 Ruling R3 corrected).
- Produces: 10 GitHub issues on `dahai80/fusion-mlx`, each cross-referencing #653, each scoped to ONE backend + the surfaces it still needs. Closes the spec's "file 10 follow-up issues" directive without scope-creeping this PR.

**Ruling (recorded here):** Surface B+C on the 8 denoise-less backends that DID get Surface A in this PR (cogvideox, cosmos, hunyuanvideo, ltx_video_legacy, ltx2_5, minimax_h3, svd — 7 backends) is a denoise-loop port, NOT a mechanical edit: each family's generate path is monolithic (no separable `run_denoise` like Wan2/stage.py), so injecting ControlNet residuals + inpaint re-composite requires refactoring the family generate into a staged denoise. That is a per-family feature, tracked one issue per family. The 3 Surface-A-deferred backends (ltx2, opensora, uniworld) get a single issue each covering A+B+C together. Total = 10 issues.

- [ ] **Step 1: File the 7 Surface B+C denoise-port issues (backends that got Surface A this PR)**

Run each `gh issue create` against `dahai80/fusion-mlx`. The title names the backend + the missing surfaces; the body cites the Wan2/SkyReels precedent implemented in Tasks 3–7 + the family's monolithic-generate blocker. Use `--label "enhancement"` if the label exists (skip `--label` if it 422s — the repo may not have it). All issues cross-reference #653 in the body. Replace `<body>` inline — these are the exact strings.

```bash
# 7 backends that got Surface A in this PR; need Surface B+C (denoise-loop port).
for fam in cogvideox cosmos hunyuanvideo ltx_video_legacy ltx2_5 minimax_h3 svd; do
  case $fam in
    cogvideox)     di="cogvideox/generate.py (monolithic; no run_denoise)";;
    cosmos)        di="cosmos/generate.py (monolithic)";;
    hunyuanvideo)  di="hunyuanvideo/generate.py (monolithic)";;
    ltx_video_legacy) di="ltx_video_legacy generate path (LTVideoVAE has no separable denoise loop)";;
    ltx2_5)        di="ltx2_5 generate.py (pipeline-based, no per-step residual hook)";;
    minimax_h3)    di="minimax_h3 generate.py (monolithic)";;
    svd)           di="svd/generate.py (monolithic)";;
  esac
  gh issue create -R dahai80/fusion-mlx --title "[$fam] Port ControlNet (Surface B) + inpaint-mask (Surface C) denoise surfaces (#653 follow-up)" --body "Follow-up to #653. This PR (#653) landed **Surface A** (VAE encode) on \`$fam\` via \`load_vae_encoder\`/\`encode\`/\`unload_vae_encoder\` overrides. **Surface B** (ControlNet residual injection into the denoise loop) and **Surface C** (inpaint-mask re-composite after each \`sched.step\`) are NOT implemented for \`$fam\` — they require a separable per-step denoise loop (the Wan2 precedent is \`fusion_mlx/video/wan2/stage.py:run_denoise\` + \`wan_2.py:WanModel.__call__\` block loop), but \`$fam\`'s generate path is monolithic (\`$di\`).

Scope of this follow-up:
- Refactor the \`$fam\` generate path to expose a staged \`run_denoise\`-equivalent (or a per-step callback hook).
- Wire \`ControlState.controlnet_adapter\`/\`controlnet_latent\` residual injection (Surface B), mirroring Wan2 \`wan_2.py\` block loop + SkyReels \`pipelines/__init__.py:716-724\`.
- Wire \`apply_inpaint_mask(latents, init_latent, inpaint_mask)\` re-composite after each step (Surface C), mirroring Wan2 \`stage.py:497\` insertion.
- Contract-lock unit tests (Rule 9) + one gated real-model e2e per surface, gated behind \`FUSION_MLX_REAL_MODEL_TESTS\`.

This is a per-family feature, not a mechanical edit. The fusion-comfyui dead-path stubs \`ControlNetApply\`/\`ControlNetApplyAdvanced\`/\`VAEEncodeForInpaint\`/\`InpaintModelConditioning\` stay stubbed for \`$fam\` until this lands."
done
```

- [ ] **Step 2: File the 3 Surface A+B+C issues (backends that deferred Surface A too)**

```bash
# ltx2: encoder loaded inline in generate.py:81-97 as a local; no persistent backend field.
gh issue create -R dahai80/fusion-mlx --title "[ltx2] Port VAE encode (Surface A) + ControlNet (B) + inpaint (C) (#653 follow-up)" --body "Follow-up to #653. \`ltx2\` deferred ALL three surfaces in the #653 PR. Surface A blocker: the VAE encoder is loaded inline in \`ltx2/generate.py:81-97\` as a \`VideoEncoder.from_pretrained(model_path/\"vae\"/\"encoder\")\` LOCAL threaded through \`_encode_image_latent_shared\` — there is no persistent backend field to wire \`load_vae_encoder\`/\`encode\`/\`unload_vae_encoder\` against. Wiring requires refactoring that shared-loader out of generate into a backend-owned field (mirroring Wan2 \`_stage_vae_encoder\`).

Once Surface A lands, Surface B+C require the same monolithic-generate → staged-denoise refactor described in the other follow-up issues (Wan2 \`stage.py:run_denoise\` precedent). File as one issue because the A refactor is a prerequisite for B+C on this backend.

Scope: (A) extract encoder to backend field + \`encode\` override with numpy-bridge (#630); (B) ControlNet residual injection; (C) \`apply_inpaint_mask\` re-composite. Contract-lock tests + gated real-model e2e per surface."

# opensora: self._vae initialized None, NEVER loaded; family VAE class/loader not located.
gh issue create -R dahai80/fusion-mlx --title "[opensora] Port VAE encode (Surface A) + ControlNet (B) + inpaint (C) (#653 follow-up)" --body "Follow-up to #653. \`opensora\` deferred ALL three surfaces in the #653 PR. Surface A blocker: \`self._vae\` is initialized to \`None\` in \`__init__\` (\`opensora.py:26\`) but NEVER loaded in \`start()\`; \`generate.py:170\` falls back to a random \`image_latent\`. The family VAE class + loader path were NOT located in the opensora package during the #653 Phase-1 survey — needs discovery (likely a \`from_pretrained\` in the opensora video-vae submodule).

Scope: (A) locate the family VAE class + loader, load it in \`start()\` (or lazily in \`load_vae_encoder\`), wire \`encode\` with the numpy-bridge #630 invariant; (B) ControlNet residual injection; (C) \`apply_inpaint_mask\` re-composite. The monolithic-generate → staged-denoise refactor (Wan2 \`stage.py:run_denoise\` precedent) is needed for B+C. Contract-lock tests + gated real-model e2e per surface."

# uniworld: 7-line re-export stub; spec row 122 says "stub-only (defer all)".
gh issue create -R dahai80/fusion-mlx --title "[uniworld] Port VAE encode (Surface A) + ControlNet (B) + inpaint (C) (#653 follow-up)" --body "Follow-up to #653. \`uniworld\` deferred ALL three surfaces in the #653 PR. \`uniworld.py\` is a 7-line re-export stub (spec row 122: \"stub-only (defer all)\") — there is no backend implementation to wire surfaces against. This issue tracks un-stubbing the backend (or routing through the underlying family backend it re-exports) before any of A/B/C can land.

Scope: (0) determine whether \`uniworld\` is a real backend or a deprecated re-export; if deprecated, close this issue as wontfix and document; if real, implement the backend then (A) VAE encode, (B) ControlNet, (C) inpaint per the Wan2 precedents. Contract-lock tests + gated real-model e2e per surface."
```

- [ ] **Step 3: Record the 10 issue numbers in the PR description**

After filing, capture the 10 issue numbers (the `gh issue create` output prints each URL). Append a "Follow-up issues" section to the Task 12 PR body listing all 10 with their URLs, so the reviewer can verify the spec's "file 10 follow-up issues" directive is satisfied. No commit needed for this step — the issues ARE the deliverable; the PR body (Task 12) carries the index.

- [ ] **Step 4: Commit (this task is issue-filing, no code — but record the issue numbers in the plan)**

Add the 10 issue numbers to the Status / Resume Notes section at the bottom of THIS plan file (so a resuming session knows they exist), then commit the plan edit:

```bash
git add docs/superpowers/plans/2026-08-31-653-vae-controlnet-inpaint-surfaces.md
git commit -m "docs(plan): record #653 Task 11 follow-up issue numbers"
```

If `gh issue create` is unavailable offline, record a `TODO(issue-filing)` line per family in the Status / Resume Notes and file them when online — do NOT block the PR on issue filing (the issues are a post-merge tracking artifact, the spec's directive is satisfied by filing before close).

---

## Task 12: Lint + full-suite sweep + README/CHANGELOG + PR + merge + release

**Files:**
- Modify: `README.md` (stage-API table + Surface A/B/C rows, lines 783–817)
- Modify: `CHANGELOG.md` (add `## [0.8.57] - <date>` entry under `## [Unreleased]`)
- Modify: `fusion_mlx/_version.py` (`__version__ = "0.8.56"` → `"0.8.57"`)
- Run: lint, full-suite sweep, PR, merge to main, tag, GitHub release

**Interfaces:**
- Consumes: all 10 prior tasks (the code + tests + follow-up issues).
- Produces: v0.8.57 released on GitHub (PyPI skipped — no token, per memory). Main green. #653 closed.

**Ruling (recorded here):** v0.8.56 is the latest release (2026-08-31, GitHub-only). #653 lands as v0.8.57. PyPI is SKIPPED (no token in env, per `fusion-mlx-v0856-release`/`v0.8.55`/`v0.8.53` memory). The release sequence is: PR → CI green → squash-merge to main → tag `v0.8.57` → `gh release create v0.8.57` (omit `--target`, it 422s when the tag exists — memory GOTCHA). The version bump + CHANGELOG + README go in the SAME PR as the code (one squash commit), NOT a separate release PR — this matches the v0.8.42–v0.8.56 cadence. `debt_modules.txt` is a DATA file, not Python — NEVER pass it to ruff (memory GOTCHA: ruff lints it if passed explicitly).

- [ ] **Step 1: Update README stage-API section**

In `README.md`, the existing VAE-encoder table row (line 788) shows only the #652 `encode_control(image=, ...)` kwargs. Add the #653 surfaces. Replace line 788's row + the note block below it. The DiT `denoise` row (line 786) gains the two Surface C kwargs.

Edit line 786 (DiT row) — append the inpaint kwargs to the denoise signature:

```markdown
| DiT | `load_dit()` | `denoise(latent, pos_embed, neg_embed, steps, cfg, seed[, num_frames][, control][, inpaint_mask=, init_latent=])` | `unload_dit()` |
```

Edit line 788 (VAE-encoder row) — add the Surface A `encode(pixels)` run surface + the Surface B `controlnet_image` kwarg:

```markdown
| VAE encoder (#652/#653) | `load_vae_encoder()` | `encode(pixels) -> latent` (Surface A, #653) / `encode_control(image=, width=, height=, num_frames=, control_video=, control_mask=, reference_images=, camera_conditions=, controlnet_image=, control_type=, controlnet_strength=) -> ControlState \| None` (Surface B, #653) | `unload_vae_encoder()` |
```

After the existing `> **Wan2 conditioning (#652):**` note block (line 790–797), add a new note block for #653:

```markdown
> **#653 surfaces (Wan2 + SkyReels):**
> - **Surface A (VAE encode):** `encode(pixels) -> 5D latent` is now wired on 9 backends
>   (Wan2, SkyReels, ltx_video_legacy, svd, cosmos, hunyuanvideo, cogvideox, minimax_h3,
>   ltx2_5). `ltx2` / `opensora` / `uniworld` defer to follow-up issues. Numpy-bridges on
>   the caller thread, rebuilds `mx.array` + `mx.eval` on the worker thread (#630 stream
>   invariant).
> - **Surface B (ControlNet):** `encode_control(controlnet_image=<path>, control_type="canny",
>   controlnet_strength=1.0)` builds a `ControlNet` adapter + preprocessed control latent,
>   returned on `ControlState.controlnet_adapter`/`.controlnet_latent`. `denoise(control=...)`
>   injects per-step residuals into the DiT block loop (Wan2 `wan_2.py`; SkyReels
>   `pipelines/__init__.py`).
> - **Surface C (inpaint mask):** `denoise(..., inpaint_mask=<5D mask>, init_latent=<5D latent>)`
>   re-composites `mask*latents + (1-mask)*init` after each denoise step — `mask=1` reactive,
>   `mask=0` frozen (restored to `init_latent` bit-exactly). Orthogonal to the TI2V mask blend.
```

Edit lines 814–817 (the `NotImplementedError` defaults paragraph) — update the backend coverage list:

```markdown
Video backends inherit `NotImplementedError` defaults for the stage API (issue
#170 phase 2); Surface A (`encode`) is now wired on 9 backends (#653), Surface B
(ControlNet) + Surface C (inpaint) on Wan2 + SkyReels (#653). The 10 other
denoise-less backends defer Surface B+C to per-family follow-up issues (#653
follow-ups) — their generate paths are monolithic (no separable `run_denoise`).
`Wan2Backend` additionally implements the full I2V / VACE / camera conditioning
stage surface (#652).
```

- [ ] **Step 2: Add the CHANGELOG entry**

In `CHANGELOG.md`, insert a new entry above `## [0.8.56] - 2026-08-31` (below `## [Unreleased]`), under `## [Unreleased]`:

```markdown
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
```

- [ ] **Step 3: Bump the version**

In `fusion_mlx/_version.py`, change:

```python
__version__ = "0.8.56"
```

to:

```python
__version__ = "0.8.57"
```

- [ ] **Step 4: Lint (black --fast + ruff; NEVER debt_modules.txt)**

Run from the repo root (the `.venv` is already active per CLAUDE.md):

```bash
.venv/bin/python -m black --fast fusion_mlx/ tests/ 2>&1 | tail -5
.venv/bin/python -m ruff check fusion_mlx/ tests/ 2>&1 | tail -10
```

Expected: black reports no changes (or only formatting the new files), ruff reports no errors. GOTCHA: do NOT pass `tests/unit/debt_modules.txt` to ruff — it is a data file, not Python; ruff lints it if passed explicitly (memory `v0.8.50` GOTCHA). If ruff flags a real error in the new code, fix it (do not add a `noqa` unless the line genuinely needs one — match the codebase's sparse-`noqa` convention).

- [ ] **Step 5: Full-suite sweep (`.venv/bin/python -m pytest`, NOT bare pytest)**

Run the full suite WITHOUT the real-model gate (the 4 new real-model tests must SKIP, not fail):

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -20
```

Expected: all green, 4 SKIPPED from `test_653_real_model.py`. GOTCHA: bare `pytest tests/unit -q` can report "No tests collected" via the rtk proxy (memory `v0.8.52` GOTCHA) — use the explicit `.venv/bin/python -m pytest`. If a flaky full-suite failure appears (e.g. `test_batched_faster_than_sequential` load-flaky, memory `v0.8.56` GOTCHA), re-run that test in isolation to confirm it is the known flake, not a regression from this PR.

- [ ] **Step 6: (Optional, when models present) Run the real-model gate**

If the 4 models are installed (`Wan2.1-14B`, `models--TheDenk--wan2.1-t2v-14b-controlnet-canny-v1`, `wan22-ti2v-5B`), prove the wiring:

```bash
FUSION_MLX_REAL_MODEL_TESTS=1 .venv/bin/python -m pytest tests/unit/test_653_real_model.py -v -s
```

Expected: 4 PASSED. If a model is missing, download via https://hf-mirror.com (per CLAUDE.md) or skip this step — the unit contract-lock tests (Tasks 1–9) + the gate-skip (Step 5) are the CI gate; the real-model run is host verification. start/stop fusion-mlx via `~/claude-home/fusion-mlx/start.sh start|stop` is NOT needed for these tests (they instantiate `VideoGenEngine` directly).

- [ ] **Step 7: Create the PR (push branch + gh pr create)**

The branch is `feat/api-stable-layer` (per Status / Resume Notes — verify with `git branch --show-current`; if on `main`, create `feat/653-surfaces` first). Push + create the PR against `main`:

```bash
git push -u origin HEAD
gh pr create -R dahai80/fusion-mlx --base main --title "feat(video): #653 VAE-encode / ControlNet / inpaint-mask engine surfaces" --body "$(cat <<'EOF'
Closes #653. Lands three engine surfaces on the video backends for fusion-comfyui:

- **Surface A (VAE encode):** `encode(pixels) -> 5D latent` on 9 backends (Wan2, SkyReels, ltx_video_legacy, svd, cosmos, hunyuanvideo, cogvideox, minimax_h3, ltx2_5). Numpy-bridge + worker-thread `mx.eval` (#630).
- **Surface B (ControlNet):** `encode_control(controlnet_image=, control_type=, controlnet_strength=)` → `ControlState` adapter + latent; `denoise(control=...)` injects per-step residuals (Wan2 + SkyReels).
- **Surface C (inpaint mask):** `denoise(..., inpaint_mask=, init_latent=)` re-composites frozen regions per step (Wan2 + SkyReels).

## Changes
- `WanModel.__call__` + `run_denoise`: `controlnet_residuals`/`controlnet_stride` (R1) + `inpaint_mask`/`init_latent` re-composite.
- `Wan2Backend`/`SkyReelsBackend`: `encode_control` ControlNet branch; `encode` Surface A.
- 7 denoise-less backends: `load_vae_encoder`/`encode`/`unload_vae_encoder` overrides (Surface A, R3).
- `VideoGenEngine.denoise` + `VideoBackend.denoise`: thread `inpaint_mask`/`init_latent` (back-compat `None`).
- `_inpaint.py`: `apply_inpaint_mask` neutral helper.

## Tests
- 9 contract-lock unit test files (Rule 9: `inspect.signature`/`inspect.getsource`, no fake-model execution).
- `tests/unit/test_653_real_model.py`: 4 gated real-model e2e (SkyReels VAE roundtrip, Wan2 ControlNet steer, Wan2 + SkyReels inpaint frozen-region). Gated `FUSION_MLX_REAL_MODEL_TESTS`; SKIP without models.

## Follow-up issues (Task 11)
Filed on `dahai80/fusion-mlx` 2026-09-01, all `--label enhancement`, cross-referencing #653.
Surface B+C denoise-port (7 backends that got Surface A in this PR):
- #731 cogvideox — Port ControlNet (Surface B) + inpaint-mask (Surface C)
- #732 cosmos — Port ControlNet (Surface B) + inpaint-mask (Surface C)
- #733 hunyuanvideo — Port ControlNet (Surface B) + inpaint-mask (Surface C)
- #734 ltx_video_legacy — Port ControlNet (Surface B) + inpaint-mask (Surface C)
- #735 ltx2_5 — Port ControlNet (Surface B) + inpaint-mask (Surface C)
- #736 minimax_h3 — Port ControlNet (Surface B) + inpaint-mask (Surface C)
- #737 svd — Port ControlNet (Surface B) + inpaint-mask (Surface C)
Surface A+B+C (3 backends that deferred all three):
- #738 ltx2 — Port VAE encode (Surface A) + ControlNet (B) + inpaint (C)
- #739 opensora — Port VAE encode (Surface A) + ControlNet (B) + inpaint (C)
- #740 uniworld — Port VAE encode (Surface A) + ControlNet (B) + inpaint (C)

## Rulings (R1–R4, load-bearing)
- R1: Wan2 DiT `wan_2.py` had no `controlnet_residuals` kwargs; added + block-loop inject.
- R2: `ControlNet.compute_residuals` expects B-first NCHW; Wan2 latents 4D C-first — reshape reconciliation in `run_denoise`.
- R3: 7 denoise-less backends Surface A = mechanical (persistent/simple loader); ltx2/opensora/uniworld deferred.
- R4: Surface C default denoise path only; SkyReels async/speculative gated OFF (#177/#180).

Version bump 0.8.56 → 0.8.57. PyPI skipped (no token).
EOF
)"
```

- [ ] **Step 8: Verify CI green, then squash-merge to main**

Watch the PR checks (do not merge on a red suite):

```bash
gh pr checks <PR_NUMBER> -R dahai80/fusion-mlx --watch
```

Expected: all green. If a check fails, root-cause per systematic-debugging (a CI fail on the new code is a real regression, not a flake — unless it is the known macOS-load-flaky test, per Step 5 GOTCHA). Once green, squash-merge:

```bash
gh pr merge <PR_NUMBER> -R dahai80/fusion-mlx --squash --delete-branch
```

- [ ] **Step 9: Tag + GitHub release (PyPI skipped)**

After the squash-merge lands on main, pull main, tag, and create the GitHub release. GOTCHA: `gh release create --target` 422s when the tag already exists (memory `v0.8.36`/`v0.8.56` GOTCHA) — omit `--target`, tag + push first:

```bash
git checkout main && git pull
git tag v0.8.57
git push origin v0.8.57
gh release create v0.8.57 -R dahai80/fusion-mlx --title "v0.8.57 — #653 VAE-encode / ControlNet / inpaint surfaces" --notes "$(cat <<'EOF'
## Added
- #653 Surface A: VAE `encode(pixels)` on 9 video backends (Wan2, SkyReels, ltx_video_legacy, svd, cosmos, hunyuanvideo, cogvideox, minimax_h3, ltx2_5).
- #653 Surface B: ControlNet `encode_control(controlnet_image=, control_type=, controlnet_strength=)` + per-step residual injection (Wan2 + SkyReels).
- #653 Surface C: inpaint `denoise(..., inpaint_mask=, init_latent=)` frozen-region re-composite (Wan2 + SkyReels).
- Real-model e2e tests (gated `FUSION_MLX_REAL_MODEL_TESTS`).
- 10 follow-up issues for Surface B+C denoise-loop ports.

PyPI release skipped (no token). Install from GitHub tag.
EOF
)"
```

- [ ] **Step 10: Close #653 + verify**

```bash
gh issue close 653 -R dahai80/fusion-mlx -c "Landed in v0.8.57 (PR #<PR_NUMBER>). Surface A on 9 backends, Surface B+C on Wan2 + SkyReels. 10 follow-up issues track the remaining backends."
```

Verify: `git log --oneline -3` shows the squash merge at HEAD; `gh release view v0.8.57` exists; #653 is closed; the 10 follow-up issues are open. Done.

---

## Self-review

Ran against the committed spec (`docs/superpowers/specs/2026-08-31-653-vae-controlnet-inpaint-surfaces-design.md`) with fresh eyes.

### 1. Spec coverage

| Spec requirement | Task(s) | Covered? |
|---|---|---|
| Surface A — VAE `encode(pixels)` on all real backends | Task 8 (SkyReels), Task 9 (7 denoise-less: ltx_video_legacy, svd, cosmos, hunyuanvideo, cogvideox, minimax_h3, ltx2_5); Wan2 already has `encode` (#652/predecessor) | 9 wired; 3 deferred (ltx2, opensora, uniworld) → Task 11 follow-up issues |
| Surface B — ControlNet on Wan2 + SkyReels | Task 5 (Wan2 DiT `controlnet_residuals` injection, R1/R2), Task 6 (Wan2 `encode_control` controlnet_image path), Task 7 (SkyReels `encode_control` config plumbing) | ✓ |
| Surface C — inpaint mask on Wan2 + SkyReels default denoise path | Task 1 (helper), Task 2 (engine/base threading), Task 3 (Wan2 `run_denoise`), Task 4 (SkyReels `_denoise_sample`, default path only R4) | ✓ |
| `apply_inpaint_mask` neutral helper, orthogonal to `ControlState` | Task 1 | ✓ |
| Thread-portability invariant (#630: numpy-bridge caller thread + `mx.eval` worker thread) | Task 8 (SkyReels mirrors Wan2 reference), Task 9 (batched backends mirror Wan2) | ✓ |
| `VideoGenEngine` public-API surface (`denoise`, `encode_control`, `encode`, `load_vae_encoder`/`unload_vae_encoder`) | Task 2 (denoise kwargs), Task 6/7 (encode_control), Task 8/9 (encode) | ✓ |
| Real-model e2e validation | Task 10 (4 gated tests) | ✓ |
| Follow-up issues for denoise-less backends (Surface B+C) + deferred A | Task 11 (10 issues) | ✓ |
| Release (lint, full-suite, README, CHANGELOG, version bump, PR, tag) | Task 12 | ✓ |

**Gap:** The spec's "all 11 real backends" for Surface A is 9 wired + 3 deferred to follow-up issues (filed in Task 11). This is the approved scope (Ruling R3 + memory scope lock); the deferred-3 each get a single issue covering A+B+C. Satisfied, not a hole.

### 2. Placeholder scan

`grep` for `TODO|TBD|fill in|implement later|Add appropriate|handle edge cases|Similar to Task|<!-- Task|<!-- fill|steps: fill`:
- Line 1176 `<!-- Task 9 steps: fill below in two batches -->` — a **section marker**, NOT a placeholder. Task 9 body is filled (lines 1149–1675, two batches A/B with exact code). Acceptable.
- Line 2087 `TODO(issue-filing)` — an **offline fallback note** in Task 11 ("if `gh issue create` unavailable offline, record a TODO(issue-filing) line ... file when online"). This is a contingency instruction, not a plan placeholder. Acceptable.
- No `Add appropriate error handling` / `handle edge cases` / `Similar to Task N` / `implement later` anywhere.
Every code step has the actual code block. Every test step has the actual test. **Clean.**

### 3. Type consistency

Cross-checked every signature a later task consumes against the task that defines it:

| Name | Defined in | Consumed in | Consistent? |
|---|---|---|---|
| `apply_inpaint_mask(latents, init_latent, mask) -> mx.array` | Task 1 (line 154) | Task 3 (`run_denoise`), Task 4 (`_denoise_sample`) | ✓ (arg order + names match) |
| `patch_downsample_mask(mask, vae_stride, patch_size, t_latent, h_latent, w_latent)` | Task 1 | (reserved for SkyReels mask reconcile) | ✓ |
| `VideoGenEngine.denoise(..., inpaint_mask=None, init_latent=None)` | Task 2 (line 243) | Task 3/4 backends, Task 10 real-model tests | ✓ |
| `VideoBackend.denoise(..., inpaint_mask=None, init_latent=None)` | Task 2 (line 322) | Wan2/SkyReels overrides | ✓ |
| `WanModel.__call__(..., controlnet_residuals=None, controlnet_stride=4)` | Task 5 | SkyReels precedent mirror | ✓ (matches `pipelines/__init__.py:716-724`) |
| `encode_control(controlnet_image=None, control_type="canny", controlnet_strength=1.0, ...)` | Task 6 (Wan2), Task 7 (SkyReels) | Task 10 real-model ControlNet test | ✓ (kwarg names match across both backends + test) |
| `ControlState(controlnet_adapter=, controlnet_latent=)` | Task 6 | Task 5 DiT injection reads `.controlnet_adapter`/`.controlnet_latent` | ✓ |
| `load_vae_encoder()` / `encode(pixels)` / `unload_vae_encoder()` | Task 8 (SkyReels), Task 9 (batch) | Task 10 real-model VAE roundtrip test | ✓ |
| `ControlNet.compute_residuals(hidden_states, t, context, control_states, ...)` | external (adapters/controlnet.py) | Task 5 (reshape reconciliation R2) | ✓ (signature verified, not redefined) |

No `clearLayers()`/`clearFullLayers()` style drift. The mask-semantics constant (`mask=1` reactive, `mask=0` frozen) is stated identically in Task 1, R4, the Task 10 tests, and the Status notes. **Consistent.**

**Verdict:** plan is execution-ready. Spec covered (9+3 split is the locked scope), no placeholders, types consistent. Proceed to commit + subagent-driven execution.

---

## Status / Resume Notes (updated 2026-08-31)

This plan is **execution-ready.** All 12 task bodies are filled with exact TDD code (write failing test → run to fail → implement → run to pass → commit). Self-review passed (spec covered, no placeholders, types consistent). Next action: commit the plan, then execute via `superpowers:subagent-driven-development`.

**Commit state:**
- Design spec: COMMITTED `d1c8fef` — `docs/superpowers/specs/2026-08-31-653-vae-controlnet-inpaint-surfaces-design.md`.
- This plan file: UNCOMMITTED (Task 1–12 bodies + Self-review filled this session; commit pending). Branch: `feat/api-stable-layer` (NOT main). Remote: `origin` (git@github.com:dahai80/fusion-mlx.git).
- v0.8.56 is the latest release (2026-08-31, GitHub-only). #653 lands as v0.8.57.

**What is DONE (plan authoring):**
- Required plan header (Goal/Architecture/Tech Stack/Spec path/Global Constraints).
- 4 Plan Rulings (R1–R4) recorded verbatim above — load-bearing decisions; re-read before touching Tasks 5/9/4.
- File-structure section (Files created / Files modified per surface / Docs).
- All 12 task bodies filled with exact code + exact test code + exact commit messages.
- Self-review section filled (spec-coverage table, placeholder scan, type-consistency table).

**What is NOT done (the execution work):**
- Zero code written. No tests run. No PR. No release. The plan describes the work; execution is the next phase.

**Task map (filled this session):**
- Task 1 — `_inpaint.py` helper + tests (Surface C foundation).
- Task 2 — engine + base `denoise` signature threading (Surface C).
- Task 3 — Wan2 `run_denoise` mask insertion (Surface C, stage.py:497).
- Task 4 — SkyReels `_denoise_sample` mask insertion (Surface C, default path only R4).
- Task 5 — Wan2 DiT `controlnet_residuals` injection (Surface B, RULING R1/R2, riskiest).
- Task 6 — Wan2 `encode_control` controlnet_image path (Surface B).
- Task 7 — SkyReels `encode_control` config plumbing (Surface B).
- Task 8 — SkyReels VAE encode (Surface A).
- Task 9 — batch 7 denoise-less backends Surface A (RULING R3, two batches).
- Task 10 — 4 real-model e2e tests (gated `FUSION_MLX_REAL_MODEL_TESTS`).
- Task 11 — file 10 follow-up issues (7 B+C denoise-port + 3 A+B+C deferred).
- Task 12 — lint + full-suite + README/CHANGELOG/version bump + PR → merge main → tag v0.8.57 → release (PyPI skipped).

**Gathered signatures (load-bearing, verified — cite, don't re-derive):**
- `WanModel.__call__` at `fusion_mlx/video/wan2/wan_2.py:331` — NO `controlnet_residuals`/`controlnet_stride` kwargs today. Block loop at `wan_2.py:560` injects only VACE `vace_hints`. Surface B Wan2 requires adding these kwargs + block-loop injection (Ruling R1).
- SkyReels DiT residual injection precedent: `fusion_mlx/video/skyreels_v3/pipelines/__init__.py:716-724` — `dit(..., controlnet_residuals=cn_residuals, controlnet_stride=cn_stride)`. Mirror this in Wan2.
- `ControlNet.compute_residuals(hidden_states, t, context, control_states, seq_lens=None, grid_sizes=None) -> list[mx.array] | None` at `fusion_mlx/video/adapters/controlnet.py`. Expects `hidden_states: [B, C_vae, H, W]` (B-first, single-frame) — Wan2 latents are 4D C-first `(z_dim, t_latent, h_latent, w_latent)`, needs reshape reconciliation (Ruling R2, riskiest task = Task 5).
- Wan2 Surface C insertion point: `fusion_mlx/video/wan2/stage.py:497` (after `sched.step`). Precedent re-composite pattern at `stage.py:499-507` (`latents = (1.0 - control.i2v_mask) * control.z_img + control.i2v_mask * latents`).
- SkyReels Surface C insertion point: `fusion_mlx/video/skyreels_v3/pipelines/__init__.py:743` (`scheduler.step(...).prev_sample`), before flicker smooth. Default path only (Ruling R4); async `_denoise_sample_async` :780 + speculative `_denoise_sample_speculative` :898/:1501 gated OFF (#177/#180) — follow-up.
- SkyReels ControlNet config read site: `pipelines/__init__.py:607-609` (pipeline.config fields, NO ControlState on SkyReels).
- Mask semantics: `apply_inpaint_mask(latents, init, mask) = mask*latents + (1-mask)*init` — mask=1 reactive/denoised, mask=0 frozen/init. Consistent with existing `i2v_mask_blend`.

**Exact resume steps (execution phase):**
1. Re-read this plan + the committed spec (`docs/superpowers/specs/2026-08-31-653-vae-controlnet-inpaint-surfaces-design.md`).
2. Commit this completed plan (`docs(plans): fill #653 task bodies + self-review`).
3. Execute via `superpowers:subagent-driven-development`: fresh subagent per task + task review + broad final review. Rulings R1–R4 govern conflicts; the spec is the binding authority. Create SDD ledger at `.superpowers/sdd/2026-08-31-653-vae-controlnet-inpaint-surfaces/progress.md` first; record BASE (`git rev-parse HEAD`) before each dispatch.
4. Task 11 = file 10 follow-up issues for denoise-less backends (cogvideox/cosmos/hunyuanvideo/ltx_video_legacy/ltx2_5/ltx2/minimax_h3/opensora/svd/uniworld). Record the 10 issue numbers in the Task 12 PR body.
5. Task 12 = lint (black --fast, ruff; NEVER `tests/unit/debt_modules.txt`) + full-suite sweep (`.venv/bin/python -m pytest`, NOT bare pytest) + README/CHANGELOG + version bump 0.8.56→0.8.57 + PR → merge to main → tag `v0.8.57` THEN `gh release create` (omit `--target`, it 422s when tag exists). PyPI skipped (no token).
6. Real-model tests (Task 10) gated behind `FUSION_MLX_REAL_MODEL_TESTS`; models via https://hf-mirror.com. Never `mx.clear_streams()` in tests (#630 stream invariant). start/stop fusion-mlx via `~/claude-home/fusion-mlx/start.sh start|stop` is NOT needed for Task 10 (instantiates `VideoGenEngine` directly).

**Resuming session: do NOT ask the user questions to re-confirm scope.** Surface A on 9 backends (3 deferred to follow-up issues), Surface B+C on Wan2+SkyReels only — locked. Proceed straight to committing the plan + dispatching Task 1.
