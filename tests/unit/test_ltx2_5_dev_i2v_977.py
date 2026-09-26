# SPDX-License-Identifier: Apache-2.0
# #977: ltx2_5 dev I2V single-stage conditioning wiring. Verifies the dev path
# no longer raises NotImplementedError, and the shared _build_i2v_state helper
# builds a LatentState with the condition frame frozen clean (mask=0) and free
# frames renoised to sig[0]. No real 22B model load (OOM gate).
from __future__ import annotations

import ast
import inspect

import mlx.core as mx

from fusion_mlx.video.ltx2.conditioning import LatentState
from fusion_mlx.video.ltx2_5 import generate as gen_mod


def test_build_i2v_state_condition_frame_frozen_clean():
    b, c, f, h, w = 1, 128, 4, 2, 2
    latent_shape = (b, c, f, h, w)
    image_latent = mx.ones((1, c, 1, h, w)) * 7.0
    mx.random.seed(0)
    state = gen_mod._build_i2v_state(
        image_latent,
        latent_shape,
        noise_scale=0.909,
        image_frame_idx=0,
        image_strength=1.0,
        model_dtype=mx.float32,
    )
    mx.eval(state.latent, state.clean_latent, state.denoise_mask)
    assert isinstance(state, LatentState)
    # strength=1.0 -> condition frame mask = 0 (frozen clean)
    assert mx.all(state.denoise_mask[:, :, 0:1] == 0.0).item()
    # free frames mask = 1 (renoised)
    assert mx.all(state.denoise_mask[:, :, 1:] == 1.0).item()
    # clean_latent at frame 0 == image latent
    assert mx.all(state.clean_latent[:, :, 0:1] == 7.0).item()
    # condition frame latent = image (apply_conditioning splices image into
    # latent; mask=0 -> noise*0 + base*1 = base = image_latent)
    assert mx.all(state.latent[:, :, 0:1] == 7.0).item()


def test_build_i2v_state_free_frames_renoised():
    b, c, f, h, w = 1, 2, 3, 1, 1
    latent_shape = (b, c, f, h, w)
    image_latent = mx.ones((1, c, 1, h, w)) * 5.0
    mx.random.seed(1)
    state = gen_mod._build_i2v_state(
        image_latent,
        latent_shape,
        noise_scale=1.0,
        image_frame_idx=0,
        image_strength=1.0,
        model_dtype=mx.float32,
    )
    mx.eval(state.latent)
    # noise_scale=1.0, free frame mask=1 -> latent = noise*1 + base*0 = noise (!=0)
    free = state.latent[:, :, 1:]
    assert not mx.all(free == 0.0).item()
    # condition frame latent = image (5.0)
    assert mx.all(state.latent[:, :, 0:1] == 5.0).item()


def test_build_i2v_state_partial_strength_keeps_some_noise_on_condition():
    b, c, f, h, w = 1, 2, 2, 1, 1
    latent_shape = (b, c, f, h, w)
    image_latent = mx.ones((1, c, 1, h, w)) * 3.0
    mx.random.seed(2)
    state = gen_mod._build_i2v_state(
        image_latent,
        latent_shape,
        noise_scale=1.0,
        image_frame_idx=0,
        image_strength=0.5,
        model_dtype=mx.float32,
    )
    mx.eval(state.denoise_mask)
    # strength=0.5 -> condition mask = 1-0.5 = 0.5 (partially denoised)
    assert mx.allclose(
        state.denoise_mask[:, :, 0:1], mx.array([[[0.5]]]), atol=1e-5
    ).item()


def _dev_branch_source() -> str:
    # Extract the `if var_str == "dev":` branch body source from
    # generate_video. PR#985 shipped the I2V wiring (_build_i2v_state call)
    # but left a dead `if image is not None: raise NotImplementedError`
    # BEFORE it, so the wiring never ran and dev+image returned 500. This
    # helper lets the regression test inspect the branch structurally
    # without loading the 22B model (OOM gate).
    src = inspect.getsource(gen_mod.generate_video)
    tree = ast.parse(src)
    dev_body: list[ast.stmt] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "var_str"
            and test.ops
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "dev"
        ):
            dev_body = node.body
            break
    assert dev_body, "dev branch (`if var_str == 'dev':`) not found"
    return ast.unparse(ast.Module(body=dev_body, type_ignores=[]))


def test_dev_branch_has_no_image_notimplemented_raise():
    # Regression guard for PR#985/#977: the dev branch must NOT contain a
    # `raise NotImplementedError` gated on `image is not None` (that was the
    # dead-code block that shadowed the I2V wiring and caused the 500 the
    # user hit). Only the audio A/V follow-up raise is allowed.
    dev_src = _dev_branch_source()
    tree = ast.parse(dev_src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        func = node.exc.func
        name = func.id if isinstance(func, ast.Name) else ""
        if name != "NotImplementedError":
            continue
        # Find the enclosing `if image is not None:` — walk parents via the
        # parsed subtree is awkward, so check the raise is not directly inside
        # an image-guarded block by scanning the unparsed source around it.
        raise_src = ast.unparse(node)
        idx = dev_src.find(raise_src)
        assert idx >= 0
        # The allowed audio raise is gated on `if audio:`. Confirm no raise
        # sits under an `if image is not None:` guard.
        before = dev_src[:idx]
        last_if = before.rfind("if image is not None")
        last_audio = before.rfind("if audio")
        assert last_audio > last_if, (
            "dev branch has a NotImplementedError raise not guarded by "
            "`if audio:` — an image-guarded raise would shadow the I2V "
            "wiring (#977 regression). raise:\n" + raise_src
        )


def test_dev_branch_calls_build_i2v_state_when_image_set():
    # The I2V wiring (_build_i2v_state) must be present inside the dev
    # branch and reachable when `image is not None`. PR#985 had the call
    # but it was dead code under the deleted raise; this asserts the call
    # exists in the branch so a future refactor can't silently drop it.
    dev_src = _dev_branch_source()
    assert (
        "_build_i2v_state(" in dev_src
    ), "dev branch must call _build_i2v_state for I2V conditioning (#977)"
    assert "image is not None" in dev_src or "image is not  None" in dev_src
    # encode path also required
    assert "_encode_image_latent(" in dev_src
