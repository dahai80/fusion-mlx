# SPDX-License-Identifier: Apache-2.0
import inspect

from fusion_mlx.video.cosmos.generate import generate_video


def test_generate_video_accepts_inpaint_params_default_none():
    sig = inspect.signature(generate_video)
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
    assert sig.parameters["inpaint_mask"].default is None
    assert sig.parameters["init_latent"].default is None


def test_generate_video_calls_apply_inpaint_mask():
    src = inspect.getsource(generate_video)
    assert "apply_inpaint_mask(latents, init_latent, inpaint_mask)" in src
    assert "inpaint_mask is not None" in src


def test_generate_video_re_composite_after_step():
    # Surface C must fire AFTER each scheduler.step (frozen-region restore),
    # not before. Verify the ordering in source: step() call precedes the
    # apply_inpaint_mask call within the per-step loop.
    src = inspect.getsource(generate_video)
    step_pos = src.find("scheduler.step(")
    inpaint_pos = src.find("apply_inpaint_mask(")
    assert step_pos != -1
    assert inpaint_pos != -1
    assert step_pos < inpaint_pos, "inpaint re-composite must follow scheduler.step"
