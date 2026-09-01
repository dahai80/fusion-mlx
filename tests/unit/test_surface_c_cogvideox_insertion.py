import inspect

from fusion_mlx.video.cogvideox.generate import run_denoise


def test_run_denoise_accepts_inpaint_params_default_none():
    sig = inspect.signature(run_denoise)
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
    assert sig.parameters["inpaint_mask"].default is None
    assert sig.parameters["init_latent"].default is None


def test_run_denoise_calls_apply_inpaint_mask():
    src = inspect.getsource(run_denoise)
    assert "apply_inpaint_mask(latents, init_latent, inpaint_mask)" in src
    assert "inpaint_mask is not None" in src


def test_run_denoise_re_composite_after_step():
    # Surface C must fire AFTER each sched.step (frozen-region restore),
    # not before. Verify the ordering in source: step() call precedes the
    # apply_inpaint_mask call within the per-step loop.
    src = inspect.getsource(run_denoise)
    step_pos = src.find("sched.step(")
    inpaint_pos = src.find("apply_inpaint_mask(")
    assert step_pos != -1
    assert inpaint_pos != -1
    assert step_pos < inpaint_pos, "inpaint re-composite must follow sched.step"
