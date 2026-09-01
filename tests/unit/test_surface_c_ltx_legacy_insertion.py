import inspect

from fusion_mlx.video.ltx_video_legacy.denoise import denoise


def test_denoise_calls_apply_inpaint_mask():
    src = inspect.getsource(denoise)
    assert "apply_inpaint_mask(latents, init_latent, inpaint_mask)" in src
    assert "inpaint_mask is not None" in src


def test_denoise_re_composite_after_step():
    src = inspect.getsource(denoise)
    step_pos = src.find("scheduler.step(")
    inpaint_pos = src.find("apply_inpaint_mask(")
    assert step_pos != -1
    assert inpaint_pos != -1
    assert step_pos < inpaint_pos, "inpaint re-composite must follow scheduler.step"
