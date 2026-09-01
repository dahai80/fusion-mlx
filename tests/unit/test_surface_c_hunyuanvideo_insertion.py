import inspect

from fusion_mlx.video.hunyuanvideo.generate import generate_video


def test_generate_video_recomposites_with_apply_inpaint_mask():
    src = inspect.getsource(generate_video)
    assert "apply_inpaint_mask(latents, init_latent, inpaint_mask)" in src
    assert "inpaint_mask is not None" in src


def test_apply_inpaint_mask_after_scheduler_step():
    src = inspect.getsource(generate_video)
    step_pos = src.find("scheduler.step(")
    inpaint_pos = src.find("apply_inpaint_mask(")
    assert step_pos != -1, "scheduler.step( not found in generate_video source"
    assert inpaint_pos != -1, "apply_inpaint_mask( not found in generate_video source"
    assert (
        step_pos < inpaint_pos
    ), "apply_inpaint_mask must come AFTER scheduler.step (re-composite after step)"
