import inspect

from fusion_mlx.video.opensora.generate import generate_video


def test_generate_video_accepts_inpaint_kwargs():
    sig = inspect.signature(generate_video)
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
    assert sig.parameters["inpaint_mask"].default is None
    assert sig.parameters["init_latent"].default is None


def test_denoise_loop_recomposites_packed_latent():
    src = inspect.getsource(generate_video)
    assert "apply_inpaint_mask(x_packed, init_latent, inpaint_mask)" in src
    assert "inpaint_mask is not None" in src
    loop_pos = src.find("for step_idx, (t_cur, t_next) in enumerate(schedule)")
    assert loop_pos != -1, "schedule loop not found"
    apply_pos = src.find("apply_inpaint_mask(x_packed, init_latent, inpaint_mask)")
    assert apply_pos != -1, "apply_inpaint_mask call not found"
    assert apply_pos > loop_pos, "apply_inpaint_mask must appear inside the loop body"


def test_loop_inpaint_is_after_step_update():
    src = inspect.getsource(generate_video)
    update_pos = src.find("x_packed = x_packed + dt * model_out")
    assert update_pos != -1
    apply_pos = src.find("apply_inpaint_mask(x_packed, init_latent, inpaint_mask)")
    assert apply_pos > update_pos, "re-composite must follow the step update"
