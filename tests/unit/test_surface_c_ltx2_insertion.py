import inspect

from fusion_mlx.video.ltx2.denoise import denoise_distilled


def test_denoise_distilled_accepts_inpaint_kwargs():
    sig = inspect.signature(denoise_distilled)
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
    assert sig.parameters["inpaint_mask"].default is None
    assert sig.parameters["init_latent"].default is None


def test_denoise_distilled_recomposites_in_loop_body():
    src = inspect.getsource(denoise_distilled)
    assert "apply_inpaint_mask(latents, init_latent, inpaint_mask)" in src
    assert "inpaint_mask is not None" in src
    for_line_pos = src.find("for i in range(num_steps)")
    assert for_line_pos != -1, "loop not found"
    apply_pos = src.find("apply_inpaint_mask(latents, init_latent, inpaint_mask)")
    assert apply_pos != -1, "apply_inpaint_mask call not found"
    assert (
        apply_pos > for_line_pos
    ), "apply_inpaint_mask must appear inside the loop body"


def test_denoise_dev_recomposites_in_loop_body():
    from fusion_mlx.video.ltx2.denoise import denoise_dev

    src = inspect.getsource(denoise_dev)
    assert "apply_inpaint_mask(latents, init_latent, inpaint_mask)" in src
    for_line_pos = src.find("for i in range(num_steps)")
    assert for_line_pos != -1, "loop not found"
    apply_pos = src.find("apply_inpaint_mask(latents, init_latent, inpaint_mask)")
    assert apply_pos != -1, "apply_inpaint_mask call not found"
    assert apply_pos > for_line_pos


def test_denoise_dev_av_recomposites_video_only():
    from fusion_mlx.video.ltx2.denoise import denoise_dev_av

    src = inspect.getsource(denoise_dev_av)
    assert "apply_inpaint_mask(video_latents, init_latent, inpaint_mask)" in src


def test_denoise_res2s_av_recomposites_video_only():
    from fusion_mlx.video.ltx2.denoise import denoise_res2s_av

    src = inspect.getsource(denoise_res2s_av)
    assert "apply_inpaint_mask(video_latents, init_latent, inpaint_mask)" in src
