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
