import inspect

from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsBasePipeline


def test_denoise_sample_accepts_inpaint_kwargs():
    sig = inspect.signature(SkyReelsBasePipeline._denoise_sample)
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
    assert sig.parameters["inpaint_mask"].default is None
    assert sig.parameters["init_latent"].default is None


def test_denoise_sample_recomposites_after_scheduler_step():
    src = inspect.getsource(SkyReelsBasePipeline._denoise_sample)
    assert "apply_inpaint_mask(latents, init_latent, inpaint_mask)" in src
    assert "inpaint_mask is not None" in src


def test_skyreels_backend_denoise_threads_inpaint_kwargs():
    from fusion_mlx.engines.video_backends.skyreels import SkyReelsBackend

    sig = inspect.signature(SkyReelsBackend.denoise)
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
