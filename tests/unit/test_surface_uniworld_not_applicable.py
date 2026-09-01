import inspect

from fusion_mlx.video.uniworld.backend import UniWorldBackend


def _make_backend():
    b = UniWorldBackend.__new__(UniWorldBackend)
    b._loaded = True
    return b


def test_generate_rejects_controlnet_image():
    src = inspect.getsource(UniWorldBackend.generate)
    assert "controlnet_image is not None" in src
    assert "Surface B" in src
    assert "#740" in src


def test_generate_rejects_inpaint_mask():
    src = inspect.getsource(UniWorldBackend.generate)
    assert "params.inpaint_mask is not None" in src
    assert "Surface C" in src
    assert "#740" in src


def test_no_surface_methods_overridden():
    # UniWorld is image-only; the base NotImplementedError surface methods
    # (load_vae_encoder / encode / encode_control / unload_vae_encoder) must
    # remain inherited — they genuinely do not apply (no video VAE loop).
    from fusion_mlx.engines.video_backends.base import VideoBackend

    assert UniWorldBackend.load_vae_encoder is VideoBackend.load_vae_encoder
    assert UniWorldBackend.encode is VideoBackend.encode
    assert UniWorldBackend.encode_control is VideoBackend.encode_control
    assert UniWorldBackend.unload_vae_encoder is VideoBackend.unload_vae_encoder
