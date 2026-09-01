import asyncio
import inspect

from fusion_mlx.engines.video_backends.hunyuanvideo import HunyuanVideoBackend
from fusion_mlx.video.hunyuanvideo.generate import generate_video


def test_generate_video_signature_has_surface_kwargs():
    sig = inspect.signature(generate_video)
    for name in (
        "controlnet_image",
        "controlnet_adapter",
        "controlnet_latent",
        "inpaint_mask",
        "init_latent",
    ):
        assert name in sig.parameters
        assert sig.parameters[name].default is None


def test_generate_video_has_fail_visible_controlnet_gate():
    src = inspect.getsource(generate_video)
    assert "controlnet_image is not None" in src
    assert "RuntimeError" in src


def test_encode_control_raises_on_controlnet_image():
    backend = HunyuanVideoBackend.__new__(HunyuanVideoBackend)

    async def _call():
        await backend.encode_control(controlnet_image="foo.png")

    try:
        asyncio.run(_call())
    except RuntimeError:
        return
    raise AssertionError("encode_control should raise RuntimeError on controlnet_image")


def test_encode_control_returns_none_on_pure_t2v():
    backend = HunyuanVideoBackend.__new__(HunyuanVideoBackend)

    async def _call():
        return await backend.encode_control()

    result = asyncio.run(_call())
    assert result is None
