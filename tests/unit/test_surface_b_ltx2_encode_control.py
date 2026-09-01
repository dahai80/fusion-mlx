import asyncio
import inspect

from fusion_mlx.engines.video_backends.ltx2 import LTX2Backend
from fusion_mlx.video.ltx2.denoise import denoise_distilled


def test_denoise_distilled_accepts_surface_kwargs():
    sig = inspect.signature(denoise_distilled)
    assert "controlnet_image" in sig.parameters
    assert "inpaint_mask" in sig.parameters
    assert "init_latent" in sig.parameters
    assert sig.parameters["controlnet_image"].default is None
    assert sig.parameters["inpaint_mask"].default is None
    assert sig.parameters["init_latent"].default is None


def test_denoise_distilled_has_controlnet_fail_visible_gate():
    src = inspect.getsource(denoise_distilled)
    assert "controlnet_image is not None" in src
    assert "RuntimeError" in src


def test_encode_control_raises_on_controlnet_image():
    backend = LTX2Backend.__new__(LTX2Backend)

    async def _call():
        return await backend.encode_control(controlnet_image="canny.png")

    try:
        asyncio.run(_call())
    except RuntimeError as exc:
        assert "Surface B" in str(exc)
        assert "#738" in str(exc)
    else:
        raise AssertionError(
            "encode_control should raise RuntimeError on controlnet_image"
        )


def test_encode_control_returns_none_on_pure_t2v():
    backend = LTX2Backend.__new__(LTX2Backend)

    async def _call():
        return await backend.encode_control()

    result = asyncio.run(_call())
    assert result is None
