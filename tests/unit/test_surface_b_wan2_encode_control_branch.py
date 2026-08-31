import inspect

from fusion_mlx.engines.video_backends.wan2 import Wan2Backend


def test_encode_control_accepts_controlnet_kwargs():
    sig = inspect.signature(Wan2Backend.encode_control)
    assert "controlnet_image" in sig.parameters
    assert "control_type" in sig.parameters
    assert "controlnet_strength" in sig.parameters
    assert sig.parameters["controlnet_strength"].default == 1.0
    assert sig.parameters["control_type"].default == "canny"


def test_encode_control_branches_on_controlnet_image():
    src = inspect.getsource(Wan2Backend.encode_control)
    assert "controlnet_image" in src
    assert "ControlState(" in src
    assert "controlnet_adapter=" in src
    assert "controlnet_latent=" in src
