import inspect

from fusion_mlx.engines.video_backends.skyreels import SkyReelsBackend


def test_skyreels_encode_control_exists():
    sig = inspect.signature(SkyReelsBackend.encode_control)
    assert "controlnet_image" in sig.parameters
    assert "control_type" in sig.parameters
    assert "controlnet_strength" in sig.parameters
    assert sig.parameters["controlnet_strength"].default == 1.0
    assert sig.parameters["control_type"].default == "canny"


def test_skyreels_encode_control_assigns_config_fields():
    src = inspect.getsource(SkyReelsBackend.encode_control)
    assert "pipeline.config" in src
    assert "controlnet_image" in src
    assert "control_type" in src
    assert "controlnet_strength" in src
