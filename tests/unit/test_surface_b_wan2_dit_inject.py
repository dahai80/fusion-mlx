import inspect

from fusion_mlx.video.wan2.stage import ControlState, run_denoise
from fusion_mlx.video.wan2.wan_2 import WanModel


def test_control_state_has_controlnet_fields():
    fields = ControlState.__dataclass_fields__
    assert "controlnet_adapter" in fields
    assert "controlnet_latent" in fields


def test_wan_model_call_accepts_controlnet_kwargs():
    sig = inspect.signature(WanModel.__call__)
    assert "controlnet_residuals" in sig.parameters
    assert "controlnet_stride" in sig.parameters
    assert sig.parameters["controlnet_residuals"].default is None
    assert sig.parameters["controlnet_stride"].default == 4


def test_wan_model_block_loop_injects_controlnet_residuals():
    src = inspect.getsource(WanModel.__call__)
    assert "controlnet_residuals" in src
    assert "controlnet_stride" in src
    assert "i % controlnet_stride == 0" in src


def test_run_denoise_threads_controlnet_residuals():
    sig = inspect.signature(run_denoise)
    assert "controlnet_adapter" in sig.parameters
    assert "controlnet_latent" in sig.parameters
    src = inspect.getsource(run_denoise)
    assert "controlnet_residuals=" in src
    assert "compute_residuals" in src
    assert "controlnet_stride=" in src
