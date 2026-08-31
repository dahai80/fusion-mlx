import inspect

from fusion_mlx.engines.video_backends.skyreels import SkyReelsBackend


def test_skyreels_load_vae_encoder_override():
    src = inspect.getsource(SkyReelsBackend.load_vae_encoder)
    assert "NotImplementedError" not in src
    assert "self._stage_flags" in src or "pipeline" in src


def test_skyreels_encode_override():
    sig = inspect.signature(SkyReelsBackend.encode)
    assert "pixels" in sig.parameters
    src = inspect.getsource(SkyReelsBackend.encode)
    assert ".encode(" in src
    assert "NotImplementedError" not in src


def test_skyreels_unload_vae_encoder_override():
    src = inspect.getsource(SkyReelsBackend.unload_vae_encoder)
    assert "NotImplementedError" not in src


def test_skyreels_encode_numpy_bridge_thread():
    src = inspect.getsource(SkyReelsBackend.encode)
    assert "np.array" in src or "np.asarray" in src
    assert "run_in_executor" in src
    assert "mx.eval" in src
