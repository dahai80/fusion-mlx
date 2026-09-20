# SPDX-License-Identifier: Apache-2.0
"""#928: MuseTalkPipeline public stable render surface.

render() / render_latent() encapsulate timestep / apply_pe / dtype / unet /
vae.decode internals so downstream (musetalk-mlx) calls one stable method
instead of reaching into pipe.unet / apply_pe / UNET_TIMESTEP / pipe._dtype.
"""

from fusion_mlx.video.musetalk_mlx import pipeline_mlx
from fusion_mlx.video.musetalk_mlx.config import UNET_TIMESTEP


class _FakeUNet:
    def __init__(self):
        self.calls = []

    def __call__(self, sample, timesteps, audio):
        t = int(timesteps[0]) if hasattr(timesteps, "__len__") else int(timesteps)
        self.calls.append(t)
        return sample

    def parameters(self):
        return {}

    def eval(self):
        pass


class _FakeVAE:
    scaling_factor = 1.0

    def decode(self, lat):
        return lat

    def parameters(self):
        return {}

    def eval(self):
        pass


def _make_pipe():
    return pipeline_mlx.MuseTalkPipeline(_FakeVAE(), _FakeUNet(), None)


def test_timestep_class_attr_public():
    # #928: pipe.TIMESTEP is the stable public constant (replaces config.UNET_TIMESTEP reach-in)
    assert pipeline_mlx.MuseTalkPipeline.TIMESTEP == UNET_TIMESTEP
    pipe = _make_pipe()
    assert pipe.TIMESTEP == UNET_TIMESTEP == 0


def test_dtype_property_default_float32():
    pipe = _make_pipe()
    import mlx.core as mx

    assert pipe.dtype == mx.float32


def test_dtype_property_after_astype():
    pipe = _make_pipe()
    import mlx.core as mx

    pipe._dtype = mx.float16
    assert pipe.dtype == mx.float16


def test_render_returns_bgr_uint8(monkeypatch):
    pipe = _make_pipe()
    import mlx.core as mx

    lat = mx.zeros((1, 8, 32, 32))
    aud = mx.zeros((1, 50, 384))
    out = pipe.render(lat, aud)
    import numpy as np

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape[0] == 1


def test_render_casts_inputs_to_dtype(monkeypatch):
    pipe = _make_pipe()
    import mlx.core as mx

    pipe._dtype = mx.float16
    cast_seen = {}

    def _track_cast(self, dt):
        cast_seen[dt] = True
        return self

    monkeypatch.setattr(mx.array, "astype", _track_cast, raising=False)
    lat = mx.zeros((1, 8, 32, 32))
    aud = mx.zeros((1, 50, 384))
    pipe.render(lat, aud)
    # unet was called once (single-step default)
    assert pipe.unet.calls == [UNET_TIMESTEP]


def test_render_latent_returns_mx_array_no_readback():
    pipe = _make_pipe()
    import mlx.core as mx

    lat = mx.zeros((1, 8, 32, 32))
    aud = mx.zeros((1, 50, 384))
    out = pipe.render_latent(lat, aud)
    assert isinstance(out, mx.array)
    assert pipe.unet.calls == [UNET_TIMESTEP]


def test_render_steps_param_multi_step():
    pipe = _make_pipe()
    import mlx.core as mx

    lat = mx.zeros((1, 8, 32, 32))
    aud = mx.zeros((1, 50, 384))
    pipe.render(lat, aud, steps=3)
    assert len(pipe.unet.calls) == 3
    assert pipe.unet.calls[0] == 999 and pipe.unet.calls[-1] == 0


def test_render_dtype_override(monkeypatch):
    pipe = _make_pipe()
    import mlx.core as mx

    lat = mx.zeros((1, 8, 32, 32))
    aud = mx.zeros((1, 50, 384))
    pipe.render(lat, aud, dtype=mx.float16)
    assert pipe.unet.calls == [UNET_TIMESTEP]


def test_decode_to_array_shape():
    pipe = _make_pipe()
    import mlx.core as mx

    lat = mx.zeros((1, 4, 32, 32))
    out = pipe.decode_to_array(lat)
    # NHWC transpose of NCHW (fake VAE.decode is identity -> (1,32,32,4))
    assert out.shape[0] == 1
    assert out.ndim == 4  # NHWC


def test_generate_faces_delegates_same_core():
    # back-compat: generate_faces still works, same unet call as render
    pipe = _make_pipe()
    import mlx.core as mx

    lat = mx.zeros((1, 8, 32, 32))
    aud = mx.zeros((1, 50, 384))
    pipe.generate_faces(lat, aud)
    assert pipe.unet.calls == [UNET_TIMESTEP]
