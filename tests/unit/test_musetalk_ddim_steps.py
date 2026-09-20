# SPDX-License-Identifier: Apache-2.0
"""#927: MuseTalk pipeline runtime DDIM step-count control.

set_ddim_steps / generate_faces(steps) — default 1 = single-step t=0 inpaint
(released contract); n>1 = multi-step DDIM loop at decreasing t (FR-END-003
serious-tier thermal lever: 15→8 under throttle).
"""

import pytest

from fusion_mlx.video.musetalk_mlx import pipeline_mlx
from fusion_mlx.video.musetalk_mlx.config import UNET_TIMESTEP


class _FakeUNet:
    def __init__(self):
        self.calls = []  # list of (sample, timesteps, audio)

    def __call__(self, sample, timesteps, audio):
        t = int(timesteps[0]) if hasattr(timesteps, "__len__") else int(timesteps)
        self.calls.append(t)
        # echo sample shape so decode_latents can be stubbed
        return sample

    def parameters(self):
        return {}

    def eval(self):
        pass


class _FakeVAE:
    scaling_factor = 1.0

    def parameters(self):
        return {}

    def eval(self):
        pass


def _make_pipe(monkeypatch):
    unet = _FakeUNet()
    vae = _FakeVAE()
    pipe = pipeline_mlx.MuseTalkPipeline(vae, unet, whisper_encoder=None)
    # stub decode so we observe unet timesteps without real VAE
    monkeypatch.setattr(pipe, "decode_latents", lambda lat: lat)
    return pipe, unet


def test_default_single_step(monkeypatch):
    pipe, unet = _make_pipe(monkeypatch)
    import mlx.core as mx

    lat = mx.zeros((1, 8, 32, 32))
    audio = mx.zeros((1, 50, 384))
    pipe.generate_faces(lat, audio)
    assert unet.calls == [UNET_TIMESTEP]
    assert pipe._ddim_steps == 1


def test_set_ddim_steps_logs_change(monkeypatch, caplog):
    pipe, _ = _make_pipe(monkeypatch)
    with caplog.at_level("INFO"):
        pipe.set_ddim_steps(15)
    assert pipe._ddim_steps == 15
    assert any("ddim_steps" in r.message for r in caplog.records)


def test_set_ddim_steps_rejects_zero():
    pipe = pipeline_mlx.MuseTalkPipeline(_FakeVAE(), _FakeUNet(), None)
    with pytest.raises(ValueError):
        pipe.set_ddim_steps(0)


def test_multi_step_loop_decreasing_t(monkeypatch):
    pipe, unet = _make_pipe(monkeypatch)
    import mlx.core as mx

    lat = mx.zeros((1, 8, 32, 32))
    audio = mx.zeros((1, 50, 384))
    pipe.generate_faces(lat, audio, steps=4)
    assert len(unet.calls) == 4
    # decreasing t: first is max_t (999), last is 0
    assert unet.calls[0] == 999
    assert unet.calls[-1] == 0
    assert unet.calls == sorted(unet.calls, reverse=True)


def test_per_call_steps_overrides_runtime(monkeypatch):
    pipe, unet = _make_pipe(monkeypatch)
    pipe.set_ddim_steps(15)
    import mlx.core as mx

    lat = mx.zeros((1, 8, 32, 32))
    audio = mx.zeros((1, 50, 384))
    pipe.generate_faces(lat, audio, steps=1)  # per-call override
    assert len(unet.calls) == 1
    assert unet.calls[0] == UNET_TIMESTEP


def test_runtime_setting_used_when_steps_none(monkeypatch):
    pipe, unet = _make_pipe(monkeypatch)
    pipe.set_ddim_steps(3)
    import mlx.core as mx

    lat = mx.zeros((1, 8, 32, 32))
    audio = mx.zeros((1, 50, 384))
    pipe.generate_faces(lat, audio)  # steps=None -> use self._ddim_steps
    assert len(unet.calls) == 3


def test_ddim_timesteps_schedule():
    pipe = pipeline_mlx.MuseTalkPipeline(_FakeVAE(), _FakeUNet(), None)
    assert pipe._ddim_timesteps(1) == [UNET_TIMESTEP]
    ts = pipe._ddim_timesteps(3)
    assert ts[0] == 999 and ts[-1] == 0
    assert len(ts) == 3
