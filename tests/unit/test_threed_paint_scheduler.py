# SPDX-License-Identifier: Apache-2.0
# #989 Hunyuan3D-2.1 paint — Session 4 scheduler tests (DDIM v_prediction).
from __future__ import annotations

import numpy as np
import pytest

from fusion_mlx.threed.paint.scheduler import DDIMScheduler, make_scheduler_from_config

_SCHED_CFG = {
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "num_train_timesteps": 1000,
    "prediction_type": "v_prediction",
    "set_alpha_to_one": True,
    "steps_offset": 1,
    "timestep_spacing": "trailing",
    "rescale_betas_zero_snr": True,
    "clip_sample": False,
}


def test_scheduler_alpha_bounds():
    s = DDIMScheduler()
    assert 0.0 < float(s.alphas_cumprod[0]) <= 1.0
    # zero-SNR: final cumprod near 0 (not NaN).
    assert float(s.alphas_cumprod[-1]) < 0.01
    assert np.all(np.isfinite(s.alphas_cumprod))
    assert float(s.final_alpha_cumprod) == 1.0  # set_alpha_to_one


def test_scheduler_timesteps_trailing_descending():
    s = DDIMScheduler()
    ts = s.set_timesteps(8)
    assert len(ts) == 8
    # Descending (high -> low) for denoise.
    assert all(ts[i] > ts[i + 1] for i in range(len(ts) - 1))
    # All within [0, num_train-1] (no OOB into alphas_cumprod).
    assert int(ts.max()) <= 999
    assert int(ts.min()) >= 0


def test_scheduler_step_count():
    for n in (4, 8, 30):
        s = DDIMScheduler()
        ts = s.set_timesteps(n)
        assert len(ts) == n


def test_scheduler_vpred_step_finite():
    s = DDIMScheduler()
    s.set_timesteps(8)
    x0 = np.zeros((1, 4, 8, 8), dtype=np.float32)
    noise = np.random.randn(1, 4, 8, 8).astype(np.float32)
    t = int(s.timesteps[0])
    xt = s.add_noise(x0, noise, t)
    out = np.random.randn(1, 4, 8, 8).astype(np.float32) * 0.1
    prev = s.step(out, t, xt)
    assert prev.shape == xt.shape
    assert bool(np.all(np.isfinite(prev)))


def test_scheduler_vpred_roundtrip():
    # With the TRUE v-prediction model output, one DDIM step from xt should
    # move toward x0 (pred_prev closer to x0 than xt was).
    s = DDIMScheduler()
    s.set_timesteps(8)
    x0 = np.random.randn(1, 4, 8, 8).astype(np.float32) * 0.5
    noise = np.random.randn(1, 4, 8, 8).astype(np.float32)
    t = int(s.timesteps[0])
    a = s._alpha_prod_t(t)
    sa, sm = a**0.5, (1.0 - a) ** 0.5
    xt = sa * x0 + sm * noise
    # True v = sa*noise - sm*x0
    v_true = sa * noise - sm * x0
    prev = s.step(v_true, t, xt)
    err_xt = float(np.abs(xt - x0).mean())
    err_prev = float(np.abs(prev - x0).mean())
    assert err_prev < err_xt  # step moves toward x0


def test_scheduler_from_config():
    s = make_scheduler_from_config(_SCHED_CFG)
    assert s.prediction_type == "v_prediction"
    assert s.timestep_spacing == "trailing"
    s.set_timesteps(30)
    assert len(s.timesteps) == 30


def test_scheduler_unknown_beta_schedule():
    with pytest.raises(ValueError):
        DDIMScheduler(beta_schedule="nope")
