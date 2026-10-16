# SPDX-License-Identifier: Apache-2.0
"""#1202: DPM-Solver++(2M) flow-matching contrib scheduler for mflux.

Math-level tests: Euler fallback on step 0, 2nd-order multistep correction
on later steps, state reset per denoise loop, registry wiring so
scheduler="dpmpp_2m" resolves through mflux Config.
"""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

mflux = pytest.importorskip("mflux")

from fusion_mlx.engines.image_schedulers import (  # noqa: E402
    DPMPP2MSdeScheduler,
    register_image_schedulers,
)


def _config(num_steps=4):
    cfg = MagicMock()
    cfg.num_inference_steps = num_steps
    cfg.model_config = MagicMock()
    return cfg


class TestScheduleShape:
    def test_sigmas_shape_and_bounds(self):
        sched = DPMPP2MSdeScheduler(_config(8))
        assert sched.sigmas.shape == (9,)  # num_steps + terminal zero
        assert float(sched.sigmas[0]) == 1.0
        assert float(sched.sigmas[-1]) == 0.0
        # monotonically decreasing
        s = [float(x) for x in sched.sigmas]
        assert all(a >= b for a, b in zip(s, s[1:]))

    def test_timesteps_shape(self):
        sched = DPMPP2MSdeScheduler(_config(8))
        assert sched.timesteps.shape == (8,)


class TestIntegrator:
    def test_step0_is_euler(self):
        sched = DPMPP2MSdeScheduler(_config(4))
        latents = mx.zeros((2, 4), dtype=mx.float32)
        v = mx.ones((2, 4), dtype=mx.float32)
        sigmas = sched.sigmas
        h0 = float(sigmas[0] - sigmas[1])
        out = sched.step(noise=v, timestep=0, latents=latents, sigmas=sigmas)
        mx.eval(out)
        assert abs(float(out[0, 0]) + h0) < 1e-5

    def test_later_step_applies_second_order_correction(self):
        sched = DPMPP2MSdeScheduler(_config(4))
        sigmas = sched.sigmas
        latents = mx.zeros((2, 4), dtype=mx.float32)
        v0 = mx.ones((2, 4), dtype=mx.float32)
        sched.step(noise=v0, timestep=0, latents=latents, sigmas=sigmas)
        h0 = float(sigmas[0] - sigmas[1])
        v1 = mx.full((2, 4), 2.0, dtype=mx.float32)
        x1 = latents - h0 * v0
        out = sched.step(noise=v1, timestep=1, latents=x1, sigmas=sigmas)
        mx.eval(out)
        h1 = float(sigmas[1] - sigmas[2])
        # AB2: x2 = x1 - h1*v1 - (h1^2/(2*h0))*(v1 - v0), x1 = -h0*v0
        expected = -h0 - h1 * 2.0 - (h1 * h1 / (2 * h0)) * (2.0 - 1.0)
        assert abs(float(out[0, 0]) - expected) < 1e-4

    def test_constant_velocity_matches_euler_exactly(self):
        # With v_i == v_{i-1} the multistep correction vanishes — the
        # scheduler must agree with plain Euler (sanity of the sign).
        sched = DPMPP2MSdeScheduler(_config(6))
        sigmas = sched.sigmas
        v = mx.full((2, 4), 0.5, dtype=mx.float32)
        x = mx.zeros((2, 4), dtype=mx.float32)
        for t in range(6):
            x = sched.step(noise=v, timestep=t, latents=x, sigmas=sigmas)
        mx.eval(x)
        euler = -0.5 * float(sigmas[0] - sigmas[6])
        assert abs(float(x[0, 0]) - euler) < 1e-4

    def test_matches_mflux_euler_step0(self):
        # Ground-truth cross-check against mflux's own Euler on the same
        # inputs — the step-0 fallback must be bit-for-bit the same update.
        from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
            FlowMatchEulerDiscreteScheduler,
        )

        cfg = _config(4)
        ours = DPMPP2MSdeScheduler(cfg)
        theirs = FlowMatchEulerDiscreteScheduler(cfg)
        latents = mx.random.normal((2, 4), dtype=mx.float32)
        v = mx.random.normal((2, 4), dtype=mx.float32)
        a = ours.step(noise=v, timestep=0, latents=latents, sigmas=ours.sigmas)
        b = theirs.step(noise=v, timestep=0, latents=latents, sigmas=theirs.sigmas)
        mx.eval(a, b)
        assert float(mx.abs(a - b).max()) < 1e-6

    def test_linear_velocity_field_exact(self):
        # For v(sigma) linear in sigma the AB2 extrapolation is exact, so
        # the ONLY error is the step-0 Euler local error h0^2/2: the
        # integrated result must equal -0.5 - h0^2/2 analytically. This
        # pins both the direction (minus) and the 2nd-order exactness of
        # every subsequent step.
        sched = DPMPP2MSdeScheduler(_config(4))
        sigmas = sched.sigmas
        x = mx.zeros((1, 1), dtype=mx.float32)
        for t in range(4):
            v = mx.full((1, 1), float(sigmas[t]), dtype=mx.float32)
            x = sched.step(noise=v, timestep=t, latents=x, sigmas=sigmas)
        mx.eval(x)
        h0 = float(sigmas[0] - sigmas[1])
        expected = -0.5 - h0 * h0 / 2.0
        assert abs(float(x[0, 0]) - expected) < 1e-4

    def test_state_resets_on_new_loop(self):
        sched = DPMPP2MSdeScheduler(_config(4))
        sigmas = sched.sigmas
        latents = mx.zeros((2, 4), dtype=mx.float32)
        v = mx.ones((2, 4), dtype=mx.float32)
        for t in range(4):
            sched.step(noise=v, timestep=t, latents=latents, sigmas=sigmas)
        assert sched._prev_v is not None
        # A second generate() on the same scheduler starts at t=0 again and
        # must not see the stale previous-velocity state.
        out = sched.step(noise=v, timestep=0, latents=latents, sigmas=sigmas)
        mx.eval(out)
        h0 = float(sigmas[0] - sigmas[1])
        assert abs(float(out[0, 0]) + h0) < 1e-5


class TestRegistration:
    def test_register_makes_dpmpp_2m_resolvable(self):
        register_image_schedulers()
        from mflux.models.common.schedulers import SCHEDULER_REGISTRY

        assert SCHEDULER_REGISTRY.get("dpmpp_2m") is DPMPP2MSdeScheduler
        assert SCHEDULER_REGISTRY.get("dpmpp_2m_sde") is DPMPP2MSdeScheduler

    def test_registration_idempotent(self):
        register_image_schedulers()
        register_image_schedulers()
        from mflux.models.common.schedulers import SCHEDULER_REGISTRY

        assert SCHEDULER_REGISTRY.get("dpmpp_2m") is DPMPP2MSdeScheduler

    def test_config_scheduler_string_builds_instance(self):
        register_image_schedulers()
        from mflux.models.common.config.config import Config
        from mflux.models.common.config.model_config import ModelConfig

        cfg = Config(
            model_config=ModelConfig.schnell(),
            num_inference_steps=4,
            scheduler="dpmpp_2m",
        )
        assert isinstance(cfg.scheduler, DPMPP2MSdeScheduler)
