import types

import mlx.core as mx

import fusion_mlx.video.skyreels_v3.pipelines as pm
from fusion_mlx.video.skyreels_v3.pipelines import (
    SkyReelsBasePipeline,
    SkyReelsPipelineConfig,
)


def _make_base() -> SkyReelsBasePipeline:
    p = SkyReelsBasePipeline.__new__(SkyReelsBasePipeline)
    p.config = SkyReelsPipelineConfig()
    return p


def _clear_env(monkeypatch):
    monkeypatch.delenv("FUSION_SKYREELS_DYNAMIC_CFG", raising=False)
    monkeypatch.delenv("FUSION_SKYREELS_CFG_KEEP_RATIO", raising=False)


def test_cfg_keep_steps_default(monkeypatch):
    _clear_env(monkeypatch)
    p = _make_base()
    assert p._cfg_keep_steps(30) == int(30 * 0.6)
    assert p._cfg_keep_steps(30) == 18


def test_cfg_keep_steps_disabled(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DYNAMIC_CFG", "0")
    p = _make_base()
    assert p._cfg_keep_steps(30) == 30
    assert p._cfg_keep_steps(7) == 7


def test_cfg_keep_steps_ratio_clamp(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DYNAMIC_CFG", "1")
    p = _make_base()
    monkeypatch.setenv("FUSION_SKYREELS_CFG_KEEP_RATIO", "1.5")
    assert p._cfg_keep_steps(10) == 10
    monkeypatch.setenv("FUSION_SKYREELS_CFG_KEEP_RATIO", "-0.2")
    assert p._cfg_keep_steps(10) == 0


def test_cfg_keep_steps_invalid_ratio(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DYNAMIC_CFG", "1")
    monkeypatch.setenv("FUSION_SKYREELS_CFG_KEEP_RATIO", "abc")
    p = _make_base()
    assert p._cfg_keep_steps(30) == 18


def test_cfg_keep_steps_custom_ratio(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DYNAMIC_CFG", "1")
    monkeypatch.setenv("FUSION_SKYREELS_CFG_KEEP_RATIO", "0.5")
    p = _make_base()
    assert p._cfg_keep_steps(4) == 2
    assert p._cfg_keep_steps(10) == 5


def _stub_denoise_deps(monkeypatch, n_steps: int):
    class FakeSched:
        def __init__(self, **kwargs):
            self.timesteps = mx.array([float(1000 - i * 200) for i in range(n_steps)])

        def set_timesteps(self, n):
            pass

        def step(self, mp, t, lat):
            return types.SimpleNamespace(prev_sample=lat)

    class FakeFix:
        def reset_step_filter(self):
            pass

        def filter_step(self, x):
            return x

        def smooth_temporal(self, x):
            return x

        def align_boundary(self, x, y):
            return x

    monkeypatch.setattr(pm, "FlowUniPCMultistepScheduler", lambda **k: FakeSched())
    monkeypatch.setattr(
        pm,
        "_flicker_cfg_for_branch",
        lambda b: types.SimpleNamespace(enable_boundary_align=False),
    )
    monkeypatch.setattr(pm, "TemporalFlickerFix", lambda cfg: FakeFix())
    monkeypatch.setattr(pm, "perform_guidance", lambda x, s: x[: x.shape[0] // 2])
    monkeypatch.setattr(pm.mx, "eval", lambda x: None)


def test_denoise_routes_dynamic_cfg(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DYNAMIC_CFG", "1")
    monkeypatch.setenv("FUSION_SKYREELS_CFG_KEEP_RATIO", "0.5")
    _stub_denoise_deps(monkeypatch, n_steps=4)

    p = _make_base()
    p.config.num_inference_steps = 4
    p.config.guidance_scale = 5.0
    p.step_strategy = types.SimpleNamespace(
        reset=lambda: None, set_current_step=lambda i: None
    )

    seen = []

    def fake_dit(lat, t, ctx, sl, gs, **kwargs):
        seen.append(int(lat.shape[0]))
        return mx.zeros_like(lat)

    p.dit = fake_dit

    latents = mx.zeros((1, 16, 2, 4, 4))
    context = mx.zeros((1, 257, 4096))
    p._denoise_sample(latents, context, seq_lens=[8], grid_sizes=[(1, 2, 2)])

    assert seen == [2, 2, 1, 1]


def test_denoise_routes_disabled_all_b2(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DYNAMIC_CFG", "0")
    _stub_denoise_deps(monkeypatch, n_steps=4)

    p = _make_base()
    p.config.num_inference_steps = 4
    p.config.guidance_scale = 5.0
    p.step_strategy = types.SimpleNamespace(
        reset=lambda: None, set_current_step=lambda i: None
    )

    seen = []

    def fake_dit(lat, t, ctx, sl, gs, **kwargs):
        seen.append(int(lat.shape[0]))
        return mx.zeros_like(lat)

    p.dit = fake_dit

    latents = mx.zeros((1, 16, 2, 4, 4))
    context = mx.zeros((1, 257, 4096))
    p._denoise_sample(latents, context, seq_lens=[8], grid_sizes=[(1, 2, 2)])

    assert seen == [2, 2, 2, 2]


def test_denoise_routes_full_keep(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DYNAMIC_CFG", "1")
    monkeypatch.setenv("FUSION_SKYREELS_CFG_KEEP_RATIO", "1.0")
    _stub_denoise_deps(monkeypatch, n_steps=3)

    p = _make_base()
    p.config.num_inference_steps = 3
    p.config.guidance_scale = 5.0
    p.step_strategy = types.SimpleNamespace(
        reset=lambda: None, set_current_step=lambda i: None
    )

    seen = []

    def fake_dit(lat, t, ctx, sl, gs, **kwargs):
        seen.append(int(lat.shape[0]))
        return mx.zeros_like(lat)

    p.dit = fake_dit

    latents = mx.zeros((1, 16, 2, 4, 4))
    context = mx.zeros((1, 257, 4096))
    p._denoise_sample(latents, context, seq_lens=[8], grid_sizes=[(1, 2, 2)])

    assert seen == [2, 2, 2]
