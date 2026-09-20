# SPDX-License-Identifier: Apache-2.0
"""#920: MuseTalk pipeline MLX allocator cache cap (tune_mlx_memory).

Default caps the allocator cache at 1 GiB — unbounded cache (13.55 GB observed)
causes multi-second allocator-scan render spikes in sustained loops. Env
overrides; 0 disables; no-op on non-Metal builds.
"""

import pytest

from fusion_mlx.video.musetalk_mlx import pipeline_mlx


@pytest.fixture
def metal_mock(monkeypatch):
    calls = {}

    class _Metal:
        def set_cache_limit(self, n):
            calls["cache"] = n

        def set_memory_limit(self, n):
            calls["memory"] = n

    import mlx.core as mx

    monkeypatch.setattr(mx, "metal", _Metal(), raising=False)
    return calls


def test_default_caps_cache_at_1gib(metal_mock, monkeypatch):
    monkeypatch.delenv("FUSION_MUSETALK_MLX_CACHE_LIMIT", raising=False)
    monkeypatch.delenv("FUSION_MUSETALK_MLX_MEMORY_LIMIT", raising=False)
    pipeline_mlx.tune_mlx_memory()
    assert metal_mock["cache"] == 1024**3
    assert "memory" not in metal_mock


def test_env_override_cache_limit(metal_mock, monkeypatch):
    monkeypatch.setenv("FUSION_MUSETALK_MLX_CACHE_LIMIT", "2048")
    pipeline_mlx.tune_mlx_memory()
    assert metal_mock["cache"] == 2048


def test_zero_disables_cache_cap(metal_mock, monkeypatch):
    monkeypatch.setenv("FUSION_MUSETALK_MLX_CACHE_LIMIT", "0")
    pipeline_mlx.tune_mlx_memory()
    assert "cache" not in metal_mock


def test_memory_limit_opt_in_only(metal_mock, monkeypatch):
    monkeypatch.setenv("FUSION_MUSETALK_MLX_MEMORY_LIMIT", str(3 * 1024**3))
    pipeline_mlx.tune_mlx_memory()
    assert metal_mock["memory"] == 3 * 1024**3


def test_no_metal_noop(monkeypatch):
    import mlx.core as mx

    monkeypatch.delattr(mx, "metal", raising=False)
    # must not raise
    pipeline_mlx.tune_mlx_memory()


def test_constructors_call_tune_mlx_memory(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline_mlx, "tune_mlx_memory", lambda: calls.append(1))
    monkeypatch.setattr(
        pipeline_mlx.Path, "read_text", lambda self: '{"dtype": "float16"}'
    )
    # from_pretrained_mlx exercises tune via constructor; stub weight loaders
    monkeypatch.setattr(
        pipeline_mlx, "load_native", lambda *a, **k: None, raising=False
    )
    try:
        pipeline_mlx.MuseTalkPipeline.from_pretrained_mlx("/tmp/fake-dist")
    except Exception:
        pass  # loader stubs may still fail on shape mismatch — cap call is what we assert
    assert calls


def test_smart_conv_default_on(monkeypatch):
    # #919/#921: SmartConv2d wraps VAE + UNet by default — real-weights A/B
    # (sd-vae-ft-mse + MuseTalk V15, b2): 80.0 -> 66.8 ms/round, image drift vs
    # fp32 at or below native-fp16 noise. Opt-out: FUSION_MUSETALK_SMART_CONV=0.
    monkeypatch.delenv("FUSION_MUSETALK_SMART_CONV", raising=False)
    monkeypatch.setattr(pipeline_mlx, "tune_mlx_memory", lambda: None)
    monkeypatch.setattr(
        pipeline_mlx.Path, "read_text", lambda self, **kw: '{"dtype": "float16"}'
    )
    from fusion_mlx.video.musetalk_mlx.utils import weights as weights_mod

    monkeypatch.setattr(weights_mod, "load_native", lambda *a, **k: None)
    called = []
    import fusion_mlx.graph_opt as graph_opt

    monkeypatch.setattr(graph_opt, "apply_smart_conv", lambda net: called.append(net))
    try:
        pipeline_mlx.MuseTalkPipeline.from_pretrained_mlx("/tmp/fake-dist")
    except Exception:
        pass
    assert called  # default ON — applied to vae + unet
    assert len(called) == 2


def test_smart_conv_opt_out(monkeypatch):
    monkeypatch.setenv("FUSION_MUSETALK_SMART_CONV", "0")
    monkeypatch.setattr(pipeline_mlx, "tune_mlx_memory", lambda: None)
    monkeypatch.setattr(
        pipeline_mlx.Path, "read_text", lambda self, **kw: '{"dtype": "float16"}'
    )
    from fusion_mlx.video.musetalk_mlx.utils import weights as weights_mod

    monkeypatch.setattr(weights_mod, "load_native", lambda *a, **k: None)
    called = []
    import fusion_mlx.graph_opt as graph_opt

    monkeypatch.setattr(graph_opt, "apply_smart_conv", lambda net: called.append(net))
    try:
        pipeline_mlx.MuseTalkPipeline.from_pretrained_mlx("/tmp/fake-dist")
    except Exception:
        pass
    assert not called  # opted out — native path only
