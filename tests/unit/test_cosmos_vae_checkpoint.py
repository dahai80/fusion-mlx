# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Cosmos VAE (#441 Metal checkpoint, #461 factored-conv)."""

import pytest

pytest.importorskip("mlx")  # suite needs mlx runtime; skip if absent

import mlx.core as mx

from fusion_mlx.video.cosmos.vae import CosmosVideoVAE, _vae_checkpoint_enabled


def test_checkpoint_enabled_default(monkeypatch):
    monkeypatch.delenv("FUSION_COSMOS_VAE_CHECKPOINT", raising=False)
    assert _vae_checkpoint_enabled() is True


def test_checkpoint_disabled_explicit(monkeypatch):
    monkeypatch.setenv("FUSION_COSMOS_VAE_CHECKPOINT", "0")
    assert _vae_checkpoint_enabled() is False


def test_checkpoint_disabled_empty(monkeypatch):
    monkeypatch.setenv("FUSION_COSMOS_VAE_CHECKPOINT", "")
    assert _vae_checkpoint_enabled() is False


def test_checkpoint_disabled_false_str(monkeypatch):
    monkeypatch.setenv("FUSION_COSMOS_VAE_CHECKPOINT", "false")
    assert _vae_checkpoint_enabled() is False


def test_checkpoint_enabled_explicit(monkeypatch):
    monkeypatch.setenv("FUSION_COSMOS_VAE_CHECKPOINT", "1")
    assert _vae_checkpoint_enabled() is True


# Tiny config so a decode forward is fast enough for a unit test.
# patch_size=2, one hybrid up-block: latent (1,4,2,4,4) -> (1,3,3,8,8).
_TINY_CONFIG = {
    "in_channels": 3,
    "latent_channels": 4,
    "decode_block_out_channels": [8, 8],
    "encoder_block_out_channels": [8, 8],
    "num_layers": 1,
    "patch_size": 2,
    "patch_type": "haar",
    "resolution": 16,
    "spatial_compression_ratio": 4,
    "temporal_compression_ratio": 4,
    "attention_resolutions": [],
}


def _tiny_vae():
    return CosmosVideoVAE(latent_channels=4, in_channels=3, config=_TINY_CONFIG)


def test_decode_checkpoint_matches_non_checkpoint(monkeypatch):
    # checkpoint=True must produce the same output as checkpoint=False
    # (it only changes WHEN intermediates are eval'd/freed, not the math)
    monkeypatch.setenv("FUSION_COSMOS_VAE_CHECKPOINT", "0")
    vae = _tiny_vae()
    z = mx.random.normal((1, 4, 2, 4, 4))
    out_plain = vae.decode(z, checkpoint=False)
    mx.eval(out_plain)
    out_ckpt = vae.decode(z, checkpoint=True)
    mx.eval(out_ckpt)
    assert out_plain.shape == out_ckpt.shape
    assert mx.allclose(out_plain, out_ckpt, atol=1e-5).item() is True


def test_decode_checkpoint_kwarg_overrides_env(monkeypatch):
    # env says disabled, but explicit checkpoint=True must checkpoint
    monkeypatch.setenv("FUSION_COSMOS_VAE_CHECKPOINT", "0")
    vae = _tiny_vae()
    z = mx.random.normal((1, 4, 2, 4, 4))
    out = vae.decode(z, checkpoint=True)
    mx.eval(out)
    # 1 up-block: spatial 4x4 -> 8x8 (2x), temporal 2 -> 3
    assert out.shape == (1, 3, 3, 8, 8)


def test_decode_default_follows_env(monkeypatch):
    monkeypatch.setenv("FUSION_COSMOS_VAE_CHECKPOINT", "1")
    vae = _tiny_vae()
    z = mx.random.normal((1, 4, 2, 4, 4))
    out = vae.decode(z)
    mx.eval(out)
    assert out.shape[0] == 1 and out.shape[1] == 3


# --- Cosmos flow scheduler sigma-shift regression (#460) ---
# sigma_max must stay in [0,1] (normalized flow space). The Cosmos
# time-shift s' = shift*s/(1+(shift-1)*s) saturates for s>>1; a raw
# sigma_max=80 collapses every timestep to ~1.49 so the sample never
# denoises (all-black output). These guard against that regression.


def test_flow_scheduler_sigma_max_is_normalized():
    from fusion_mlx.video.cosmos.scheduler import CosmosFlowScheduler

    sch = CosmosFlowScheduler()
    assert sch.sigma_max <= 1.0, "sigma_max must be in normalized [0,1] flow space"


def test_flow_scheduler_timesteps_are_monotonic_and_spread():
    from fusion_mlx.video.cosmos.scheduler import CosmosFlowScheduler

    sch = CosmosFlowScheduler()
    sch.set_timesteps(6)
    ts = [float(t) for t in sch.timesteps.tolist()]
    assert all(ts[i] > ts[i + 1] for i in range(len(ts) - 1)), ts
    assert ts[0] - ts[-1] > 0.5, f"timesteps collapsed: {ts}"
    assert ts[0] < 1.01, f"first timestep too high: {ts[0]}"


def test_flow_scheduler_old_sigma_max_collapsed():
    from fusion_mlx.video.cosmos.scheduler import CosmosFlowScheduler

    sch = CosmosFlowScheduler(sigma_max=80.0)
    sch.set_timesteps(6)
    ts = [float(t) for t in sch.timesteps.tolist()]
    assert ts[0] - ts[-1] < 0.1, f"expected collapse, got spread: {ts}"
