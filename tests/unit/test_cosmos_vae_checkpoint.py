# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Cosmos VAE decode checkpointing (#441 Metal freeze)."""

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


def _tiny_vae():
    # CosmosVideoVAE hardcodes base_ch=128; patch to a tiny channel
    # count so a decode forward is fast enough for a unit test.
    import mlx.nn as nn

    from fusion_mlx.video.cosmos.vae import (
        CosmosVAEConv3d,
        CosmosVAEDownBlock,
        CosmosVAEResBlock,
        CosmosVAEUpBlock,
    )

    orig_init = CosmosVideoVAE.__init__
    orig_up = CosmosVideoVAE.UP_BLOCKS

    def tiny_init(self, latent_channels=4, in_channels=3):
        nn.Module.__init__(self)
        self.latent_channels = latent_channels
        self.in_channels = in_channels
        ch_mult = [1, 2]
        base_ch = 32  # must be divisible by GroupNorm num_groups (32)
        self.enc_conv_in = CosmosVAEConv3d(in_channels, base_ch, 3, 1, 1)
        prev_ch = base_ch
        enc_blocks = []
        for i, mult in enumerate(ch_mult):
            cur_ch = base_ch * mult
            down = i < len(ch_mult) - 1
            enc_blocks.append(CosmosVAEDownBlock(prev_ch, cur_ch, downsample=down))
            prev_ch = cur_ch
        self.enc_blocks = enc_blocks
        self.enc_mid1 = CosmosVAEResBlock(prev_ch)
        self.enc_mid2 = CosmosVAEResBlock(prev_ch)
        self.enc_conv_out = CosmosVAEConv3d(prev_ch, latent_channels * 2, 3, 1, 1)
        self.dec_conv_in = CosmosVAEConv3d(latent_channels, prev_ch, 3, 1, 1)
        self.dec_mid1 = CosmosVAEResBlock(prev_ch)
        self.dec_mid2 = CosmosVAEResBlock(prev_ch)
        dec_blocks = []
        for i, mult in reversed(list(enumerate(ch_mult))):
            cur_ch = base_ch * mult
            up = i > 0
            dec_blocks.append(CosmosVAEUpBlock(prev_ch, cur_ch, upsample=up))
            prev_ch = cur_ch
        self.dec_blocks = dec_blocks
        self.dec_conv_out = CosmosVAEConv3d(prev_ch, in_channels, 3, 1, 1)

    CosmosVideoVAE.__init__ = tiny_init
    # UP_BLOCKS is a class attr; patch it (2-block decoder: [False, True])
    # so _compute_output_shape reports the right temporal/spatial upscale.
    CosmosVideoVAE.UP_BLOCKS = [False, True]
    try:
        return CosmosVideoVAE(latent_channels=4, in_channels=3)
    finally:
        CosmosVideoVAE.__init__ = orig_init
        CosmosVideoVAE.UP_BLOCKS = orig_up


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
    # 1 up-block: spatial 4x4 -> 8x8 (2x), temporal 2 -> 4 (2x)
    assert out.shape == (1, 3, 4, 8, 8)


def test_decode_default_follows_env(monkeypatch):
    monkeypatch.setenv("FUSION_COSMOS_VAE_CHECKPOINT", "1")
    vae = _tiny_vae()
    z = mx.random.normal((1, 4, 2, 4, 4))
    out = vae.decode(z)
    mx.eval(out)
    assert out.shape[0] == 1 and out.shape[1] == 3
