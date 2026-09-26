# SPDX-License-Identifier: Apache-2.0
# Unit tests for ACE-Step MLX port (issue #988). Verifies shapes + finiteness
# of FSQ quantizer, DiT layer/model, condition encoder, tokenizer, pipeline,
# and Oobleck VAE against the real model config (no real weights needed for
# shape tests; VAE test uses real weights when FUSION_MLX_REAL_MODEL_TESTS=1).
from __future__ import annotations

import os

import mlx.core as mx
import pytest

from fusion_mlx.audio.acestep.blocks import (
    FSQ,
    DiTLayer,
    ResidualFSQ,
    create_4d_mask,
    rope_freqs,
)
from fusion_mlx.audio.acestep.condition import ConditionEncoder, pack_sequences
from fusion_mlx.audio.acestep.config import AceStepConfig
from fusion_mlx.audio.acestep.modeling import AceStepDiTModel
from fusion_mlx.audio.acestep.tokenizer import (
    AceStepAudioTokenizer,
    AudioTokenDetokenizer,
)


def _small_cfg() -> AceStepConfig:
    return AceStepConfig(
        num_hidden_layers=2,
        num_lyric_encoder_hidden_layers=2,
        num_timbre_encoder_hidden_layers=2,
        num_attention_pooler_hidden_layers=2,
    )


def test_fsq_shapes():
    fsq = FSQ([8, 8, 8, 5, 5, 5])
    x = mx.random.normal((2, 10, 6))
    out, idx = fsq(x)
    mx.eval(out, idx)
    assert out.shape == x.shape
    assert idx.shape == (2, 10)
    assert fsq.codebook_size == 8 * 8 * 8 * 5 * 5 * 5


def test_residual_fsq_projection():
    rf = ResidualFSQ(dim=2048, levels=[8, 8, 8, 5, 5, 5], num_quantizers=1)
    x = mx.random.normal((2, 10, 2048))
    out, idx = rf(x)
    mx.eval(out, idx)
    assert out.shape == x.shape
    assert idx.shape == (2, 10, 1)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_create_4d_mask():
    m = create_4d_mask(
        16, mx.float32, attention_mask=mx.ones((2, 16)), sliding_window=4
    )
    mx.eval(m)
    assert m.shape == (2, 1, 16, 16)


def test_rope_freqs():
    cos, sin = rope_freqs(128, 32, 1000000.0)
    mx.eval(cos, sin)
    assert cos.shape == (32, 128)
    assert sin.shape == (32, 128)


def test_dit_layer():
    cfg = _small_cfg()
    layer = DiTLayer(cfg, layer_idx=0, use_cross=True)
    B, L, D = 2, 16, cfg.hidden_size
    h = mx.random.normal((B, L, D))
    temb_proj = mx.random.normal((B, 6, D))  # (B, 6, D) timestep_proj
    cos, sin = rope_freqs(cfg.head_dim, L, cfg.rope_theta)
    mask = create_4d_mask(
        L, mx.float32, attention_mask=mx.ones((B, L)), sliding_window=cfg.sliding_window
    )
    enc_h = mx.random.normal((B, 8, cfg.hidden_size))
    out = layer(
        h,
        position_embeddings=(cos, sin),
        temb=temb_proj,
        self_mask=mask,
        encoder_hidden_states=enc_h,
    )
    mx.eval(out)
    assert out.shape == h.shape
    assert bool(mx.all(mx.isfinite(out)).item())


def test_dit_model():
    cfg = _small_cfg()
    m = AceStepDiTModel(cfg)
    B, T = 2, 32
    acoustic = cfg.audio_acoustic_hidden_dim
    h = mx.random.normal((B, T, acoustic))
    ctx = mx.random.normal((B, T, cfg.in_channels - acoustic))
    ts = mx.array([0.5, 0.7])
    ts_r = mx.array([0.1, 0.2])
    am = mx.ones((B, T))
    enc_h = mx.random.normal((B, 16, cfg.hidden_size))
    em = mx.ones((B, 16))
    out = m(h, ts, ts_r, am, enc_h, em, ctx)
    mx.eval(out)
    assert out.shape == (B, T, acoustic), out.shape


def test_dit_model_odd_seq_padding():
    cfg = _small_cfg()
    m = AceStepDiTModel(cfg)
    B, T = 1, 33
    acoustic = cfg.audio_acoustic_hidden_dim
    h = mx.random.normal((B, T, acoustic))
    ctx = mx.random.normal((B, T, cfg.in_channels - acoustic))
    out = m(
        h,
        mx.array([0.5]),
        mx.array([0.1]),
        mx.ones((B, T)),
        mx.zeros((B, 8, cfg.hidden_size)),
        mx.ones((B, 8)),
        ctx,
    )
    mx.eval(out)
    assert out.shape == (B, T, acoustic), out.shape


def test_condition_encoder():
    cfg = _small_cfg()
    ce = ConditionEncoder(cfg)
    B, Lt, Ll = 2, 16, 24
    th = mx.random.normal((B, Lt, cfg.text_hidden_dim))
    lh = mx.random.normal((B, Ll, cfg.text_hidden_dim))
    rap = mx.random.normal((3, cfg.timbre_hidden_dim))
    rom = mx.array([0, 0, 1])
    enc_h, enc_m = ce(th, mx.ones((B, Lt)), lh, mx.ones((B, Ll)), rap, rom)
    mx.eval(enc_h, enc_m)
    assert enc_h.shape[0] == B
    assert enc_h.shape[-1] == cfg.hidden_size


def test_pack_sequences():
    h1 = mx.random.normal((2, 4, 8))
    h2 = mx.random.normal((2, 3, 8))
    m1 = mx.array([[1, 1, 1, 0], [1, 1, 0, 0]])
    m2 = mx.ones((2, 3))
    out, mask = pack_sequences(h1, h2, m1, m2)
    mx.eval(out, mask)
    assert out.shape == (2, 7, 8)
    assert mask.shape == (2, 7)


def test_tokenizer_detokenizer():
    cfg = _small_cfg()
    tok = AceStepAudioTokenizer(cfg)
    detok = AudioTokenDetokenizer(cfg)
    B, T, P = 1, 16, cfg.pool_window_size
    acoustic = cfg.audio_acoustic_hidden_dim
    x = mx.random.normal((B, T * P, acoustic))
    q, idx = tok.tokenize(x)
    mx.eval(q, idx)
    assert q.shape == (B, T, cfg.fsq_dim)
    assert idx.shape == (B, T, cfg.fsq_input_num_quantizers)
    y = detok(q)
    mx.eval(y)
    assert y.shape == (B, T * P, acoustic)


@pytest.mark.real_model
def test_oobleck_vae_real():
    if not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"):
        pytest.skip("set FUSION_MLX_REAL_MODEL_TESTS=1 for real VAE weights")
    from fusion_mlx.audio.acestep.vae import AutoencoderOobleckMLX

    vae_dir = os.path.expanduser(
        "~/.fusion-mlx/models/models--ACE-Step--Ace-Step1.5/snapshots/"
        "19671f406d603126926c1b7e2adc169acbcade22/vae"
    )
    if not os.path.exists(vae_dir):
        pytest.skip("ACE-Step1.5 VAE weights not downloaded")
    vae = AutoencoderOobleckMLX.from_pretrained(vae_dir)
    wav = mx.random.normal((1, 2, 96000))
    mean, std = vae.encode(wav)
    mx.eval(mean, std)
    assert mean.shape[1] == 64
    wav_back = vae.decode(mean)
    mx.eval(wav_back)
    assert wav_back.shape[0] == 1
    assert wav_back.shape[1] == 2
    assert bool(mx.all(mx.isfinite(wav_back)).item())
