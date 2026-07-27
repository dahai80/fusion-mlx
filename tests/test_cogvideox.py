# SPDX-License-Identifier: Apache-2.0
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np


def _skip_if_no_mlx():
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("mlx not available")


# --- Config tests ---


class TestCogVideoXConfig:
    def test_2b_defaults(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig

        cfg = CogVideoXConfig.cogvideox_2b()
        assert cfg.num_attention_heads == 30
        assert cfg.num_layers == 30
        assert cfg.use_rotary_positional_embeddings is False
        assert cfg.vae_latent_channels == 16
        assert cfg.patch_size == 2

    def test_5b_defaults(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig

        cfg = CogVideoXConfig.cogvideox_5b()
        assert cfg.num_attention_heads == 48
        assert cfg.num_layers == 42
        assert cfg.use_rotary_positional_embeddings is True

    def test_5b_i2v(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig

        cfg = CogVideoXConfig.cogvideox_5b_i2v()
        assert cfg.num_attention_heads == 48
        assert cfg.num_layers == 42
        assert cfg.use_rotary_positional_embeddings is True

    def test_inner_dim(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig

        cfg = CogVideoXConfig.cogvideox_2b()
        assert cfg.inner_dim == 30 * 64  # 1920

    def test_ff_inner_dim(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig

        cfg = CogVideoXConfig.cogvideox_2b()
        assert cfg.ff_inner_dim == 1920 * 4

    def test_from_dict_roundtrip(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig

        cfg = CogVideoXConfig.cogvideox_5b()
        d = cfg.to_dict()
        cfg2 = CogVideoXConfig.from_dict(d)
        assert cfg2.num_attention_heads == 48
        assert cfg2.use_rotary_positional_embeddings is True
        assert cfg2.num_layers == 42


# --- Scheduler tests ---


class TestFlowMatchScheduler:
    def test_set_timesteps(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.scheduler import FlowMatchScheduler

        sched = FlowMatchScheduler(num_train_timesteps=1000, shift=1.0)
        sched.set_timesteps(10, shift=1.0)
        assert len(sched.timesteps) == 10

    def test_shift_sigmas(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.scheduler import FlowMatchScheduler

        sched = FlowMatchScheduler(num_train_timesteps=1000, shift=3.0)
        sigmas = np.linspace(1.0, 0.0, 11)
        shifted = sched._shift_sigmas(sigmas)
        assert len(shifted) == len(sigmas)
        assert shifted[0] == 1.0
        assert abs(shifted[-1]) < 1e-6

    def test_step_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx
        from fusion_mlx.video.cogvideox.scheduler import FlowMatchScheduler

        sched = FlowMatchScheduler(num_train_timesteps=1000, shift=1.0)
        sched.set_timesteps(5, shift=1.0)
        noise = mx.random.normal((1, 16, 7, 30, 40))
        t_val = sched.timesteps[0].item()
        result = sched.step(noise, t_val, noise)
        assert result.shape == noise.shape


# --- RoPE tests ---


class TestRoPE:
    def test_compute_3d_rope_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx
        from fusion_mlx.video.cogvideox.rope import compute_3d_rope

        cos, sin = compute_3d_rope(7, 30, 40, head_dim=64)
        # Should produce shapes for video tokens
        assert cos is not None
        assert sin is not None

    def test_apply_rope(self):
        _skip_if_no_mlx()
        import mlx.core as mx
        from fusion_mlx.video.cogvideox.rope import compute_3d_rope, apply_rope

        cos, sin = compute_3d_rope(7, 30, 40, patch_size=2, head_dim=64)
        seq_len = cos.shape[0]
        x = mx.random.normal((1, seq_len, 64))
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape


# --- Transformer tests ---


class TestCogVideoXTransformer:
    def test_instantiation_2b(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig
        from fusion_mlx.video.cogvideox.transformer import CogVideoXTransformer3DModel

        cfg = CogVideoXConfig.cogvideox_2b()
        model = CogVideoXTransformer3DModel(cfg)
        assert len(model.transformer_blocks) == 30

    def test_instantiation_5b(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig
        from fusion_mlx.video.cogvideox.transformer import CogVideoXTransformer3DModel

        cfg = CogVideoXConfig.cogvideox_5b()
        model = CogVideoXTransformer3DModel(cfg)
        assert len(model.transformer_blocks) == 42

    def test_forward_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig
        from fusion_mlx.video.cogvideox.transformer import CogVideoXTransformer3DModel

        cfg = CogVideoXConfig.cogvideox_2b()
        model = CogVideoXTransformer3DModel(cfg)
        latent = mx.random.normal((1, 7, 16, 64, 64))
        context = mx.random.normal((1, 226, cfg.text_embed_dim))
        timestep = mx.array([500.0])
        out = model(latent, encoder_hidden_states=context, timestep=timestep)
        assert out.shape == latent.shape

    def test_forward_shape_5b(self):
        _skip_if_no_mlx()
        import mlx.core as mx
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig
        from fusion_mlx.video.cogvideox.transformer import CogVideoXTransformer3DModel

        cfg = CogVideoXConfig.cogvideox_5b()
        model = CogVideoXTransformer3DModel(cfg)
        rope_cos, rope_sin = model._precompute_rope(7, 64, 64)
        latent = mx.random.normal((1, 7, 16, 64, 64))
        context = mx.random.normal((1, 226, cfg.text_embed_dim))
        timestep = mx.array([500.0])
        out = model(
            latent,
            encoder_hidden_states=context,
            timestep=timestep,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        assert out.shape == latent.shape


# --- VAE tests ---


class TestCogVideoXVAE:
    def test_vae_instantiation(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig
        from fusion_mlx.video.cogvideox.vae import AutoencoderKLCogVideoX

        cfg = CogVideoXConfig.cogvideox_2b()
        vae = AutoencoderKLCogVideoX(cfg)
        assert hasattr(vae, "decoder_conv_in")
        assert hasattr(vae, "encoder_conv_in")
        assert len(vae.encoder_blocks) > 0
        assert len(vae.decoder_blocks) > 0

    def test_causal_conv3d_instantiation(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.vae import CausalConv3d

        conv = CausalConv3d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0
        )
        assert conv.weight is not None


# --- Backend tests ---


class TestCogVideoBackend:
    def test_detect_positive(self):
        from fusion_mlx.engines.video_backends.cogvideox import CogVideoBackend

        assert CogVideoBackend.detect("THUDM/CogVideoX-2b")
        assert CogVideoBackend.detect("cogvideox-5b")
        assert CogVideoBackend.detect("cog_video-test")

    def test_detect_negative(self):
        from fusion_mlx.engines.video_backends.cogvideox import CogVideoBackend

        assert not CogVideoBackend.detect("wan2.1-14b")
        assert not CogVideoBackend.detect("ltx-video")

    def test_constraints(self):
        from fusion_mlx.engines.video_backends.cogvideox import CogVideoBackend

        backend = CogVideoBackend("THUDM/CogVideoX-2b")
        c = backend.constraints()
        assert c.supports_i2v is True
        assert c.dim_divisibility == 16
        assert c.max_n >= 1

    def test_registry_resolve(self):
        from fusion_mlx.engines.video_backends import resolve_backend, CogVideoBackend

        backend = resolve_backend("THUDM/CogVideoX-2b")
        assert isinstance(backend, CogVideoBackend)

    def test_registry_alias(self):
        from fusion_mlx.engines.video_backends import resolve_backend, CogVideoBackend

        backend = resolve_backend("test-model", explicit="cogvideox")
        assert isinstance(backend, CogVideoBackend)


# --- Convert tests ---


class TestConvert:
    def test_remap_transformer_key(self):
        from fusion_mlx.video.cogvideox.convert import _remap_transformer_key

        assert (
            _remap_transformer_key("transformer_blocks.0.attn1.to_q.weight")
            == "blocks.0.attn1.to_q.weight"
        )
        assert (
            _remap_transformer_key("transformer_blocks.5.norm1.linear.weight")
            == "blocks.5.norm1.linear.weight"
        )
        assert (
            _remap_transformer_key("time_embedding.linear_1.weight")
            == "time_embed.0.weight"
        )
        assert (
            _remap_transformer_key("patch_embed.proj.weight")
            == "patch_embed.proj.weight"
        )

    def test_remap_vae_key(self):
        from fusion_mlx.video.cogvideox.convert import _remap_vae_key

        result = _remap_vae_key("encoder.down_blocks.0.resnets.0.norm1.weight")
        assert result.startswith("encoder.blocks.0.")

    def test_convert_weights_identity(self):
        from fusion_mlx.video.cogvideox.convert import convert_weights

        sd = {"patch_embed.proj.weight": np.random.randn(3, 3, 3, 3).astype(np.float32)}
        result = convert_weights(sd, "transformer")
        assert "patch_embed.proj.weight" in result
        assert result["patch_embed.proj.weight"].shape == (3, 3, 3, 3)


# --- Text encoder tests ---


class TestTextEncoder:
    def test_import(self):
        from fusion_mlx.video.cogvideox import text_encoder

        assert hasattr(text_encoder, "load_t5_encoder")
        assert hasattr(text_encoder, "encode_text")


# --- Utils tests ---


class TestUtils:
    def test_config_auto_detect_2b(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig

        cfg = CogVideoXConfig.from_dict({"num_attention_heads": 30, "num_layers": 30})
        assert cfg.num_attention_heads == 30

    def test_config_auto_detect_5b(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.cogvideox.config import CogVideoXConfig

        cfg = CogVideoXConfig.from_dict({"num_attention_heads": 48, "num_layers": 42})
        assert cfg.num_attention_heads == 48


# --- Package import test ---


class TestPackageImport:
    def test_import_cogvideox(self):
        from fusion_mlx.video import cogvideox

        assert cogvideox is not None

    def test_submodules(self):
        from fusion_mlx.video.cogvideox import config
        from fusion_mlx.video.cogvideox import transformer
        from fusion_mlx.video.cogvideox import rope
        from fusion_mlx.video.cogvideox import scheduler
        from fusion_mlx.video.cogvideox import vae
        from fusion_mlx.video.cogvideox import text_encoder
        from fusion_mlx.video.cogvideox import utils
        from fusion_mlx.video.cogvideox import convert
        from fusion_mlx.video.cogvideox import generate
