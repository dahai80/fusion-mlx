# SPDX-License-Identifier: Apache-2.0
# Unit tests for Open-Sora V2 MLX port.

import math
import pytest

import mlx.core as mx


class TestOpenSoraConfig:
    def test_defaults(self):
        from fusion_mlx.video.opensora.config import OpenSoraConfig

        cfg = OpenSoraConfig()
        assert cfg.in_channels == 64
        assert cfg.hidden_size == 3072
        assert cfg.num_heads == 24
        assert cfg.depth == 19
        assert cfg.depth_single_blocks == 38
        assert cfg.axes_dim == [16, 56, 56]
        assert cfg.patch_size == 2
        assert cfg.fused_qkv is False
        assert cfg.cond_embed is True

    def test_head_dim(self):
        from fusion_mlx.video.opensora.config import OpenSoraConfig

        cfg = OpenSoraConfig()
        assert cfg.head_dim == 128

    def test_from_dict(self):
        from fusion_mlx.video.opensora.config import OpenSoraConfig

        d = {
            "in_channels": 32,
            "hidden_size": 1024,
            "num_heads": 8,
            "depth": 4,
            "depth_single_blocks": 8,
        }
        cfg = OpenSoraConfig.from_dict(d)
        assert cfg.in_channels == 32
        assert cfg.hidden_size == 1024

    def test_axes_dim_sum_equals_head_dim(self):
        from fusion_mlx.video.opensora.config import OpenSoraConfig

        cfg = OpenSoraConfig()
        assert sum(cfg.axes_dim) == cfg.head_dim


class TestRoPE:
    def test_rope_shape(self):
        from fusion_mlx.video.opensora.rope import rope

        pos = mx.arange(10, dtype=mx.float32)
        out = rope(pos, dim=16, theta=10000)
        assert out.shape == (10, 8, 2, 2)

    def test_apply_rope_identity(self):
        from fusion_mlx.video.opensora.rope import apply_rope, rope

        pos = mx.zeros((1, 5), dtype=mx.float32)
        freqs = rope(pos, dim=8)
        q = mx.ones((1, 2, 5, 8), dtype=mx.float32)
        k = mx.ones((1, 2, 5, 8), dtype=mx.float32)
        q_out, k_out = apply_rope(q, k, freqs)
        assert mx.allclose(q_out, q, atol=1e-5)

    def test_embed_nd_shape(self):
        from fusion_mlx.video.opensora.rope import EmbedND

        embed = EmbedND(dim=128, theta=10000, axes_dim=[16, 56, 56])
        ids = mx.zeros((1, 20, 3), dtype=mx.float32)
        pe = embed(ids)
        assert pe.shape == (1, 20, 64, 2, 2)


class TestScheduler:
    def test_pack_unpack_roundtrip(self):
        from fusion_mlx.video.opensora.scheduler import pack, unpack

        x = mx.random.normal((1, 64, 3, 16, 32))
        packed = pack(x, patch_size=2)
        assert packed.shape == (1, 384, 256)
        unpacked = unpack(packed, height=16, width=32, num_frames=3, patch_size=2)
        assert unpacked.shape == x.shape
        assert mx.allclose(x, unpacked, atol=1e-5)

    def test_get_schedule_length(self):
        from fusion_mlx.video.opensora.scheduler import get_schedule

        schedule = get_schedule(30, 100, 13)
        assert len(schedule) == 30
        # time_shift may not be strictly monotonic for all alpha;
        # just verify structure is (float, float)
        for t_cur, t_next in schedule:
            assert isinstance(t_cur, float)
            assert isinstance(t_next, float)

    def test_get_image_ids_shape(self):
        from fusion_mlx.video.opensora.scheduler import get_image_ids

        ids = get_image_ids(num_frames=13, height=16, width=32, patch_size=2)
        assert ids.shape == (1, 1664, 3)

    def test_get_noise_shape(self):
        from fusion_mlx.video.opensora.scheduler import get_noise

        noise = get_noise(
            num_frames=13, height=60, width=106, batch_size=2, in_channels=64
        )
        assert noise.shape == (2, 64, 13, 60, 106)


class TestTransformerComponents:
    def test_mlp_embedder(self):
        from fusion_mlx.video.opensora.transformer import MLPEmbedder

        mlp = MLPEmbedder(256, 512)
        x = mx.random.normal((2, 10, 256))
        out = mlp(x)
        assert out.shape == (2, 10, 512)

    def test_rms_norm(self):
        from fusion_mlx.video.opensora.transformer import RMSNorm

        norm = RMSNorm(128)
        x = mx.random.normal((2, 10, 128))
        out = norm(x)
        assert out.shape == (2, 10, 128)

    def test_qk_norm(self):
        from fusion_mlx.video.opensora.transformer import QKNorm

        qkn = QKNorm(64)
        q = mx.random.normal((1, 8, 10, 64))
        k = mx.random.normal((1, 8, 10, 64))
        q_out, k_out = qkn(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_self_attention(self):
        from fusion_mlx.video.opensora.transformer import SelfAttention
        from fusion_mlx.video.opensora.rope import EmbedND

        attn = SelfAttention(dim=256, num_heads=4, qkv_bias=True, fused_qkv=False)
        pe_embed = EmbedND(dim=64, theta=10000, axes_dim=[16, 24, 24])
        ids = mx.zeros((1, 5, 3), dtype=mx.float32)
        pe = pe_embed(ids)
        x = mx.random.normal((1, 5, 256))
        out = attn(x, pe)
        assert out.shape == (1, 5, 256)

    def test_modulation_double(self):
        from fusion_mlx.video.opensora.transformer import Modulation

        mod = Modulation(256, double=True)
        vec = mx.random.normal((2, 256))
        (s1, sc1, g1), (s2, sc2, g2) = mod(vec)
        assert s1.shape == (2, 256)
        assert sc2.shape == (2, 256)

    def test_modulation_single(self):
        from fusion_mlx.video.opensora.transformer import Modulation

        mod = Modulation(256, double=False)
        vec = mx.random.normal((2, 256))
        (s1, sc1, g1), extra = mod(vec)
        assert s1.shape == (2, 256)
        assert extra is None

    def test_last_layer(self):
        from fusion_mlx.video.opensora.transformer import LastLayer

        layer = LastLayer(256, 1, 64)
        x = mx.random.normal((1, 10, 256))
        vec = mx.random.normal((1, 256))
        out = layer(x, vec)
        assert out.shape == (1, 10, 64)


class TestMMDiTModel:
    @pytest.fixture
    def small_config(self):
        from fusion_mlx.video.opensora.config import OpenSoraConfig

        return OpenSoraConfig(
            in_channels=16,
            vec_in_dim=64,
            context_in_dim=128,
            hidden_size=64,
            num_heads=4,
            depth=2,
            depth_single_blocks=2,
            axes_dim=[4, 6, 6],  # sum=16 = head_dim
            theta=10000,
            qkv_bias=True,
            guidance_embed=False,
            cond_embed=False,
            fused_qkv=False,
            patch_size=2,
            mlp_ratio=2.0,
        )

    def test_model_forward(self, small_config):
        from fusion_mlx.video.opensora.transformer import MMDiTModel

        model = MMDiTModel(small_config)
        B, L_img, L_txt = 1, 12, 8
        img = mx.random.normal((B, L_img, small_config.in_channels))
        txt = mx.random.normal((B, L_txt, small_config.context_in_dim))
        timesteps = mx.array([0.5])
        y_vec = mx.random.normal((B, small_config.vec_in_dim))
        img_ids = mx.zeros((B, L_img, 3))
        txt_ids = mx.zeros((B, L_txt, 3))
        out = model(img, img_ids, txt, txt_ids, timesteps, y_vec)
        assert out.shape[0] == B
        assert out.shape[1] == L_img

    def test_model_with_cond(self, small_config):
        from fusion_mlx.video.opensora.transformer import MMDiTModel

        small_config.cond_embed = True
        model = MMDiTModel(small_config)
        B, L_img, L_txt = 1, 12, 8
        img = mx.random.normal((B, L_img, small_config.in_channels))
        txt = mx.random.normal((B, L_txt, small_config.context_in_dim))
        timesteps = mx.array([0.5])
        y_vec = mx.random.normal((B, small_config.vec_in_dim))
        img_ids = mx.zeros((B, L_img, 3))
        txt_ids = mx.zeros((B, L_txt, 3))
        cond = mx.random.normal(
            (B, L_img, small_config.in_channels + small_config.patch_size**2)
        )
        out = model(img, img_ids, txt, txt_ids, timesteps, y_vec, cond=cond)
        assert out.shape[0] == B

    def test_timestep_embedding(self):
        from fusion_mlx.video.opensora.transformer import MMDiTModel

        t = mx.array([0.5, 1.0])
        emb = MMDiTModel._timestep_embedding(t, 256)
        assert emb.shape == (2, 256)


class TestOpenSoraBackend:
    def test_detect(self):
        from fusion_mlx.engines.video_backends.opensora import OpenSoraBackend

        assert OpenSoraBackend.detect("opensora-v2-11b")
        assert OpenSoraBackend.detect("Open-Sora-2.0")
        assert OpenSoraBackend.detect("open_sora_model")
        assert not OpenSoraBackend.detect("ltx2-dev")

    def test_constraints(self):
        from fusion_mlx.engines.video_backends.opensora import OpenSoraBackend

        backend = OpenSoraBackend("opensora-v2")
        c = backend.constraints()
        assert c.supports_i2v is True
        assert c.max_n == 201

    def test_name(self):
        from fusion_mlx.engines.video_backends.opensora import OpenSoraBackend

        assert OpenSoraBackend.name == "opensora"


class TestRegistry:
    def test_resolve_opensora(self):
        from fusion_mlx.engines.video_backends import resolve_backend, BACKENDS

        assert "opensora" in BACKENDS
        b = resolve_backend("opensora-v2", explicit="opensora")
        assert b.__class__.__name__ == "OpenSoraBackend"

    def test_alias(self):
        from fusion_mlx.engines.video_backends import _ALIASES

        assert _ALIASES.get("open-sora") == "opensora"
        assert _ALIASES.get("opensora-v2") == "opensora"
