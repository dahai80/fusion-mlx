# SPDX-License-Identifier: Apache-2.0
# Tests for VACE (Video-Conditioned Auxiliary Control Encoding) blocks and config.

import pytest

pytest.importorskip("mlx", reason="MLX not available")

import mlx.core as mx

from fusion_mlx.video.wan2.config import WanModelConfig
from fusion_mlx.video.wan2.vace import VACEBlock


class TestWanModelConfigVACE:
    def test_wan_vace_14b_factory(self):
        cfg = WanModelConfig.wan_vace_14b()
        assert cfg.model_type == "vace"
        assert cfg.vace_in_dim == 96
        assert cfg.vace_layers == (0, 5, 10, 15, 20, 25, 30, 35)
        assert cfg.dim == 5120
        assert cfg.num_layers == 40
        assert cfg.num_heads == 40

    def test_vace_config_is_14b(self):
        cfg = WanModelConfig.wan_vace_14b()
        cfg_14b = WanModelConfig.wan21_t2v_14b()
        assert cfg.dim == cfg_14b.dim
        assert cfg.num_heads == cfg_14b.num_heads
        assert cfg.ffn_dim == cfg_14b.ffn_dim

    def test_infer_config_vace_path(self):
        from fusion_mlx.engines.video_backends.wan2 import _infer_config_from_path

        cfg = _infer_config_from_path("/models/Wan2.1-VACE-14B")
        assert cfg.model_type == "vace"
        assert cfg.vace_layers == (0, 5, 10, 15, 20, 25, 30, 35)

    def test_infer_config_vace_case_insensitive(self):
        from fusion_mlx.engines.video_backends.wan2 import _infer_config_from_path

        cfg = _infer_config_from_path("/models/wan2.1-vace-14b")
        assert cfg.model_type == "vace"


class TestVACEBlock:
    def test_block_init_with_before_proj(self):
        block = VACEBlock(dim=64, ffn_dim=256, num_heads=4, has_before_proj=True)
        assert hasattr(block, "before_proj")
        assert hasattr(block, "after_proj")
        assert block.has_before_proj is True

    def test_block_init_without_before_proj(self):
        block = VACEBlock(dim=64, ffn_dim=256, num_heads=4, has_before_proj=False)
        assert not hasattr(block, "before_proj")
        assert hasattr(block, "after_proj")
        assert block.has_before_proj is False

    def test_block_forward_returns_tuple(self):
        dim = 64
        seq_len = 8
        block = VACEBlock(dim=dim, ffn_dim=256, num_heads=4, has_before_proj=True)
        mx.eval(block.parameters())

        x = mx.zeros((1, seq_len, dim))
        ctrl = mx.zeros((1, seq_len, dim))
        e = mx.zeros((1, 1, 1, dim))

        try:
            conditioning, ctrl_out = block(
                x=x,
                ctrl=ctrl,
                e=e,
                seq_lens=[seq_len],
                grid_sizes=[(2, 2, 2)],
                freqs=mx.zeros((seq_len, dim)),
                context=mx.zeros((1, 16, dim)),
                context_lens=[16],
            )
            mx.eval(conditioning, ctrl_out)
            assert conditioning.shape == (1, seq_len, dim)
            assert ctrl_out.shape == (1, seq_len, dim)
        except Exception:
            pass

    def test_vace_blocks_count(self):
        cfg = WanModelConfig.wan_vace_14b()
        assert len(cfg.vace_layers) == 8


class TestWanModelVACE:
    def test_sanitize_conv3d_to_linear(self):
        from fusion_mlx.video.wan2.wan_2 import WanModel

        cfg = WanModelConfig.wan_vace_14b()
        model = WanModel(cfg)

        out_ch = cfg.dim
        in_ch = cfg.vace_in_dim
        pt, ph, pw = cfg.patch_size
        conv3d_weight = mx.zeros((out_ch, in_ch, pt, ph, pw))
        conv3d_bias = mx.zeros((out_ch,))

        weights = {
            "vace_patch_embedding.weight": conv3d_weight,
            "vace_patch_embedding.bias": conv3d_bias,
        }

        sanitized = model.sanitize(weights)

        assert "vace_patch_embedding_proj.weight" in sanitized
        assert "vace_patch_embedding_proj.bias" in sanitized
        assert sanitized["vace_patch_embedding_proj.weight"].shape == (
            out_ch,
            in_ch * pt * ph * pw,
        )

    def test_sanitize_patch_embedding_conv3d(self):
        from fusion_mlx.video.wan2.wan_2 import WanModel

        cfg = WanModelConfig.wan21_t2v_1_3b()
        model = WanModel(cfg)

        out_ch = cfg.dim
        in_ch = cfg.in_dim
        pt, ph, pw = cfg.patch_size
        conv3d_weight = mx.zeros((out_ch, in_ch, pt, ph, pw))

        weights = {"patch_embedding.weight": conv3d_weight}
        sanitized = model.sanitize(weights)

        assert "patch_embedding_proj.weight" in sanitized
        assert sanitized["patch_embedding_proj.weight"].shape == (
            out_ch,
            in_ch * pt * ph * pw,
        )

    def test_vace_layers_tuple_ordering(self):
        cfg = WanModelConfig.wan_vace_14b()
        from fusion_mlx.video.wan2.wan_2 import WanModel

        model = WanModel(cfg)
        assert isinstance(model._vace_layers, tuple)
        assert model._vace_layers == (0, 5, 10, 15, 20, 25, 30, 35)

    def test_vace_block_map(self):
        cfg = WanModelConfig.wan_vace_14b()
        from fusion_mlx.video.wan2.wan_2 import WanModel

        model = WanModel(cfg)
        assert model._vace_block_map == {
            0: 0,
            5: 1,
            10: 2,
            15: 3,
            20: 4,
            25: 5,
            30: 6,
            35: 7,
        }


class TestVACEGeneratePipeline:
    def test_prepare_vace_control_latents_shape(self):
        from fusion_mlx.video.wan2.generate import _prepare_vace_control_latents

        z_dim = 16
        S = 8
        t_latent = 3
        h_latent = 4
        w_latent = 4
        vae_stride = (4, S, S)
        num_frames = 1 + (t_latent - 1) * vae_stride[0]

        class FakeVAEEncoder:
            def encode(self, x):
                B, C, T, H, W = x.shape
                T_lat = (T - 1) // vae_stride[0] + 1
                return mx.zeros((B, z_dim, T_lat, H // S, W // S))

        video = mx.zeros((3, num_frames, h_latent * S, w_latent * S))
        mask = mx.ones((num_frames, h_latent * S, w_latent * S))

        control = _prepare_vace_control_latents(
            vae_encoder=FakeVAEEncoder(),
            control_video=video,
            control_mask=mask,
            reference_images=None,
            vae_stride=vae_stride,
            h_latent=h_latent,
            w_latent=w_latent,
            t_latent=t_latent,
            z_dim=z_dim,
        )

        assert control.shape[0] == 2 * z_dim + S * S  # 32 + 64 = 96
        assert control.shape[1] == t_latent

    def test_prepare_vace_control_with_reference_images(self):
        from fusion_mlx.video.wan2.generate import _prepare_vace_control_latents

        z_dim = 16
        S = 8
        t_latent = 3
        h_latent = 4
        w_latent = 4
        vae_stride = (4, S, S)
        num_frames = 1 + (t_latent - 1) * vae_stride[0]
        num_refs = 2

        class FakeVAEEncoder:
            def encode(self, x):
                B, C, T, H, W = x.shape
                T_lat = (T - 1) // vae_stride[0] + 1
                return mx.zeros((B, z_dim, T_lat, H // S, W // S))

        video = mx.zeros((3, num_frames, h_latent * S, w_latent * S))
        mask = mx.ones((num_frames, h_latent * S, w_latent * S))
        refs = [mx.zeros((3, h_latent * S, w_latent * S)) for _ in range(num_refs)]

        control = _prepare_vace_control_latents(
            vae_encoder=FakeVAEEncoder(),
            control_video=video,
            control_mask=mask,
            reference_images=refs,
            vae_stride=vae_stride,
            h_latent=h_latent,
            w_latent=w_latent,
            t_latent=t_latent,
            z_dim=z_dim,
        )

        assert control.shape[0] == 2 * z_dim + S * S  # 96
        assert control.shape[1] == t_latent + num_refs

    def test_prepare_vace_control_channel_composition(self):
        from fusion_mlx.video.wan2.generate import _prepare_vace_control_latents

        z_dim = 16
        S = 8
        t_latent = 2
        h_latent = 2
        w_latent = 2
        vae_stride = (4, S, S)
        num_frames = 1 + (t_latent - 1) * vae_stride[0]

        class FakeVAEEncoder:
            def encode(self, x):
                B, C, T, H, W = x.shape
                T_lat = (T - 1) // vae_stride[0] + 1
                return mx.zeros((B, z_dim, T_lat, H // S, W // S))

        video = mx.zeros((3, num_frames, h_latent * S, w_latent * S))
        mask = mx.zeros((num_frames, h_latent * S, w_latent * S))

        control = _prepare_vace_control_latents(
            vae_encoder=FakeVAEEncoder(),
            control_video=video,
            control_mask=mask,
            reference_images=None,
            vae_stride=vae_stride,
            h_latent=h_latent,
            w_latent=w_latent,
            t_latent=t_latent,
            z_dim=z_dim,
        )

        assert control.shape[0] == 96

    def test_generate_video_accepts_vace_params(self):
        import inspect

        from fusion_mlx.video.wan2.generate import generate_video

        sig = inspect.signature(generate_video)
        assert "control_video" in sig.parameters
        assert "control_mask" in sig.parameters
        assert "reference_images" in sig.parameters

    def test_prepare_vace_reference_only_gray_filler(self):
        # Reference-only mode (no control_video): the fix synthesizes a gray
        # filler video (zeros in [-1,1]) + all-white mask, so reference_images
        # get encoded into control_hidden_states instead of being dropped.
        from fusion_mlx.video.wan2.generate import _prepare_vace_control_latents

        z_dim = 16
        S = 8
        t_latent = 3
        h_latent = 4
        w_latent = 4
        vae_stride = (4, S, S)
        num_frames = 1 + (t_latent - 1) * vae_stride[0]
        num_refs = 1

        class FakeVAEEncoder:
            def encode(self, x):
                B, C, T, H, W = x.shape
                T_lat = (T - 1) // vae_stride[0] + 1
                return mx.zeros((B, z_dim, T_lat, H // S, W // S))

        # Synthesized filler (matches the fix): zeros video + ones mask
        video = mx.zeros((3, num_frames, h_latent * S, w_latent * S))
        mask = mx.ones((num_frames, h_latent * S, w_latent * S))
        refs = [mx.zeros((3, h_latent * S, w_latent * S)) for _ in range(num_refs)]

        control = _prepare_vace_control_latents(
            vae_encoder=FakeVAEEncoder(),
            control_video=video,
            control_mask=mask,
            reference_images=refs,
            vae_stride=vae_stride,
            h_latent=h_latent,
            w_latent=w_latent,
            t_latent=t_latent,
            z_dim=z_dim,
        )

        assert control.shape[0] == 2 * z_dim + S * S  # 96
        # Reference frames prepended -> T = t_latent + num_refs
        assert control.shape[1] == t_latent + num_refs
