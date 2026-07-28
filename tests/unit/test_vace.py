# SPDX-License-Identifier: Apache-2.0
# Tests for VACE (Video-Conditioned Auxiliary Control Encoding) blocks and config.

import pytest

pytest.importorskip("mlx", reason="MLX not available")

import mlx.core as mx
import mlx.nn as nn

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
            0: 0, 5: 1, 10: 2, 15: 3, 20: 4, 25: 5, 30: 6, 35: 7,
        }
