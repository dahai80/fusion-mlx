# SPDX-License-Identifier: Apache-2.0
# Unit tests for video adapters: IP-Adapter, ControlNet, AnimateDiff, registry.
# Callers: pytest; no real MLX GPU required (mock-MLX conftest covers CI).
# Schema: VideoAdapter ABC, @register_adapter, create_adapter factory.
# User instruction: "如果task已经完成，请更新进展，如果还没有完成，继续推进落地task"

from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeBlock:
    """Minimal DiT block mock with block_{i} attrs for inject/remove."""

    pass


class _FakeDiT:
    """Minimal DiT mock exposing block_{i} and _num_blocks."""

    def __init__(self, num_blocks: int = 3):
        self._num_blocks = num_blocks
        for i in range(num_blocks):
            setattr(self, f"block_{i}", _FakeBlock())


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    def test_register_and_create_ip_adapter(self):
        from fusion_mlx.video.adapters import VIDEO_ADAPTERS, create_adapter

        assert "ip_adapter" in VIDEO_ADAPTERS
        adapter = create_adapter("ip_adapter", scale=0.5)
        assert adapter is not None
        assert adapter.name == "ip_adapter"
        assert adapter.scale == 0.5

    def test_register_and_create_controlnet(self):
        from fusion_mlx.video.adapters import VIDEO_ADAPTERS, create_adapter

        assert "controlnet" in VIDEO_ADAPTERS
        adapter = create_adapter("controlnet", scale=1.0)
        assert adapter is not None
        assert adapter.name == "controlnet"

    def test_register_and_create_animatediff(self):
        from fusion_mlx.video.adapters import VIDEO_ADAPTERS, create_adapter

        assert "animatediff" in VIDEO_ADAPTERS
        adapter = create_adapter("animatediff", scale=1.0)
        assert adapter is not None
        assert adapter.name == "animatediff"

    def test_create_unknown_returns_none(self):
        from fusion_mlx.video.adapters import create_adapter

        adapter = create_adapter("nonexistent_adapter")
        assert adapter is None


# ---------------------------------------------------------------------------
# IP-Adapter tests
# ---------------------------------------------------------------------------


class TestIPAdapter:
    def test_modify_context_no_image_returns_unchanged(self):
        from fusion_mlx.video.adapters.ip_adapter import IPAdapter

        adapter = IPAdapter(scale=1.0)
        adapter._loaded = True
        adapter.clip_vision = MagicMock()
        adapter.projection = MagicMock()
        import mlx.core as mx

        ctx = mx.zeros((1, 10, 4096))
        out = adapter.modify_context(ctx)
        assert out is ctx

    def test_modify_context_with_none_image_kwarg(self):
        from fusion_mlx.video.adapters.ip_adapter import IPAdapter

        adapter = IPAdapter(scale=1.0)
        adapter._loaded = True
        import mlx.core as mx

        ctx = mx.zeros((1, 10, 4096))
        out = adapter.modify_context(ctx, image=None)
        assert out is ctx

    def test_default_text_dim_is_4096(self):
        from fusion_mlx.video.adapters.ip_adapter import IPAdapter

        adapter = IPAdapter()
        assert adapter.text_dim == 4096

    def test_scale_defaults_to_1(self):
        from fusion_mlx.video.adapters.ip_adapter import IPAdapter

        adapter = IPAdapter()
        assert adapter.scale == 1.0

    def test_unload_clears_models(self):
        from fusion_mlx.video.adapters.ip_adapter import IPAdapter

        adapter = IPAdapter()
        adapter.clip_vision = MagicMock()
        adapter.projection = MagicMock()
        adapter._loaded = True
        adapter.unload()
        assert adapter.clip_vision is None
        assert adapter.projection is None
        assert adapter._loaded is False

    def test_remap_clip_weights_filters_visual_only(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.ip_adapter import remap_clip_weights

        # ndim=2 so it skips the transpose branch (patch_embedding accepts 4d or 2d)
        weights = {
            "text_model.encoder.0.weight": mx.zeros((10, 10)),
            "visual.conv1.weight": mx.zeros((10,)),  # ndim=1, skips transpose
        }
        result = remap_clip_weights(weights)
        assert "patch_embedding.weight" in result
        assert not any(k.startswith("text_model") for k in result)

    def test_remap_ip_adapter_weights_strips_prefix(self):
        from fusion_mlx.video.adapters.ip_adapter import remap_ip_adapter_weights

        weights = {
            "image_proj_model.norm1.weight": MagicMock(),
            "image_proj_model.linear1.weight": MagicMock(),
        }
        result = remap_ip_adapter_weights(weights)
        assert "norm1.weight" in result
        assert "linear1.weight" in result

    def test_remap_ip_adapter_kv_proj_splits(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.ip_adapter import remap_ip_adapter_weights

        fake_weight = mx.zeros((200, 1280))
        weights = {"image_proj_model.kv_proj.weight": fake_weight}
        result = remap_ip_adapter_weights(weights)
        assert "linear1.weight" in result
        assert "linear2.weight" in result


# ---------------------------------------------------------------------------
# ControlNet tests
# ---------------------------------------------------------------------------


class TestControlNet:
    def test_default_dim_and_type(self):
        from fusion_mlx.video.adapters.controlnet import ControlNet

        adapter = ControlNet()
        assert adapter.dim == 5120
        assert adapter.control_type == "canny"
        assert adapter.text_dim == 4096

    def test_unload_clears_state(self):
        from fusion_mlx.video.adapters.controlnet import ControlNet

        adapter = ControlNet()
        adapter.dit = MagicMock()
        adapter._residuals = [MagicMock()]
        adapter._loaded = True
        adapter.unload()
        assert adapter.dit is None
        assert adapter._residuals is None
        assert adapter._loaded is False

    def test_modify_denoise_step_no_image_returns_latents(self):
        from fusion_mlx.video.adapters.controlnet import ControlNet

        adapter = ControlNet()
        adapter._loaded = True
        import mlx.core as mx

        latents = mx.zeros((1, 16, 64, 64))
        out = adapter.modify_denoise_step(
            None, latents, mx.zeros((1,)), mx.zeros((1, 10, 4096))
        )
        assert out is latents

    def test_get_residuals_returns_cached(self):
        from fusion_mlx.video.adapters.controlnet import ControlNet

        adapter = ControlNet()
        fake_residuals = [MagicMock(), MagicMock()]
        adapter._residuals = fake_residuals
        assert adapter.get_residuals() is fake_residuals

    def test_remap_controlnet_weights_hint_block(self):
        from fusion_mlx.video.adapters.controlnet import remap_controlnet_weights

        weights = {
            "controlnet.input_hint_block.0.weight": MagicMock(),
            "controlnet.input_hint_block.2.weight": MagicMock(),
        }
        result = remap_controlnet_weights(weights)
        assert "hint_block.conv1.weight" in result
        assert "hint_block.conv2.weight" in result

    def test_remap_controlnet_weights_zero_convs(self):
        from fusion_mlx.video.adapters.controlnet import remap_controlnet_weights

        weights = {"controlnet.zero_convs.3.weight": MagicMock()}
        result = remap_controlnet_weights(weights)
        assert "block_3.zero_conv.weight" in result

    def test_remap_controlnet_weights_middle_block(self):
        from fusion_mlx.video.adapters.controlnet import remap_controlnet_weights

        weights = {"controlnet.middle_block.weight": MagicMock()}
        result = remap_controlnet_weights(weights)
        assert "mid_zero_conv.weight" in result


# ---------------------------------------------------------------------------
# AnimateDiff tests
# ---------------------------------------------------------------------------


class TestAnimateDiff:
    def test_default_config(self):
        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff()
        assert adapter.dim == 5120
        assert adapter.num_heads == 8
        assert adapter.num_layers == 40
        assert adapter.scale == 1.0

    def test_inject_sets_motion_module_attr(self):
        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff(config={"dim": 64, "num_heads": 4, "num_layers": 3})
        adapter.load()
        dit = _FakeDiT(num_blocks=3)
        adapter.inject(dit)
        for i in range(3):
            block = getattr(dit, f"block_{i}")
            assert hasattr(block, "motion_module"), f"block_{i} missing motion_module"
            assert hasattr(
                block, "animatediff_scale"
            ), f"block_{i} missing animatediff_scale"
            assert block.animatediff_scale == 1.0

    def test_remove_clears_motion_module_attr(self):
        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff(config={"dim": 64, "num_heads": 4, "num_layers": 3})
        adapter.load()
        dit = _FakeDiT(num_blocks=3)
        adapter.inject(dit)
        adapter.remove(dit)
        for i in range(3):
            block = getattr(dit, f"block_{i}")
            assert not hasattr(
                block, "motion_module"
            ), f"block_{i} still has motion_module"

    def test_modify_denoise_step_returns_latents_unchanged(self):
        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff()
        import mlx.core as mx

        latents = mx.zeros((1, 16, 64, 64))
        out = adapter.modify_denoise_step(
            None, latents, mx.zeros((1,)), mx.zeros((1, 10, 4096))
        )
        assert out is latents

    def test_unload_clears_modules(self):
        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff(config={"dim": 64, "num_heads": 4, "num_layers": 3})
        adapter.load()
        assert len(adapter._modules) == 3
        adapter.unload()
        assert adapter._modules == []
        assert adapter._loaded is False
        assert adapter._injected is False

    def test_zero_init_output_proj(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.animatediff import MotionModule

        mod = MotionModule(dim=64, num_heads=4)
        weight_max = mx.abs(mod.output_proj.weight).max()
        bias_max = mx.abs(mod.output_proj.bias).max()
        assert (
            float(weight_max) == 0.0
        ), f"output_proj.weight not zero-init: max={weight_max}"
        assert float(bias_max) == 0.0, f"output_proj.bias not zero-init: max={bias_max}"

    def test_inject_fewer_modules_than_blocks(self):
        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff(config={"dim": 64, "num_heads": 4, "num_layers": 2})
        adapter.load()
        dit = _FakeDiT(num_blocks=5)
        adapter.inject(dit)
        assert hasattr(dit.block_0, "motion_module")
        assert hasattr(dit.block_1, "motion_module")
        assert not hasattr(dit.block_2, "motion_module")
        adapter.remove(dit)

    def test_remap_animatediff_weights(self):
        from fusion_mlx.video.adapters.animatediff import remap_animatediff_weights

        weights = {
            "motion_modules.0.temporal_attn.to_q.weight": MagicMock(),
            "motion_modules.0.temporal_attn.to_k.weight": MagicMock(),
            "motion_modules.1.norm.weight": MagicMock(),
        }
        result = remap_animatediff_weights(weights, num_layers=3)
        assert "block_0.q_proj.weight" in result
        assert "block_0.k_proj.weight" in result
        assert "block_1.norm.weight" in result

    def test_remap_animatediff_weights_skips_over_num_layers(self):
        from fusion_mlx.video.adapters.animatediff import remap_animatediff_weights

        weights = {
            "motion_modules.5.norm.weight": MagicMock(),
        }
        result = remap_animatediff_weights(weights, num_layers=3)
        assert len(result) == 0
