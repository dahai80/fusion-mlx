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
    def test_default_config(self):
        from fusion_mlx.video.adapters.controlnet import ControlNet

        adapter = ControlNet()
        assert adapter.control_type == "canny"
        assert adapter.stride == 4
        assert adapter.model_variant == "wan2.1-t2v-14b"

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

    def test_remap_wan_controlnet_condition_embedder(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.controlnet import remap_wan_controlnet_weights

        weights = {
            "condition_embedder.time_embedder.linear_1.weight": mx.zeros((1536, 256)),
            "condition_embedder.time_embedder.linear_2.weight": mx.zeros((1536, 1536)),
            "condition_embedder.text_embedder.linear_1.weight": mx.zeros((1536, 4096)),
            "condition_embedder.time_proj.weight": mx.zeros((9216, 1536)),
        }
        result = remap_wan_controlnet_weights(weights)
        assert "time_embedding.layers.0.weight" in result
        assert "time_embedding.layers.2.weight" in result
        assert "text_embedding.layers.0.weight" in result
        assert "time_projection.layers.1.weight" in result

    def test_remap_wan_controlnet_control_encoder(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.controlnet import remap_wan_controlnet_weights

        weights = {
            "control_encoder.0.0.weight": mx.zeros((192, 3, 3, 9, 9)),
            "control_encoder.0.0.bias": mx.zeros((192,)),
            "control_encoder.0.2.weight": mx.zeros((192,)),
        }
        result = remap_wan_controlnet_weights(weights)
        assert "control_encoder.stage1.conv.weight" in result
        assert result["control_encoder.stage1.conv.weight"].shape == (192, 9, 9, 3)
        assert "control_encoder.stage1.conv.bias" in result
        assert "control_encoder.stage1.gn.weight" in result

    def test_remap_wan_controlnet_blocks(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.controlnet import remap_wan_controlnet_weights

        weights = {
            "blocks.0.attn1.to_q.weight": mx.zeros((1536, 1536)),
            "blocks.0.attn2.to_k.weight": mx.zeros((1536, 4096)),
            "blocks.0.ffn.net.0.proj.weight": mx.zeros((8960, 1536)),
            "blocks.0.scale_shift_table": mx.zeros((1, 6, 1536)),
        }
        result = remap_wan_controlnet_weights(weights)
        assert "block_0.block.self_attn.q.weight" in result
        assert "block_0.block.cross_attn.k.weight" in result
        assert "block_0.block.ffn.fc1.weight" in result
        assert "block_0.block.modulation" in result

    def test_remap_wan_controlnet_controlnet_blocks(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.controlnet import remap_wan_controlnet_weights

        weights = {
            "controlnet_blocks.0.weight": mx.zeros((5120, 1536)),
            "controlnet_blocks.5.bias": mx.zeros((5120,)),
        }
        result = remap_wan_controlnet_weights(weights)
        assert "controlnet_block_0.weight" in result
        assert "controlnet_block_5.bias" in result

    def test_remap_wan_controlnet_patch_embedding_conv3d(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.controlnet import remap_wan_controlnet_weights

        weights = {
            "patch_embedding.weight": mx.zeros((1536, 64, 1, 2, 2)),
            "patch_embedding.bias": mx.zeros((1536,)),
        }
        result = remap_wan_controlnet_weights(weights)
        assert "patch_embedding.weight" in result
        assert result["patch_embedding.weight"].shape == (1536, 2, 2, 64)
        assert "patch_embedding.bias" in result

    def test_wan_controlnet_forward_with_random_weights(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.controlnet import WanControlnet

        model = WanControlnet()
        B = 1
        hidden_states = mx.random.normal((B, 16, 8, 8))
        t = mx.array([0.5])
        context = mx.random.normal((B, 5, 4096))
        control_states = mx.random.normal((B, 3, 64, 64))
        seq_lens = [8 * 8 // (2 * 2)]
        grid_sizes = [(1, 4, 4)]
        residuals = model.forward(
            hidden_states, t, context, control_states,
            seq_lens=seq_lens, grid_sizes=grid_sizes,
        )
        assert len(residuals) == 6
        for r in residuals:
            assert r.shape[-1] == 5120


# ---------------------------------------------------------------------------
# AnimateDiff tests
# ---------------------------------------------------------------------------


class TestAnimateDiff:
    def test_default_config(self):
        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff()
        assert adapter.num_layers == 40
        assert adapter.scale == 1.0
        assert adapter.variant == "high_noise"

    def test_modify_denoise_step_returns_latents_unchanged(self):
        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff()
        import mlx.core as mx

        latents = mx.zeros((1, 16, 64, 64))
        out = adapter.modify_denoise_step(
            None, latents, mx.zeros((1,)), mx.zeros((1, 10, 4096))
        )
        assert out is latents

    def test_unload_clears_state(self):
        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff()
        adapter._lora_map = {"test": {"lora_A": MagicMock(), "lora_B": MagicMock()}}
        adapter._original_weights = {"test": MagicMock()}
        adapter._loaded = True
        adapter._injected = True
        adapter.unload()
        assert adapter._lora_map == {}
        assert adapter._original_weights == {}
        assert adapter._loaded is False
        assert adapter._injected is False

    def test_remap_animatediff_lora_weights_self_attn(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.animatediff import remap_animatediff_lora_weights

        weights = {
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": mx.zeros((32, 5120)),
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": mx.zeros((5120, 32)),
            "diffusion_model.blocks.0.self_attn.k.lora_A.weight": mx.zeros((32, 5120)),
            "diffusion_model.blocks.0.self_attn.k.lora_B.weight": mx.zeros((5120, 32)),
        }
        result = remap_animatediff_lora_weights(weights, num_layers=40)
        assert "blocks.0.self_attn.q.weight" in result
        assert "blocks.0.self_attn.k.weight" in result
        assert "lora_A" in result["blocks.0.self_attn.q.weight"]
        assert "lora_B" in result["blocks.0.self_attn.q.weight"]

    def test_remap_animatediff_lora_weights_ffn(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.animatediff import remap_animatediff_lora_weights

        weights = {
            "diffusion_model.blocks.5.ffn.0.lora_A.weight": mx.zeros((32, 5120)),
            "diffusion_model.blocks.5.ffn.0.lora_B.weight": mx.zeros((13824, 32)),
            "diffusion_model.blocks.5.ffn.2.lora_A.weight": mx.zeros((32, 13824)),
            "diffusion_model.blocks.5.ffn.2.lora_B.weight": mx.zeros((5120, 32)),
        }
        result = remap_animatediff_lora_weights(weights, num_layers=40)
        assert "blocks.5.ffn.fc1.weight" in result
        assert "blocks.5.ffn.fc2.weight" in result

    def test_remap_animatediff_lora_weights_skips_over_num_layers(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.animatediff import remap_animatediff_lora_weights

        weights = {
            "diffusion_model.blocks.50.self_attn.q.lora_A.weight": mx.zeros((32, 5120)),
        }
        result = remap_animatediff_lora_weights(weights, num_layers=40)
        assert len(result) == 0

    def test_remap_animatediff_lora_weights_cross_attn(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.animatediff import remap_animatediff_lora_weights

        weights = {
            "diffusion_model.blocks.3.cross_attn.q.lora_A.weight": mx.zeros((32, 5120)),
            "diffusion_model.blocks.3.cross_attn.q.lora_B.weight": mx.zeros((5120, 32)),
            "diffusion_model.blocks.3.cross_attn.v.lora_A.weight": mx.zeros((32, 5120)),
            "diffusion_model.blocks.3.cross_attn.v.lora_B.weight": mx.zeros((5120, 32)),
        }
        result = remap_animatediff_lora_weights(weights, num_layers=40)
        assert "blocks.3.cross_attn.q.weight" in result
        assert "blocks.3.cross_attn.v.weight" in result

    def test_inject_merges_lora_into_dit(self):
        import mlx.core as mx
        import mlx.nn as nn

        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff(scale=1.0, config={"num_layers": 2})

        class _TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Module()
                self.self_attn.q = nn.Linear(64, 64)

        class _TinyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = [_TinyBlock(), _TinyBlock()]

        dit = _TinyDiT()
        orig_weight = mx.array(dit.blocks[0].self_attn.q.weight)

        lora_a = mx.random.normal((8, 64)) * 0.01
        lora_b = mx.random.normal((64, 8)) * 0.01
        adapter._lora_map = {
            "blocks.0.self_attn.q.weight": {"lora_A": lora_a, "lora_B": lora_b}
        }
        adapter._loaded = True
        adapter.inject(dit)

        expected = orig_weight + lora_b.astype(orig_weight.dtype) @ lora_a.astype(orig_weight.dtype)
        actual = dit.blocks[0].self_attn.q.weight
        diff = mx.abs(actual - expected).max()
        assert float(diff) < 1e-4, f"LoRA merge mismatch: diff={diff}"

        adapter.remove(dit)
        restored_diff = mx.abs(dit.blocks[0].self_attn.q.weight - orig_weight).max()
        assert float(restored_diff) < 1e-4, f"LoRA restore mismatch: diff={restored_diff}"

    def test_inject_no_lora_is_noop(self):
        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff()
        adapter._lora_map = {}
        adapter._loaded = True
        adapter.inject("fake_dit")
        assert adapter._injected is True

    def test_variant_config(self):
        from fusion_mlx.video.adapters.animatediff import AnimateDiff

        adapter = AnimateDiff(config={"animatediff_variant": "low_noise"})
        assert adapter.variant == "low_noise"


# ---------------------------------------------------------------------------
# E2E tests: real DiT architecture + adapters (random weights, no GPU)
# ---------------------------------------------------------------------------


class TestAdapterE2E:
    def test_controlnet_forward_produces_valid_residuals(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.controlnet import WanControlnet

        model = WanControlnet()
        B = 1
        # Small spatial dims (8x8 latent, 64x64 control image)
        hidden_states = mx.random.normal((B, 16, 8, 8))
        t = mx.array([0.5])
        context = mx.random.normal((B, 5, 4096))
        control_states = mx.random.normal((B, 3, 64, 64))
        seq_lens = [16]
        grid_sizes = [(1, 4, 4)]
        residuals = model.forward(
            hidden_states, t, context, control_states,
            seq_lens=seq_lens, grid_sizes=grid_sizes,
        )
        assert len(residuals) == 6, f"Expected 6 residuals, got {len(residuals)}"
        for i, r in enumerate(residuals):
            assert r.shape[0] == B, f"residual[{i}] batch={r.shape[0]}"
            assert r.shape[-1] == 5120, f"residual[{i}] dim={r.shape[-1]}"

    def test_animatediff_lora_inject_remove_roundtrip(self):
        import mlx.core as mx
        import mlx.nn as nn

        from fusion_mlx.video.adapters.animatediff import AnimateDiff, remap_animatediff_lora_weights

        adapter = AnimateDiff(scale=1.0, config={"num_layers": 2})

        class _TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Module()
                self.self_attn.q = nn.Linear(64, 64)
                self.cross_attn = nn.Module()
                self.cross_attn.q = nn.Linear(64, 64)
                self.ffn = nn.Module()
                self.ffn.fc1 = nn.Linear(64, 128)
                self.ffn.fc2 = nn.Linear(128, 64)

        class _TinyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = [_TinyBlock(), _TinyBlock()]

        dit = _TinyDiT()
        orig_flat = {k: mx.array(v) for k, v in nn.utils.tree_flatten(dit.parameters())}

        lora_a = mx.random.normal((8, 64)) * 0.01
        lora_b = mx.random.normal((64, 8)) * 0.01
        lora_a2 = mx.random.normal((8, 64)) * 0.01
        lora_b2 = mx.random.normal((128, 8)) * 0.01

        adapter._lora_map = {
            "blocks.0.self_attn.q.weight": {"lora_A": lora_a, "lora_B": lora_b},
            "blocks.0.ffn.fc1.weight": {"lora_A": lora_a2, "lora_B": lora_b2},
        }
        adapter._loaded = True

        adapter.inject(dit)
        merged_flat = {k: v for k, v in nn.utils.tree_flatten(dit.parameters())}

        for key in ["blocks.0.self_attn.q.weight", "blocks.0.ffn.fc1.weight"]:
            diff = float(mx.abs(merged_flat[key] - orig_flat[key]).max())
            assert diff > 1e-6, f"{key} should differ after inject (diff={diff})"

        for key in ["blocks.0.cross_attn.q.weight", "blocks.0.ffn.fc2.weight", "blocks.1.self_attn.q.weight"]:
            diff = float(mx.abs(merged_flat[key] - orig_flat[key]).max())
            assert diff < 1e-6, f"{key} should be unchanged (diff={diff})"

        adapter.remove(dit)
        restored_flat = {k: v for k, v in nn.utils.tree_flatten(dit.parameters())}
        for key in orig_flat:
            diff = float(mx.abs(restored_flat[key] - orig_flat[key]).max())
            assert diff < 1e-6, f"{key} not restored (diff={diff})"

    def test_remap_real_lora_weights_shape(self):
        import mlx.core as mx

        from fusion_mlx.video.adapters.animatediff import remap_animatediff_lora_weights

        weights = {
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": mx.zeros((32, 5120)),
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": mx.zeros((5120, 32)),
            "diffusion_model.blocks.39.ffn.2.lora_A.weight": mx.zeros((32, 13824)),
            "diffusion_model.blocks.39.ffn.2.lora_B.weight": mx.zeros((5120, 32)),
        }
        result = remap_animatediff_lora_weights(weights, num_layers=40)
        assert "blocks.0.self_attn.q.weight" in result
        assert "blocks.39.ffn.fc2.weight" in result
        assert result["blocks.0.self_attn.q.weight"]["lora_A"].shape == (32, 5120)
        assert result["blocks.0.self_attn.q.weight"]["lora_B"].shape == (5120, 32)
        assert result["blocks.39.ffn.fc2.weight"]["lora_A"].shape == (32, 13824)
        assert result["blocks.39.ffn.fc2.weight"]["lora_B"].shape == (5120, 32)
