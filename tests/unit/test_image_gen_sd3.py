import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.engines.image_gen import VARIANT_MAP, _infer_variant
from fusion_mlx.image.sd3.config import (
    ClipGConfig,
    ClipLConfig,
    SD3Config,
    get_config,
)
from fusion_mlx.image.sd3.generate import SD3Pipeline, _map_t5
from fusion_mlx.image.sd3.text_encoder import CLIPTextModel


class TestSD3VariantMap:
    def test_sd3_in_variant_map(self):
        assert "sd3" in VARIANT_MAP
        module_path, cls_name, config_label, default_guidance = VARIANT_MAP["sd3"]
        assert module_path == "fusion_mlx.image.sd3.generate"
        assert cls_name == "SD3Pipeline"
        assert config_label == "sd3_medium"
        assert default_guidance == 4.0

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("sd3-medium", "sd3"),
            ("stable-diffusion-3-medium", "sd3"),
            ("SD3-Medium", "sd3"),
            ("argmaxinc/mlx-stable-diffusion-3-medium", "sd3"),
        ],
    )
    def test_infer_sd3(self, path, expected):
        assert _infer_variant(path) == expected


class TestSD3Config:
    def test_default_config_values(self):
        cfg = SD3Config()
        assert cfg.inner_dim == 1536
        assert cfg.num_layers == 24
        assert cfg.num_attention_heads == 24
        assert cfg.attention_head_dim == 64
        assert cfg.joint_attention_dim == 4096
        assert cfg.pooled_projection_dim == 2048
        assert cfg.in_channels == 16
        assert cfg.out_channels == 16
        assert cfg.patch_size == 2
        assert cfg.pos_embed_max_size == 192

    def test_get_config_sd3(self):
        cfg = get_config("sd3")
        assert cfg.inner_dim == 1536

    def test_get_config_sd3_medium(self):
        cfg = get_config("sd3-medium")
        assert cfg.inner_dim == 1536

    def test_get_config_unknown_falls_back(self):
        cfg = get_config("does-not-exist")
        assert cfg.inner_dim == 1536

    def test_clip_l_config(self):
        cl = ClipLConfig()
        assert cl.dims == 768
        assert cl.num_layers == 12
        assert cl.num_heads == 12
        assert cl.act == "quick_gelu"

    def test_clip_g_config_20_heads(self):
        cg = ClipGConfig()
        assert cg.dims == 1280
        assert cg.num_layers == 32
        assert cg.num_heads == 20
        assert cg.act == "gelu"


class TestT5WeightRemap:
    def test_shared_weight(self):
        assert _map_t5("shared.weight") == "shared.weight"

    def test_final_layer_norm(self):
        assert _map_t5("encoder.final_layer_norm.weight") == "final_layer_norm.weight"

    @pytest.mark.parametrize(
        "src,expected",
        [
            (
                "encoder.block.0.layer.0.layer_norm.weight",
                "t5_blocks.0.attention.layer_norm.weight",
            ),
            (
                "encoder.block.5.layer.0.SelfAttention.q.weight",
                "t5_blocks.5.attention.SelfAttention.q.weight",
            ),
            (
                "encoder.block.5.layer.0.SelfAttention.k.weight",
                "t5_blocks.5.attention.SelfAttention.k.weight",
            ),
            (
                "encoder.block.5.layer.0.SelfAttention.v.weight",
                "t5_blocks.5.attention.SelfAttention.v.weight",
            ),
            (
                "encoder.block.5.layer.0.SelfAttention.o.weight",
                "t5_blocks.5.attention.SelfAttention.o.weight",
            ),
            (
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                "t5_blocks.0.attention.SelfAttention.relative_attention_bias.weight",
            ),
        ],
    )
    def test_attn_remap(self, src, expected):
        assert _map_t5(src) == expected

    @pytest.mark.parametrize(
        "src,expected",
        [
            (
                "encoder.block.3.layer.1.layer_norm.weight",
                "t5_blocks.3.ff.layer_norm.weight",
            ),
            (
                "encoder.block.3.layer.1.DenseReluDense.wi_0.weight",
                "t5_blocks.3.ff.DenseReluDense.wi_0.weight",
            ),
            (
                "encoder.block.3.layer.1.DenseReluDense.wi_1.weight",
                "t5_blocks.3.ff.DenseReluDense.wi_1.weight",
            ),
            (
                "encoder.block.3.layer.1.DenseReluDense.wo.weight",
                "t5_blocks.3.ff.DenseReluDense.wo.weight",
            ),
        ],
    )
    def test_ff_remap(self, src, expected):
        assert _map_t5(src) == expected

    def test_unknown_key_returns_none(self):
        assert _map_t5("encoder.embed_tokens.weight") is None


class TestCLIPTextModel:
    def test_clip_l_param_names(self):
        model = CLIPTextModel(
            dims=768,
            num_layers=12,
            num_heads=12,
            intermediate=3072,
            act="quick_gelu",
            vocab=49408,
            max_pos=77,
        )
        names = {k for k, _ in nn.utils.tree_flatten(model.parameters())}
        assert "text_model.embeddings.token_embedding.weight" in names
        assert "text_model.embeddings.position_embedding.weight" in names
        assert "text_model.encoder.layers.0.self_attn.q_proj.weight" in names
        assert "text_model.encoder.layers.0.layer_norm1.weight" in names
        assert "text_model.encoder.layers.0.mlp.fc1.weight" in names
        assert "text_model.final_layer_norm.weight" in names

    def test_clip_g_param_count_matches_hf(self):
        model = CLIPTextModel(
            dims=1280,
            num_layers=32,
            num_heads=20,
            intermediate=5120,
            act="gelu",
            vocab=49408,
            max_pos=77,
        )
        names = {k for k, _ in nn.utils.tree_flatten(model.parameters())}
        assert "text_model.encoder.layers.31.self_attn.q_proj.weight" in names
        assert "text_model.encoder.layers.31.mlp.fc2.weight" in names

    def test_clip_pooled_output_shape(self):
        model = CLIPTextModel(
            dims=64,
            num_layers=1,
            num_heads=4,
            intermediate=128,
            act="gelu",
            vocab=49408,
            max_pos=8,
        )
        tokens = mx.zeros((2, 8), dtype=mx.int32)
        pooled = model(tokens)
        mx.eval(pooled)
        assert pooled.shape == (2, 64)


class TestSD3PipelineConstruction:
    def test_construct_without_loading(self):
        pipe = SD3Pipeline(model_config=None, model_path="sd3-medium", quantize=None)
        assert pipe.config.inner_dim == 1536
        assert pipe.transformer is None
        assert pipe.vae is None
        assert pipe._loaded is False
