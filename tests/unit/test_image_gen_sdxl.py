import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.engines.image_gen import VARIANT_MAP, _infer_variant
from fusion_mlx.image.sdxl.config import (
    SDXLConfig,
    SDXLTextEncoderConfig,
    SDXLUNetConfig,
    SDXLVAEConfig,
)
from fusion_mlx.image.sdxl.generate import SDXLPipeline
from fusion_mlx.image.sdxl.text_encoder import SDXLCLIPTextModel
from fusion_mlx.image.sdxl.weights import _map_key, remap_unet_weights


class TestSDXLVariantMap:
    def test_sdxl_in_variant_map(self):
        assert "sdxl" in VARIANT_MAP
        module_path, cls_name, config_label, default_guidance = VARIANT_MAP["sdxl"]
        assert module_path == "fusion_mlx.image.sdxl.generate"
        assert cls_name == "SDXLPipeline"
        assert config_label == "sdxl_base"
        assert default_guidance == 7.5

    def test_cosxl_in_variant_map(self):
        module_path, cls_name, _, _ = VARIANT_MAP["cosxl"]
        assert module_path == "fusion_mlx.image.sdxl.generate"
        assert cls_name == "SDXLPipeline"

    def test_sdxs_in_variant_map(self):
        module_path, cls_name, _, default_guidance = VARIANT_MAP["sdxs"]
        assert module_path == "fusion_mlx.image.sdxl.generate"
        assert cls_name == "SDXLPipeline"
        assert default_guidance == 4.0

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("sdxl-base", "sdxl"),
            ("stable-diffusion-xl-base-1.0", "sdxl"),
            ("SDXL-Base", "sdxl"),
            ("stabilityai/stable-diffusion-xl-base-1.0", "sdxl"),
            ("cosxl_edit", "cosxl"),
            ("CosXL-Edit", "cosxl"),
            ("sdxs-512", "sdxs"),
            ("SDXS-512-Drawing", "sdxs"),
        ],
    )
    def test_infer_sdxl_family(self, path, expected):
        assert _infer_variant(path) == expected

    def test_infer_sdxl_before_flux_fallback(self):
        # sdxl paths must not fall through to txt2img/flux1 defaults.
        assert _infer_variant("foo-sdxl-bar") == "sdxl"


class TestSDXLConfig:
    def test_unet_config_values(self):
        cfg = SDXLUNetConfig()
        assert cfg.block_out_channels == (320, 640, 1280)
        assert cfg.attention_head_dim == (5, 10, 20)
        assert cfg.cross_attention_dim == 2048
        assert cfg.transformer_layers_per_block == (1, 2, 10)
        assert cfg.projection_class_embeddings_input_dim == 2816
        assert cfg.addition_time_embed_dim == 256
        assert cfg.in_channels == 4
        assert cfg.out_channels == 4
        assert cfg.sample_size == 128

    def test_vae_config_values(self):
        cfg = SDXLVAEConfig()
        assert cfg.block_out_channels == (128, 256, 512, 512)
        assert cfg.latent_channels == 4
        assert cfg.scaling_factor == pytest.approx(0.13025)
        assert cfg.in_channels == 3

    def test_text_encoder_config_values(self):
        cfg = SDXLTextEncoderConfig()
        # CLIP-L
        assert cfg.clip_l_dims == 768
        assert cfg.clip_l_layers == 12
        assert cfg.clip_l_heads == 12
        assert cfg.clip_l_intermediate == 3072
        assert cfg.clip_l_act == "quick_gelu"
        # OpenCLIP-G
        assert cfg.clip_g_dims == 1280
        assert cfg.clip_g_layers == 32
        assert cfg.clip_g_heads == 20
        assert cfg.clip_g_intermediate == 5120
        assert cfg.clip_g_act == "gelu"
        assert cfg.clip_g_projection_dim == 1280

    def test_top_config_aggregates(self):
        cfg = SDXLConfig()
        assert isinstance(cfg.unet, SDXLUNetConfig)
        assert isinstance(cfg.vae, SDXLVAEConfig)
        assert isinstance(cfg.text, SDXLTextEncoderConfig)
        assert cfg.paths.repo == "stabilityai/stable-diffusion-xl-base-1.0"


class TestSDXLWeightRemap:
    @pytest.mark.parametrize(
        "src,expected",
        [
            (
                "time_embedding.linear_1.weight",
                "time_embedding.0.weight",
            ),
            (
                "time_embedding.linear_2.weight",
                "time_embedding.1.weight",
            ),
            (
                "add_embedding.linear_1.weight",
                "add_embedding.0.weight",
            ),
            (
                "add_embedding.linear_2.weight",
                "add_embedding.1.weight",
            ),
            (
                "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
                "down_blocks.1.attentions.0.transformer_blocks.0.ff.net_0_proj.weight",
            ),
            (
                "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.weight",
                "down_blocks.1.attentions.0.transformer_blocks.0.ff.net_2.weight",
            ),
        ],
    )
    def test_unet_key_remap(self, src, expected):
        assert _map_key(src) == expected

    def test_passthrough_key(self):
        assert (
            _map_key("down_blocks.1.resnets.0.norm1.weight")
            == "down_blocks.1.resnets.0.norm1.weight"
        )

    def test_conv_weights_transposed(self):
        # A 4D conv weight must be transposed OIHW -> OHWI by remap_unet_weights.
        raw = {
            "conv_in.weight": mx.zeros((4, 4, 3, 3)),
            "conv_out.weight": mx.zeros((4, 4, 3, 3)),
        }
        pairs = remap_unet_weights(raw)
        out = dict(pairs)
        assert out["conv_in.weight"].shape == (4, 3, 3, 4)
        assert out["conv_out.weight"].shape == (4, 3, 3, 4)

    def test_linear_weights_not_transposed(self):
        raw = {
            "time_embedding.linear_1.weight": mx.zeros((320, 320)),
        }
        pairs = remap_unet_weights(raw)
        out = dict(pairs)
        assert out["time_embedding.0.weight"].shape == (320, 320)


class TestSDXLCLIPTextModel:
    def test_clip_l_param_names(self):
        model = SDXLCLIPTextModel(
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
        # CLIP-L has no text_projection.
        assert "text_projection.weight" not in names

    def test_clip_g_has_text_projection(self):
        model = SDXLCLIPTextModel(
            dims=1280,
            num_layers=32,
            num_heads=20,
            intermediate=5120,
            act="gelu",
            vocab=49408,
            max_pos=77,
            projection_dim=1280,
        )
        names = {k for k, _ in nn.utils.tree_flatten(model.parameters())}
        assert "text_projection.weight" in names
        assert "text_model.encoder.layers.31.self_attn.q_proj.weight" in names

    def test_clip_l_returns_hidden_and_pooled(self):
        model = SDXLCLIPTextModel(
            dims=64,
            num_layers=1,
            num_heads=4,
            intermediate=128,
            act="gelu",
            vocab=49408,
            max_pos=8,
        )
        tokens = mx.zeros((2, 8), dtype=mx.int32)
        hidden, pooled = model(tokens)
        mx.eval(hidden, pooled)
        assert hidden.shape == (2, 8, 64)
        assert pooled.shape == (2, 64)

    def test_clip_g_pooled_after_projection(self):
        model = SDXLCLIPTextModel(
            dims=64,
            num_layers=1,
            num_heads=4,
            intermediate=128,
            act="gelu",
            vocab=49408,
            max_pos=8,
            projection_dim=32,
        )
        tokens = mx.zeros((1, 8), dtype=mx.int32)
        hidden, pooled = model(tokens)
        mx.eval(hidden, pooled)
        assert hidden.shape == (1, 8, 64)
        assert pooled.shape == (1, 32)


class TestSDXLPipelineConstruction:
    def test_construct_without_loading(self):
        pipe = SDXLPipeline(model_config=None, model_path="sdxl-base", quantize=None)
        assert pipe.config.block_out_channels == (320, 640, 1280)
        assert pipe.unet is None
        assert pipe.vae is None
        assert pipe._loaded is False

    def test_construct_has_dual_text_encoder_config(self):
        pipe = SDXLPipeline(model_config=None, model_path="sdxl-base", quantize=None)
        assert pipe.text_cfg.clip_l_dims == 768
        assert pipe.text_cfg.clip_g_dims == 1280
