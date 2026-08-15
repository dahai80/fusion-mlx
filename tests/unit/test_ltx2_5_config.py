# SPDX-License-Identifier: Apache-2.0
# P0 checkpoint: LTX-2.5 config instantiation + field defaults + path utils.
import pytest

from fusion_mlx.video.ltx2_5 import (
    LTX2_5ModelConfig,
    LTX2_5Variant,
    component_keys,
    default_ltx2_5_config,
    resolve_component,
)
from fusion_mlx.video.ltx2_5.config import LTXModelType, LTXRopeType


class TestLTX2_5ConfigDefaults:
    def test_num_layers_48(self):
        cfg = LTX2_5ModelConfig()
        assert cfg.num_layers == 48

    def test_caption_channels_3840(self):
        cfg = LTX2_5ModelConfig()
        assert cfg.caption_channels == 3840

    def test_inner_dim(self):
        cfg = LTX2_5ModelConfig()
        assert cfg.inner_dim == 32 * 128
        assert cfg.inner_dim == 4096

    def test_cross_attention_dim(self):
        cfg = LTX2_5ModelConfig()
        assert cfg.cross_attention_dim == 4096

    def test_vae_scale_factors(self):
        cfg = LTX2_5ModelConfig()
        assert cfg.vae_scale_factors == (8, 32, 32)
        assert cfg.vae_temporal_factor == 8
        assert cfg.vae_spatial_factor == 32

    def test_model_type_audiovideo(self):
        cfg = LTX2_5ModelConfig()
        assert cfg.model_type == LTXModelType.AudioVideo
        assert cfg.model_type.is_video_enabled()
        assert cfg.model_type.is_audio_enabled()

    def test_rope_interleaved(self):
        cfg = LTX2_5ModelConfig()
        assert cfg.rope_type == LTXRopeType.INTERLEAVED

    def test_25_specific_flags(self):
        cfg = LTX2_5ModelConfig()
        assert cfg.has_duration_head is True
        assert cfg.has_two_stage_upsampler is True
        assert cfg.use_prompt_embeddings is True

    def test_in_out_channels(self):
        cfg = LTX2_5ModelConfig()
        assert cfg.in_channels == 128
        assert cfg.out_channels == 128

    def test_get_video_config(self):
        cfg = LTX2_5ModelConfig()
        vc = cfg.get_video_config()
        assert vc is not None
        assert vc.dim == 4096
        assert vc.heads == 32
        assert vc.d_head == 128
        assert vc.context_dim == 4096


class TestLTX2_5Variant:
    def test_distilled_default(self):
        assert LTX2_5Variant.from_str("distilled") == LTX2_5Variant.DISTILLED

    def test_dev(self):
        assert LTX2_5Variant.from_str("dev") == LTX2_5Variant.DEV

    def test_case_insensitive(self):
        assert LTX2_5Variant.from_str("DISTILLED") == LTX2_5Variant.DISTILLED
        assert LTX2_5Variant.from_str("Dev") == LTX2_5Variant.DEV

    def test_passthrough_enum(self):
        assert LTX2_5Variant.from_str(LTX2_5Variant.DEV) == LTX2_5Variant.DEV

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="unknown LTX-2.5 variant"):
            LTX2_5Variant.from_str("bogus")


class TestDefaultConfig:
    def test_distilled_factory(self):
        cfg = default_ltx2_5_config("distilled")
        assert isinstance(cfg, LTX2_5ModelConfig)
        assert cfg.num_layers == 48

    def test_dev_factory(self):
        cfg = default_ltx2_5_config("dev")
        assert cfg.caption_channels == 3840

    def test_default_variant(self):
        cfg = default_ltx2_5_config()
        assert cfg.num_layers == 48


class TestComponentKeys:
    def test_keys_present(self):
        keys = component_keys()
        for expected in (
            "transformer_distilled",
            "transformer_dev",
            "video_vae",
            "audio_vae",
            "text_encoder",
            "duration_head",
            "spatial_upscaler",
            "temporal_upscaler",
        ):
            assert expected in keys

    def test_resolve_component_transformer_distilled(self, tmp_path):
        root = tmp_path
        p = resolve_component(root, "transformer", variant="distilled")
        assert p.name == "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
        assert p.parent.name == "diffusion_models"

    def test_resolve_component_transformer_dev(self, tmp_path):
        p = resolve_component(tmp_path, "transformer", variant="dev")
        assert p.name == "ltx-2.5-22b-dev-transformer-bf16.safetensors"

    def test_resolve_component_explicit_key(self, tmp_path):
        p = resolve_component(tmp_path, "video_vae")
        assert "ltx-2.5-video-vae" in p.name

    def test_resolve_component_duration_head(self, tmp_path):
        p = resolve_component(tmp_path, "duration_head")
        assert p.parent.name == "model_patches"

    def test_resolve_component_temporal_upscaler(self, tmp_path):
        p = resolve_component(tmp_path, "temporal_upscaler")
        assert "temporal" in p.name and "upscaler" in p.name

    def test_resolve_component_invalid_key(self, tmp_path):
        with pytest.raises(ValueError, match="unknown LTX-2.5 component"):
            resolve_component(tmp_path, "bogus")
