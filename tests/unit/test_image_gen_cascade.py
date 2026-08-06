import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.engines.image_gen import VARIANT_MAP, _infer_variant
from fusion_mlx.image.cascade.config import (
    CascadeConfig,
    DecoderConfig,
    PriorConfig,
    VQGANConfig,
)
from fusion_mlx.image.cascade.generate import CascadePipeline
from fusion_mlx.image.cascade.scheduler import DDPMWuerstchenScheduler
from fusion_mlx.image.cascade.text_encoder import CascadeCLIPTextModel
from fusion_mlx.image.cascade.unet import StableCascadeUNet
from fusion_mlx.image.cascade.vqgan import PaellaVQModel


class TestCascadeVariantMap:
    def test_stable_cascade_in_variant_map(self):
        assert "stable_cascade" in VARIANT_MAP
        module_path, cls_name, config_label, default_guidance = VARIANT_MAP[
            "stable_cascade"
        ]
        assert module_path == "fusion_mlx.image.cascade.generate"
        assert cls_name == "CascadePipeline"
        assert config_label == "stable_cascade"
        assert default_guidance == 4.0

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("stable-cascade", "stable_cascade"),
            ("stabilityai/stable-cascade-prior", "stable_cascade"),
            ("Stable-Cascade", "stable_cascade"),
            ("wuerstchen", "stable_cascade"),
            ("wuerstchen-v2", "stable_cascade"),
        ],
    )
    def test_infer_cascade(self, path, expected):
        assert _infer_variant(path) == expected


class TestCascadeConfigs:
    def test_prior_config_defaults(self):
        cfg = PriorConfig()
        assert cfg.in_channels == 16
        assert cfg.out_channels == 16
        assert cfg.conditioning_dim == 2048
        assert cfg.block_out_channels == (2048, 2048)
        assert cfg.num_attention_heads == (32, 32)
        assert cfg.clip_text_in_channels == 1280
        assert cfg.clip_image_in_channels == 768
        assert cfg.effnet_in_channels is None
        assert cfg.pixel_mapper_in_channels is None
        assert cfg.patch_size == 1
        assert cfg.timestep_conditioning_type == ("sca", "crp")

    def test_decoder_config_defaults(self):
        cfg = DecoderConfig()
        assert cfg.in_channels == 4
        assert cfg.out_channels == 4
        assert cfg.conditioning_dim == 1280
        assert cfg.block_out_channels == (320, 640, 1280, 1280)
        assert cfg.num_attention_heads == (0, 0, 20, 20)
        assert cfg.effnet_in_channels == 16
        assert cfg.pixel_mapper_in_channels == 3
        assert cfg.clip_text_in_channels is None
        assert cfg.clip_image_in_channels is None
        assert cfg.patch_size == 2
        assert cfg.up_blocks_repeat_mappers == (3, 3, 2, 2)
        assert cfg.timestep_conditioning_type == ("sca",)

    def test_vqgan_config_defaults(self):
        cfg = VQGANConfig()
        assert cfg.latent_channels == 4
        assert cfg.embed_dim == 384
        assert cfg.scale_factor == 0.3764
        assert cfg.num_vq_embeddings == 8192

    def test_cascade_config_resolution(self):
        cfg = CascadeConfig()
        assert cfg.resolution_multiple == 42.67
        assert cfg.latent_dim_scale == 10.67
        assert cfg.scheduler_s == 0.008


class TestScheduler:
    def test_set_timesteps_linspace(self):
        sch = DDPMWuerstchenScheduler()
        sch.set_timesteps(10)
        assert len(sch.timesteps) == 11
        assert abs(sch.timesteps[0] - 1.0) < 1e-6
        assert abs(sch.timesteps[-1] - 0.0) < 1e-6

    def test_init_noise_sigma(self):
        sch = DDPMWuerstchenScheduler()
        assert abs(sch.init_noise_sigma - 1.0) < 1e-6

    def test_step_preserves_shape(self):
        sch = DDPMWuerstchenScheduler()
        sch.set_timesteps(4)
        ts = sch.timesteps[:-1]
        sample = mx.random.normal((1, 4, 8, 8), dtype=mx.float32)
        model_out = mx.zeros((1, 4, 8, 8), dtype=mx.float32)
        out = sch.step(model_out, mx.array([float(ts[0])]), sample)
        mx.eval(out)
        assert out.shape == sample.shape


class TestUnetStructure:
    def test_prior_has_no_effnet_mapper(self):
        unet = StableCascadeUNet(PriorConfig())
        assert not hasattr(unet, "effnet_mapper")
        assert not hasattr(unet, "pixels_mapper")
        assert hasattr(unet, "clip_txt_mapper")
        assert hasattr(unet, "clip_img_mapper")

    def test_decoder_has_effnet_mapper(self):
        unet = StableCascadeUNet(DecoderConfig())
        assert hasattr(unet, "effnet_mapper")
        assert hasattr(unet, "pixels_mapper")
        assert unet.clip_txt_mapper is None
        assert unet.clip_img_mapper is None

    def test_prior_param_count_nonzero(self):
        unet = StableCascadeUNet(PriorConfig())
        n = sum(v.size for _, v in nn.utils.tree_flatten(unet.parameters()))
        assert n > 0

    def test_decoder_param_count_nonzero(self):
        unet = StableCascadeUNet(DecoderConfig())
        n = sum(v.size for _, v in nn.utils.tree_flatten(unet.parameters()))
        assert n > 0

    def test_prior_forward_shape(self):
        cfg = PriorConfig()
        unet = StableCascadeUNet(cfg)
        b = 2
        sample = mx.zeros((b, cfg.in_channels, 4, 4), dtype=mx.float32)
        t = mx.zeros((b,), dtype=mx.float32)
        pooled = mx.zeros((b, 1, cfg.clip_text_pooled_in_channels), dtype=mx.float32)
        hidden = mx.zeros((b, 5, cfg.clip_text_in_channels), dtype=mx.float32)
        img = mx.zeros((b, 1, cfg.clip_image_in_channels), dtype=mx.float32)
        out = unet(
            sample=sample,
            timestep_ratio=t,
            clip_text_pooled=pooled,
            clip_text=hidden,
            clip_img=img,
        )
        mx.eval(out)
        assert out.shape == (b, cfg.out_channels, 4, 4)

    def test_decoder_forward_shape(self):
        cfg = DecoderConfig()
        unet = StableCascadeUNet(cfg)
        b = 2
        sample = mx.zeros((b, cfg.in_channels, 8, 8), dtype=mx.float32)
        t = mx.zeros((b,), dtype=mx.float32)
        pooled = mx.zeros((b, 1, cfg.clip_text_pooled_in_channels), dtype=mx.float32)
        effnet = mx.zeros((b, cfg.effnet_in_channels, 4, 4), dtype=mx.float32)
        out = unet(
            sample=sample,
            timestep_ratio=t,
            clip_text_pooled=pooled,
            effnet=effnet,
        )
        mx.eval(out)
        assert out.shape == (b, cfg.out_channels, 8, 8)


class TestVQGANStructure:
    def test_vqgan_param_count_nonzero(self):
        vc = VQGANConfig()
        model = PaellaVQModel(
            in_channels=vc.in_channels,
            out_channels=vc.out_channels,
            up_down_scale_factor=vc.up_down_scale_factor,
            levels=vc.levels,
            bottleneck_blocks=vc.bottleneck_blocks,
            embed_dim=vc.embed_dim,
            latent_channels=vc.latent_channels,
            scale_factor=vc.scale_factor,
        )
        n = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
        assert n > 0

    def test_vqgan_decode_shape(self):
        vc = VQGANConfig()
        model = PaellaVQModel(
            in_channels=vc.in_channels,
            out_channels=vc.out_channels,
            up_down_scale_factor=vc.up_down_scale_factor,
            levels=vc.levels,
            bottleneck_blocks=vc.bottleneck_blocks,
            embed_dim=vc.embed_dim,
            latent_channels=vc.latent_channels,
            scale_factor=vc.scale_factor,
        )
        latent = mx.zeros((1, vc.latent_channels, 8, 8), dtype=mx.float32)
        out = model.decode(latent)
        mx.eval(out)
        assert out.shape[0] == 1
        assert out.shape[1] == vc.out_channels


class TestCLIPTextModel:
    def test_clip_bigg_param_names(self):
        model = CascadeCLIPTextModel()
        names = {k for k, _ in nn.utils.tree_flatten(model.parameters())}
        assert "text_model.embeddings.token_embedding.weight" in names
        assert "text_model.encoder.layers.31.self_attn.q_proj.weight" in names
        assert "text_model.final_layer_norm.weight" in names

    def test_clip_pooled_output_shape(self):
        model = CascadeCLIPTextModel()
        tokens = mx.zeros((2, 77), dtype=mx.int32)
        hidden, pooled = model(tokens)
        mx.eval(hidden, pooled)
        assert hidden.shape == (2, 77, 1280)
        assert pooled.shape == (2, 1280)


class TestPipelineConstruction:
    def test_construct_without_loading(self):
        pipe = CascadePipeline(
            model_config=None,
            model_path="stabilityai/stable-cascade-prior",
            quantize=None,
        )
        assert pipe.config.prior.in_channels == 16
        assert pipe.config.decoder.in_channels == 4
        assert pipe.prior is None
        assert pipe.decoder is None
        assert pipe.vqgan is None
