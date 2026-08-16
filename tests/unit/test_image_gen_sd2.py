import inspect

import pytest

from fusion_mlx.engines.image_gen import (
    VARIANT_MAP,
    ImageGenEngine,
    _infer_variant,
)
from fusion_mlx.image.sd2.config import (
    SD2Config,
    SD2ModelPaths,
    SD2TextEncoderConfig,
    SD2UNetConfig,
    SD2VAEConfig,
)
from fusion_mlx.image.sd2.generate import SD2Pipeline


class TestSD2VariantMap:
    def test_sd2_in_variant_map(self):
        assert "sd2" in VARIANT_MAP
        module_path, cls_name, config_label, default_guidance = VARIANT_MAP["sd2"]
        assert module_path == "fusion_mlx.image.sd2.generate"
        assert cls_name == "SD2Pipeline"
        assert config_label == "sd2_base"
        assert default_guidance == 7.5

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("sd2-community/stable-diffusion-2-1", "sd2"),
            ("stable-diffusion-2-1", "sd2"),
            ("stable-diffusion-2", "sd2"),
            ("sd2.1", "sd2"),
            ("sd21", "sd2"),
            ("sd2-1", "sd2"),
            ("v2-1_768-ema-pruned.ckpt", "sd2"),
        ],
    )
    def test_infer_sd2(self, path, expected):
        assert _infer_variant(path) == expected

    def test_sd2_does_not_collide_with_sd15(self):
        assert _infer_variant("runwayml/stable-diffusion-v1-5") == "sd15"
        assert _infer_variant("stable-diffusion-2-1") == "sd2"

    def test_sd2_before_flux_fallback(self):
        assert _infer_variant("foo-sd2.1-bar") == "sd2"


class TestSD2Config:
    def test_unet_config_values(self):
        cfg = SD2UNetConfig()
        assert cfg.block_out_channels == (320, 640, 1280, 1280)
        assert cfg.attention_head_dim == (5, 10, 20, 20)
        assert cfg.cross_attention_dim == 1024
        assert cfg.transformer_layers_per_block == 1
        assert cfg.use_linear_projection is True
        assert cfg.in_channels == 4
        assert cfg.out_channels == 4

    def test_vae_config_values(self):
        cfg = SD2VAEConfig()
        assert cfg.latent_channels == 4
        assert cfg.scaling_factor == pytest.approx(0.18215)
        assert cfg.block_out_channels == (128, 256, 512, 512)

    def test_text_encoder_config_values(self):
        cfg = SD2TextEncoderConfig()
        assert cfg.hidden_size == 1024
        assert cfg.num_hidden_layers == 23
        assert cfg.num_attention_heads == 16
        assert cfg.intermediate_size == 4096
        assert cfg.hidden_act == "gelu"
        assert cfg.vocab == 49408
        assert cfg.max_pos == 77

    def test_model_paths(self):
        paths = SD2ModelPaths()
        assert paths.repo == "sd2-community/stable-diffusion-2-1"
        assert paths.unet_subfolder == "unet"
        assert paths.vae_subfolder == "vae"
        assert paths.text_subfolder == "text_encoder"
        assert paths.tokenizer_subfolder == "tokenizer"

    def test_top_config_aggregates(self):
        cfg = SD2Config()
        assert isinstance(cfg.unet, SD2UNetConfig)
        assert isinstance(cfg.vae, SD2VAEConfig)
        assert isinstance(cfg.text, SD2TextEncoderConfig)
        assert isinstance(cfg.paths, SD2ModelPaths)


class TestSD2PipelineConstruction:
    def test_construct_without_loading(self):
        pipe = SD2Pipeline(model_config=None, model_path="sd2-base", quantize=None)
        assert pipe.unet is None
        assert pipe.vae is None
        assert pipe.clip_l is None
        assert pipe._loaded is False

    def test_generate_image_accepts_img2img_kwargs(self):
        sig = inspect.signature(SD2Pipeline.generate_image)
        assert "image_path" in sig.parameters
        assert "image_strength" in sig.parameters
        assert sig.parameters["image_path"].default is None
        assert sig.parameters["image_strength"].default is None


class TestSD2EngineForwarding:
    def _make_engine(self, variant):
        eng = ImageGenEngine(model_name="foo", variant=variant)
        captured = {}

        class _StubImage:
            pass

        class _StubGen:
            image = _StubImage()

        class _StubPipe:
            def generate_image(self, **kwargs):
                captured.update(kwargs)
                return _StubGen()

        eng._flux = _StubPipe()
        eng._mflux_missing = False
        return eng, captured

    @pytest.mark.parametrize("variant", ["sd2", "sd15"])
    async def test_img2img_forwards_image_path_and_strength(self, variant):
        eng, captured = self._make_engine(variant)
        result = await eng.generate(
            prompt="a cat",
            edit_image="/tmp/init.png",
            image_strength=0.6,
            steps=2,
            width=64,
            height=64,
            n_images=1,
            output_format="raw",
        )
        assert captured.get("image_path") == "/tmp/init.png"
        assert captured.get("image_strength") == pytest.approx(0.6)
        assert len(result) == 1

    async def test_negative_prompt_forwarded_for_sd2(self):
        eng, captured = self._make_engine("sd2")
        await eng.generate(
            prompt="a cat",
            negative_prompt="blurry",
            steps=2,
            width=64,
            height=64,
            n_images=1,
            output_format="raw",
        )
        assert captured.get("negative_prompt") == "blurry"


class TestSD2QuantizationDowngrade:
    # SD2 v_prediction + int8/int4 在 >768 分辨率数值不稳定 (UNet 输出
    # 累积量化误差溢出为 NaN, 已验证 8bit/4bit @1152 均 NaN, fp16 正常).
    # 引擎层必须把 SD2 量化降级为 None, 保证 hires-fix 2pass 不黑图.
    def test_sd2_quantize_8_downgraded_to_none(self):
        eng = ImageGenEngine(
            model_name="sd2-community/stable-diffusion-2-1", quantize=8
        )
        assert eng._variant == "sd2"
        assert eng._quantize is None

    def test_sd2_quantize_4_downgraded_to_none(self):
        eng = ImageGenEngine(
            model_name="sd2-community/stable-diffusion-2-1", quantize=4
        )
        assert eng._quantize is None

    def test_sd15_quantize_8_not_downgraded(self):
        # SD1.5 8bit @1152 稳定, 不应被降级.
        eng = ImageGenEngine(model_name="runwayml/stable-diffusion-v1-5", quantize=8)
        assert eng._variant == "sd15"
        assert eng._quantize == 8

    def test_sd2_no_quantize_stays_none(self):
        eng = ImageGenEngine(
            model_name="sd2-community/stable-diffusion-2-1", quantize=None
        )
        assert eng._quantize is None
