import pytest

from fusion_mlx.engines.image_gen import (
    VARIANT_MAP,
    ImageGenEngine,
    _infer_variant,
)
from fusion_mlx.image.sd15.config import (
    SD15Config,
    SD15ModelPaths,
    SD15TextEncoderConfig,
    SD15UNetConfig,
    SD15VAEConfig,
)
from fusion_mlx.image.sd15.generate import SD15Pipeline


class TestSD15VariantMap:
    def test_sd15_in_variant_map(self):
        assert "sd15" in VARIANT_MAP
        module_path, cls_name, config_label, default_guidance = VARIANT_MAP["sd15"]
        assert module_path == "fusion_mlx.image.sd15.generate"
        assert cls_name == "SD15Pipeline"
        assert config_label == "sd15_base"
        assert default_guidance == 7.5

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("runwayml/stable-diffusion-v1-5", "sd15"),
            ("stable-diffusion-v1-5", "sd15"),
            ("stable-diffusion-v1-4", "sd15"),
            ("sd15-base", "sd15"),
            ("SD1.5", "sd15"),
        ],
    )
    def test_infer_sd15(self, path, expected):
        assert _infer_variant(path) == expected

    def test_sd15_before_flux_fallback(self):
        assert _infer_variant("foo-sd15-bar") == "sd15"
        assert _infer_variant("runwayml/stable-diffusion-v1-5") == "sd15"


class TestSD15Config:
    def test_unet_config_values(self):
        cfg = SD15UNetConfig()
        assert cfg.block_out_channels == (320, 640, 1280, 1280)
        assert cfg.attention_head_dim == 8
        assert cfg.cross_attention_dim == 768
        assert cfg.transformer_layers_per_block == 1
        assert cfg.in_channels == 4
        assert cfg.out_channels == 4

    def test_vae_config_values(self):
        cfg = SD15VAEConfig()
        assert cfg.latent_channels == 4
        assert cfg.scaling_factor == pytest.approx(0.18215)
        assert cfg.block_out_channels == (128, 256, 512, 512)

    def test_text_encoder_config_values(self):
        cfg = SD15TextEncoderConfig()
        assert cfg.clip_l_dims == 768
        assert cfg.clip_l_layers == 12
        assert cfg.clip_l_heads == 12
        assert cfg.clip_l_intermediate == 3072
        assert cfg.vocab == 49408
        assert cfg.max_pos == 77

    def test_model_paths(self):
        paths = SD15ModelPaths()
        assert paths.repo == "runwayml/stable-diffusion-v1-5"
        assert paths.unet_subfolder == "unet"
        assert paths.vae_subfolder == "vae"
        assert paths.clip_l_subfolder == "text_encoder"
        assert paths.tokenizer_subfolder == "tokenizer"

    def test_top_config_aggregates(self):
        cfg = SD15Config()
        assert isinstance(cfg.unet, SD15UNetConfig)
        assert isinstance(cfg.vae, SD15VAEConfig)
        assert isinstance(cfg.text, SD15TextEncoderConfig)
        assert isinstance(cfg.paths, SD15ModelPaths)


class TestSD15PipelineConstruction:
    def test_construct_without_loading(self):
        pipe = SD15Pipeline(model_config=None, model_path="sd15-base", quantize=None)
        assert pipe.unet is None
        assert pipe.vae is None
        assert pipe.clip_l is None
        assert pipe._loaded is False

    def test_generate_image_accepts_img2img_kwargs(self):
        import inspect

        sig = inspect.signature(SD15Pipeline.generate_image)
        assert "image_path" in sig.parameters
        assert "image_strength" in sig.parameters
        assert sig.parameters["image_path"].default is None
        assert sig.parameters["image_strength"].default is None


class TestSD15EngineImg2ImgForwarding:
    # Verify the engine routes img2img inputs (edit_image/control_image +
    # image_strength) into generate_image for sd15/sdxl/sd3 variants (#480).
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

    @pytest.mark.parametrize("variant", ["sd15", "sdxl", "sd3"])
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

    async def test_txt2img_no_image_path_for_sd15(self):
        eng, captured = self._make_engine("sd15")
        await eng.generate(
            prompt="a cat",
            steps=2,
            width=64,
            height=64,
            n_images=1,
            output_format="raw",
        )
        assert "image_path" not in captured
        assert "image_strength" not in captured

    async def test_negative_prompt_forwarded_for_sd15(self):
        eng, captured = self._make_engine("sd15")
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
