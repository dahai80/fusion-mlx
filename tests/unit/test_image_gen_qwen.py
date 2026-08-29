import pytest

from fusion_mlx.engines.image_gen import (
    VARIANT_MAP,
    ImageGenEngine,
    _infer_variant,
)


class TestQwenImageVariantMap:
    def test_qwen_image_in_variant_map(self):
        assert "qwen_image" in VARIANT_MAP
        module_path, cls_name, config_label, default_guidance = VARIANT_MAP["qwen_image"]
        assert module_path == "mflux.models.qwen.variants.txt2img.qwen_image"
        assert cls_name == "QwenImage"
        assert config_label == "qwen_image"
        assert default_guidance == 4.0

    def test_qwen_image_edit_in_variant_map(self):
        assert "qwen_image_edit" in VARIANT_MAP
        module_path, cls_name, config_label, default_guidance = VARIANT_MAP["qwen_image_edit"]
        assert module_path == "mflux.models.qwen.variants.edit.qwen_image_edit"
        assert cls_name == "QwenImageEdit"
        assert config_label == "qwen_image_edit"
        assert default_guidance == 4.0

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("Qwen/Qwen-Image", "qwen_image"),
            ("Qwen/Qwen-Image-2512", "qwen_image"),
            ("qwen-image", "qwen_image"),
            ("qwen-image-2512", "qwen_image"),
            ("Qwen-Image-2512-4bit", "qwen_image"),
            ("mlx-community/Qwen-Image-2512-4bit", "qwen_image"),
        ],
    )
    def test_infer_qwen_image_txt2img(self, path, expected):
        assert _infer_variant(path) == expected

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("Qwen/Qwen-Image-Edit", "qwen_image_edit"),
            ("qwen-image-edit", "qwen_image_edit"),
            ("Qwen-Image-Edit-2512", "qwen_image_edit"),
            ("mlx-community/Qwen-Image-Edit-2512-4bit", "qwen_image_edit"),
        ],
    )
    def test_infer_qwen_image_edit(self, path, expected):
        assert _infer_variant(path) == expected

    def test_qwen_before_flux_fallback(self):
        assert _infer_variant("Qwen-Image") == "qwen_image"
        assert _infer_variant("qwen-image-edit") == "qwen_image_edit"

    def test_qwen_does_not_match_minimax(self):
        assert _infer_variant("minimax-h3") != "qwen_image"


class TestQwenImageEngineInit:
    def test_inferred_qwen_image_from_path(self):
        eng = ImageGenEngine(model_name="Qwen/Qwen-Image-2512")
        assert eng.variant == "qwen_image"

    def test_inferred_qwen_image_edit_from_path(self):
        eng = ImageGenEngine(model_name="Qwen/Qwen-Image-Edit")
        assert eng.variant == "qwen_image_edit"

    def test_explicit_qwen_image_variant(self):
        eng = ImageGenEngine(model_name="foo", variant="qwen_image")
        assert eng.variant == "qwen_image"

    def test_explicit_qwen_image_edit_variant(self):
        eng = ImageGenEngine(model_name="foo", variant="qwen_image_edit")
        assert eng.variant == "qwen_image_edit"

    def test_qwen_image_4bit_inferred(self):
        eng = ImageGenEngine(model_name="mlx-community/Qwen-Image-2512-4bit")
        assert eng.variant == "qwen_image"
