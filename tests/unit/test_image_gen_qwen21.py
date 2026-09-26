# SPDX-License-Identifier: Apache-2.0
# Tests for Qwen-Image-2.1 transparent RGBA PNG backend (PR#736 vendor).
# No real model load — routing + request validation + variant map only.

import pytest

from fusion_mlx.engines.image_gen import (
    VARIANT_MAP,
    ImageGenEngine,
    _infer_variant,
)


class TestQwenImage21VariantMap:
    def test_qwen_image_21_in_variant_map(self):
        assert "qwen_image_21" in VARIANT_MAP
        module_path, cls_name, config_label, default_guidance = VARIANT_MAP[
            "qwen_image_21"
        ]
        assert module_path == "fusion_mlx.engines.qwen_image_21"
        assert cls_name == "QwenImage21"
        assert config_label == "qwen_image_21"
        assert default_guidance == 4.0

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("Qwen/Qwen-Image-2.1", "qwen_image_21"),
            ("qwen-image-2.1", "qwen_image_21"),
            ("qwen-2.1", "qwen_image_21"),
            ("qwen-image-21", "qwen_image_21"),
            ("Qwen-Image-2.1", "qwen_image_21"),
        ],
    )
    def test_infer_qwen_image_21(self, path, expected):
        assert _infer_variant(path) == expected

    def test_21_routes_before_generic_qwen_image(self):
        # "qwen-image-2.1" contains "qwen-image" — must NOT collapse to qwen_image.
        assert _infer_variant("qwen-image-2.1") == "qwen_image_21"
        assert _infer_variant("Qwen/Qwen-Image-2.1") == "qwen_image_21"

    def test_edit_still_routes_correctly(self):
        # Qwen-Image-Edit (2512/2509) is a distinct model from Qwen-Image-2.1;
        # the edit substring check runs before the 2.1 check.
        assert _infer_variant("qwen-image-edit") == "qwen_image_edit"
        assert _infer_variant("Qwen/Qwen-Image-Edit-2509") == "qwen_image_edit"

    def test_plain_qwen_image_unaffected(self):
        assert _infer_variant("Qwen/Qwen-Image") == "qwen_image"
        assert _infer_variant("qwen-image-2512") == "qwen_image"


class TestQwenImage21EngineInit:
    def test_inferred_from_path(self):
        eng = ImageGenEngine(model_name="Qwen/Qwen-Image-2.1")
        assert eng.variant == "qwen_image_21"

    def test_explicit_variant(self):
        eng = ImageGenEngine(model_name="foo", variant="qwen_image_21")
        assert eng.variant == "qwen_image_21"


class TestTransparentRequest:
    def test_transparent_field_default_false(self):
        from fusion_mlx.api.images import ImageGenerateRequest

        req = ImageGenerateRequest(prompt="a red apple")
        assert req.transparent is False

    def test_transparent_field_true(self):
        from fusion_mlx.api.images import ImageGenerateRequest

        req = ImageGenerateRequest(prompt="a red apple", transparent=True)
        assert req.transparent is True


class TestQwenImage21Import:
    def test_vendored_qwen21_imports(self):
        # mflux is an optional [image] extra (vendored wheel). Skip on CI
        # shards that don't install it instead of ModuleNotFoundError —
        # matches the guard in test_image_gen_flux2.py.
        pytest.importorskip("mflux")
        # Bootstrap patches ModelConfig.qwen_image_21 + re-exports QwenImage21.
        import fusion_mlx.engines.qwen_image_21 as pkg

        assert hasattr(pkg, "QwenImage21")
        from mflux.models.common.config.model_config import ModelConfig

        assert hasattr(ModelConfig, "qwen_image_21")
        cfg = ModelConfig.qwen_image_21()
        assert cfg.model_name == "Qwen/Qwen-Image-2.1"
        assert "qwen-image-2.1" in cfg.aliases
