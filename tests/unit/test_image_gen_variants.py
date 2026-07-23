import pytest

from fusion_mlx.engines.image_gen import (
    VARIANT_MAP,
    ImageGenEngine,
    _infer_flux2_config,
    _infer_variant,
    _prompt_hash,
    _text_cache_enabled,
)


class TestVariantMap:
    def test_all_variants_have_required_keys(self):
        for name, entry in VARIANT_MAP.items():
            assert len(entry) == 4, f"variant '{name}' must have 4-tuple"
            module_path, cls_name, config_label, default_guidance = entry
            assert isinstance(module_path, str) and module_path
            assert isinstance(cls_name, str) and cls_name
            assert isinstance(config_label, str) and config_label
            assert isinstance(default_guidance, (int, float))

    def test_txt2img_exists(self):
        assert "txt2img" in VARIANT_MAP

    def test_controlnet_canny_exists(self):
        assert "controlnet_canny" in VARIANT_MAP

    def test_depth_exists(self):
        assert "depth" in VARIANT_MAP

    def test_fill_exists(self):
        assert "fill" in VARIANT_MAP

    def test_kontext_exists(self):
        assert "kontext" in VARIANT_MAP

    def test_redux_exists(self):
        assert "redux" in VARIANT_MAP


class TestInferVariant:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("FLUX.1-dev-ControlNet-Canny", "controlnet_canny"),
            ("flux-1-dev-controlnet-canny", "controlnet_canny"),
            ("FLUX.1-dev-ControlNet-Upscaler", "controlnet_upscaler"),
            ("flux-controlnet-upscaler", "controlnet_upscaler"),
            ("FLUX.1-Depth", "depth"),
            ("flux-depth-dev", "depth"),
            ("FLUX.1-Fill", "fill"),
            ("flux-fill-dev", "fill"),
            ("FLUX.1-Kontext", "kontext"),
            ("flux-kontext-dev", "kontext"),
            ("FLUX.1-Redux", "redux"),
            ("flux-redux-dev", "redux"),
            ("FLUX.2-klein", "txt2img"),
            ("some-llm-model", "txt2img"),
            ("", "txt2img"),
        ],
    )
    def test_variant_inference(self, path, expected):
        assert _infer_variant(path) == expected


class TestInferFlux2Config:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("FLUX.2-klein-base-4B", "flux2_klein_base_4b"),
            ("FLUX.2-klein-base-9B", "flux2_klein_base_9b"),
            ("FLUX.2-klein-4B", "flux2_klein_4b"),
            ("FLUX.2-klein-9B-KV", "flux2_klein_9b_kv"),
            ("FLUX.2-klein", "flux2_klein_9b"),
            ("flux-2", "flux2_klein_9b"),
        ],
    )
    def test_flux2_config(self, path, expected):
        assert _infer_flux2_config(path) == expected


class TestImageGenEngineInit:
    def test_default_variant_is_txt2img(self):
        eng = ImageGenEngine(model_name="flux-2")
        assert eng.variant == "txt2img"

    def test_explicit_variant(self):
        eng = ImageGenEngine(model_name="flux-depth", variant="depth")
        assert eng.variant == "depth"

    def test_inferred_variant_from_path(self):
        eng = ImageGenEngine(model_name="FLUX.1-Redux-dev")
        assert eng.variant == "redux"

    def test_unknown_variant_falls_back(self):
        eng = ImageGenEngine(model_name="flux-2", variant="nonexistent")
        assert eng.variant == "txt2img"

    def test_variant_property(self):
        eng = ImageGenEngine(model_name="flux-2", variant="kontext")
        assert eng.variant == "kontext"

    def test_stats_includes_variant(self):
        eng = ImageGenEngine(model_name="flux-2", variant="fill")
        stats = eng.get_stats()
        assert stats["variant"] == "fill"

    def test_repr_includes_variant(self):
        eng = ImageGenEngine(model_name="flux-2", variant="depth")
        r = repr(eng)
        assert "variant=depth" in r


class TestTextCache:
    def test_prompt_hash_deterministic(self):
        h1 = _prompt_hash("hello world")
        h2 = _prompt_hash("hello world")
        assert h1 == h2
        assert len(h1) == 16

    def test_prompt_hash_different_inputs(self):
        h1 = _prompt_hash("hello")
        h2 = _prompt_hash("world")
        assert h1 != h2

    def test_text_cache_enabled_default(self):
        import os

        old = os.environ.pop("FUSION_DIFFUSION_TEXT_CACHE", None)
        try:
            assert _text_cache_enabled() is True
        finally:
            if old is not None:
                os.environ["FUSION_DIFFUSION_TEXT_CACHE"] = old

    def test_text_cache_disabled(self):
        import os

        old = os.environ.get("FUSION_DIFFUSION_TEXT_CACHE")
        os.environ["FUSION_DIFFUSION_TEXT_CACHE"] = "0"
        try:
            assert _text_cache_enabled() is False
        finally:
            if old is not None:
                os.environ["FUSION_DIFFUSION_TEXT_CACHE"] = old
            else:
                del os.environ["FUSION_DIFFUSION_TEXT_CACHE"]

    def test_engine_has_text_cache_by_default(self):
        eng = ImageGenEngine(model_name="flux-2")
        assert eng._text_cache is not None

    def test_engine_text_cache_disabled(self):
        import os

        old = os.environ.get("FUSION_DIFFUSION_TEXT_CACHE")
        os.environ["FUSION_DIFFUSION_TEXT_CACHE"] = "0"
        try:
            eng = ImageGenEngine(model_name="flux-2")
            assert eng._text_cache is None
        finally:
            if old is not None:
                os.environ["FUSION_DIFFUSION_TEXT_CACHE"] = old
            else:
                del os.environ["FUSION_DIFFUSION_TEXT_CACHE"]

    def test_stats_includes_text_cache(self):
        eng = ImageGenEngine(model_name="flux-2")
        stats = eng.get_stats()
        assert "text_cache" in stats

    def test_stats_no_text_cache_when_disabled(self):
        import os

        old = os.environ.get("FUSION_DIFFUSION_TEXT_CACHE")
        os.environ["FUSION_DIFFUSION_TEXT_CACHE"] = "0"
        try:
            eng = ImageGenEngine(model_name="flux-2")
            stats = eng.get_stats()
            assert "text_cache" not in stats
        finally:
            if old is not None:
                os.environ["FUSION_DIFFUSION_TEXT_CACHE"] = old
            else:
                del os.environ["FUSION_DIFFUSION_TEXT_CACHE"]
