# SPDX-License-Identifier: Apache-2.0
"""Unit tests for R-7 profile gate system."""

from fusion_mlx.profile import (
    ALL_MODALITIES,
    profile_from_config,
    resolve_profile,
)


class TestServerProfile:
    def test_lite_disables_all_except_llm(self):
        p = resolve_profile("lite")
        assert p.name == "lite"
        assert p.engine_allowed("llm") is True
        for mod in (
            "image",
            "video",
            "audio",
            "agent",
            "embedding",
            "ner",
            "reranker",
            "ocr",
            "mcp",
            "multitenant",
            "bench",
            "tools",
            "vlm",
        ):
            assert p.engine_allowed(mod) is False, f"lite should disable {mod}"

    def test_standard_disables_heavy_modalities(self):
        p = resolve_profile("standard")
        assert p.name == "standard"
        assert p.engine_allowed("llm") is True
        assert p.engine_allowed("embedding") is True
        assert p.engine_allowed("audio") is True
        assert p.engine_allowed("ner") is True
        assert p.engine_allowed("reranker") is True
        assert p.engine_allowed("ocr") is True
        assert p.engine_allowed("mcp") is True
        assert p.engine_allowed("image") is False
        assert p.engine_allowed("video") is False
        assert p.engine_allowed("agent") is False
        assert p.engine_allowed("multitenant") is False

    def test_full_enables_everything(self):
        p = resolve_profile("full")
        assert p.name == "full"
        for mod in ALL_MODALITIES:
            assert p.engine_allowed(mod) is True, f"full should enable {mod}"

    def test_spec_decode_default(self):
        assert resolve_profile("lite").spec_decode_default() is False
        assert resolve_profile("standard").spec_decode_default() is True
        assert resolve_profile("full").spec_decode_default() is True

    def test_unknown_profile_falls_back_to_standard(self):
        p = resolve_profile("nonexistent")
        assert p.name == "standard"

    def test_default_is_standard(self):
        p = resolve_profile()
        assert p.name == "standard"

    def test_disabled_modules_extends_preset(self):
        p = resolve_profile("standard", disabled_modules=["audio", "ner"])
        assert p.engine_allowed("audio") is False
        assert p.engine_allowed("ner") is False
        assert p.engine_allowed("llm") is True

    def test_disabled_modules_ignores_unknown(self):
        p = resolve_profile("standard", disabled_modules=["fake_modality"])
        assert p.name == "standard"

    def test_summary_contains_profile_name(self):
        p = resolve_profile("lite")
        s = p.summary()
        assert "profile=lite" in s
        assert "spec_decode_default=False" in s


class TestProfileFromConfig:
    def test_config_with_explicit_profile(self):
        class FakeConfig:
            profile = "full"
            disabled_modules = None

        p = profile_from_config(FakeConfig())
        assert p.name == "full"

    def test_config_with_disabled_modules(self):
        class FakeConfig:
            profile = "standard"
            disabled_modules = ["image", "video"]

        p = profile_from_config(FakeConfig())
        assert p.engine_allowed("image") is False
        assert p.engine_allowed("video") is False

    def test_config_no_profile_defaults_to_standard(self):
        class FakeConfig:
            profile = None
            disabled_modules = None

        p = profile_from_config(FakeConfig())
        assert p.name == "standard"


class TestEngineDisabledError:
    def test_error_message_contains_modality_and_profile(self):
        from fusion_mlx.exceptions import EngineDisabledError

        err = EngineDisabledError("image", "lite")
        assert "image" in str(err)
        assert "lite" in str(err)
