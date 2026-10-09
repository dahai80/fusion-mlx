# SPDX-License-Identifier: Apache-2.0
"""#1085: Mandarin (zh-CN) TTS support.

Covers:
- is_cjk_text: CJK detection.
- normalize_cjk_text: K-12 math notation normalization (percent, fraction,
  decimal, math symbols).
- is_mandarin_voice / resolve_mandarin_voice: voice routing.
- kokoro_voice_capabilities: voice metadata for the /v1/audio/voices endpoint.
- GET /v1/audio/voices: endpoint returns voice capability metadata.
"""

from fusion_mlx.audio.cjk import (
    is_cjk_text,
    is_mandarin_voice,
    kokoro_voice_capabilities,
    normalize_cjk_text,
    resolve_mandarin_voice,
)


class TestIsCjkText:
    def test_pure_chinese(self):
        assert is_cjk_text("你好世界") is True

    def test_mixed_cjk_ascii(self):
        assert is_cjk_text("Hello 世界") is True

    def test_pure_english(self):
        assert is_cjk_text("Hello world") is False

    def test_empty(self):
        assert is_cjk_text("") is False

    def test_none(self):
        assert is_cjk_text(None) is False  # type: ignore[arg-type]

    def test_numbers_only(self):
        assert is_cjk_text("12345") is False


class TestNormalizeCjkText:
    def test_percent(self):
        assert "百分之五十六" in normalize_cjk_text("56%")

    def test_decimal(self):
        result = normalize_cjk_text("3.14")
        assert "三点一四" in result

    def test_fraction(self):
        result = normalize_cjk_text("3/4")
        assert "四分之三" in result

    def test_pi_symbol(self):
        assert "圆周率" in normalize_cjk_text("π")

    def test_degree_celsius(self):
        assert "摄氏度" in normalize_cjk_text("25°C")

    def test_multiply_divide(self):
        result = normalize_cjk_text("6×7÷2")
        assert "乘以" in result
        assert "除以" in result

    def test_no_change_for_plain_chinese(self):
        text = "你好世界"
        assert normalize_cjk_text(text) == text

    def test_combined_math(self):
        result = normalize_cjk_text("面积是π×r²，约等于3.14")
        assert "圆周率" in result
        assert "乘以" in result
        assert "约等于" in result
        assert "三点一四" in result

    def test_empty(self):
        assert normalize_cjk_text("") == ""


class TestMandarinVoice:
    def test_is_mandarin_voice_zf(self):
        assert is_mandarin_voice("zf_xiaoxiao") is True

    def test_is_mandarin_voice_zm(self):
        assert is_mandarin_voice("zm_yunjian") is True

    def test_is_mandarin_voice_english(self):
        assert is_mandarin_voice("af_heart") is False

    def test_is_mandarin_voice_none(self):
        assert is_mandarin_voice(None) is False

    def test_resolve_mandarin_voice_keeps_mandarin(self):
        assert resolve_mandarin_voice("zf_xiaoni") == "zf_xiaoni"

    def test_resolve_mandarin_voice_overrides_english(self):
        assert resolve_mandarin_voice("af_heart") == "zf_xiaoxiao"

    def test_resolve_mandarin_voice_overrides_none(self):
        assert resolve_mandarin_voice(None) == "zf_xiaoxiao"


class TestKokoroVoiceCapabilities:
    def test_returns_list_of_dicts(self):
        caps = kokoro_voice_capabilities()
        assert isinstance(caps, list)
        assert len(caps) >= 5
        for entry in caps:
            assert "voice" in entry
            assert "languages" in entry
            assert "gender" in entry
            assert "zh" in entry["languages"]

    def test_includes_xiaoxiao(self):
        caps = kokoro_voice_capabilities()
        voices = [e["voice"] for e in caps]
        assert "zf_xiaoxiao" in voices


class TestVoicesEndpoint:
    def test_list_voices_returns_metadata(self):
        import asyncio

        from fusion_mlx.api.audio_routes import list_voices

        result = asyncio.run(list_voices())
        assert "voices" in result
        voices = result["voices"]
        assert isinstance(voices, list)
        assert len(voices) > 0
        kokoro_entry = next((v for v in voices if v.get("family") == "kokoro"), None)
        assert kokoro_entry is not None
        assert kokoro_entry["default_voice"] == "af_heart"
        assert "en" in kokoro_entry["languages"]
        assert "zh" in kokoro_entry["languages"]
        assert "mandarin_voices" in kokoro_entry
