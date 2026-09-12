# SPDX-License-Identifier: Apache-2.0
"""G-7 (#0912 audit): TTS voice resolution regression tests.

kitten_tts raises ValueError in mlx_audio _prepare_inputs when the requested
voice is not in the model's voice set. Kokoro's "af_heart" default was passed
to every family, crashing kitten-tts-nano. _resolve_model_voice validates
against the model's introspectable voices and falls back to the first
available voice with a loud warning.
"""

from fusion_mlx.audio.tts import TTSEngine as AudioTTSEngine
from fusion_mlx.engines.tts import TTSEngine as EngineTTSEngine


class _KittenMock:
    voices = {
        "expr-voice-2-f": 1,
        "expr-voice-2-m": 1,
        "expr-voice-5-m": 1,
    }
    voice_aliases = {"Bella": "expr-voice-2-f", "Leo": "expr-voice-5-m"}


class _KokoroMock:
    voices = None
    voice_aliases = None


def _both_resolvers():
    return [EngineTTSEngine, AudioTTSEngine]


def test_available_voices_from_dict():
    for R in _both_resolvers():
        avail = R._available_voices(_KittenMock())
        assert "expr-voice-2-f" in avail
        assert "expr-voice-5-m" in avail


def test_available_voices_includes_aliases():
    for R in _both_resolvers():
        avail = R._available_voices(_KittenMock())
        assert "Bella" in avail
        assert "Leo" in avail


def test_available_voices_empty_for_kokoro():
    for R in _both_resolvers():
        assert R._available_voices(_KokoroMock()) == []


def test_resolve_valid_voice_passthrough():
    for R in _both_resolvers():
        assert (
            R._resolve_model_voice(_KittenMock(), "expr-voice-5-m") == "expr-voice-5-m"
        )


def test_resolve_alias_passthrough():
    for R in _both_resolvers():
        assert R._resolve_model_voice(_KittenMock(), "Bella") == "Bella"


def test_resolve_invalid_voice_falls_back():
    for R in _both_resolvers():
        resolved = R._resolve_model_voice(_KittenMock(), "af_heart")
        assert resolved in R._available_voices(_KittenMock())
        assert resolved != "af_heart"


def test_resolve_none_voice_falls_back_when_voices_present():
    for R in _both_resolvers():
        resolved = R._resolve_model_voice(_KittenMock(), None)
        assert resolved in R._available_voices(_KittenMock())


def test_resolve_none_voice_passthrough_when_no_voices():
    for R in _both_resolvers():
        assert R._resolve_model_voice(_KokoroMock(), None) is None


def test_resolve_invalid_voice_passthrough_when_no_voices():
    for R in _both_resolvers():
        assert R._resolve_model_voice(_KokoroMock(), "af_heart") == "af_heart"
