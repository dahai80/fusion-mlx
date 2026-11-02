# SPDX-License-Identifier: Apache-2.0
# #1486 item 1: Spark-TTS 0.5B as a TTS backend. Covers family detection,
# discovery classification (spark ships model_type "qwen2" — must not be
# classified as LLM), and voice-cloning capability detection.

import json

from fusion_mlx.engines.tts import TTSEngine
from fusion_mlx.pool.model_discovery import detect_model_type


class TestSparkFamilyDetection:
    def test_family_detect_by_name(self):
        assert TTSEngine._detect_family("Spark-TTS-0.5B-bf16") == "spark"

    def test_family_detect_repo_id(self):
        assert TTSEngine._detect_family("mlx-community/Spark-TTS-0.5B-4bit") == (
            "spark"
        )

    def test_family_other_models_unchanged(self):
        assert TTSEngine._detect_family("kokoro") == "kokoro"
        assert TTSEngine._detect_family("qwen3.5-9b-4bit") is None

    def test_other_tts_names_not_swallowed_by_spark(self):
        # No real TTS family name contains "spark" other than spark itself,
        # but guard against over-broad matches regressing other families.
        assert TTSEngine._detect_family("cosyvoice-300m") == "cosyvoice"
        assert TTSEngine._detect_family("vibevoice-1.5B") == "vibevoice"


class TestSparkDiscovery:
    def test_spark_config_classifies_audio_tts(self, tmp_path):
        # mlx-community Spark conversions carry model_type "qwen2" +
        # Qwen2ForCausalLM (the LLM base). Without the name check discovery
        # files them under LLM and /v1/audio/speech never sees them.
        model_dir = tmp_path / "Spark-TTS-0.5B-bf16"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
        )
        assert detect_model_type(model_dir) == "audio_tts"

    def test_plain_qwen2_llm_still_llm(self, tmp_path):
        model_dir = tmp_path / "Qwen3.5-9B-Instruct"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
        )
        assert detect_model_type(model_dir) == "llm"


class TestSparkVoiceCloning:
    def test_cloning_detected_from_generate_signature(self):
        # Spark-TTS-0.5B-bf16 on disk: real mlx_audio load is covered by
        # real-model tests; here verify the discriminator — Spark's
        # generate(ref_audio=...) signature enables cloning support.
        eng = TTSEngine("Spark-TTS-0.5B-bf16")

        class _FakeSpark:
            def generate(self, text, ref_audio=None, ref_text=None, **kw):
                return []

        eng._model = _FakeSpark()
        assert eng.supports_voice_cloning() is True

    def test_no_cloning_without_ref_audio_param(self):
        eng = TTSEngine("kokoro")

        class _FakeKokoro:
            def generate(self, text, voice=None, speed=1.0, **kw):
                return []

        eng._model = _FakeKokoro()
        assert eng.supports_voice_cloning() is False
