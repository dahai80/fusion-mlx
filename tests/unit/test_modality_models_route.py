# SPDX-License-Identifier: Apache-2.0
"""Tests for modality field on /v1/models response (#251)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fusion_mlx.api.openai_models import ModelInfo
from fusion_mlx.model_aliases import AliasProfile
from fusion_mlx.routes_internal.models import (
    _MODEL_TYPE_TO_MODALITY,
    _entry_payload,
    _resolve_modality,
)


class TestModelInfoModality:
    def test_default_modality_is_text(self):
        info = ModelInfo(id="test-model")
        assert info.modality == "text"

    def test_explicit_modality(self):
        info = ModelInfo(id="flux-sdxl", modality="image")
        assert info.modality == "image"

    def test_video_modality(self):
        info = ModelInfo(id="wan2-14b", modality="video")
        assert info.modality == "video"

    def test_audio_modality(self):
        info = ModelInfo(id="whisper-large", modality="audio")
        assert info.modality == "audio"


class TestModelTypeToModality:
    def test_llm_is_text(self):
        assert _MODEL_TYPE_TO_MODALITY["llm"] == "text"

    def test_vlm_is_text(self):
        assert _MODEL_TYPE_TO_MODALITY["vlm"] == "text"

    def test_embedding_is_text(self):
        assert _MODEL_TYPE_TO_MODALITY["embedding"] == "text"

    def test_reranker_is_text(self):
        assert _MODEL_TYPE_TO_MODALITY["reranker"] == "text"

    def test_ner_is_text(self):
        assert _MODEL_TYPE_TO_MODALITY["ner"] == "text"

    def test_audio_stt_is_audio(self):
        assert _MODEL_TYPE_TO_MODALITY["audio_stt"] == "audio"

    def test_audio_tts_is_audio(self):
        assert _MODEL_TYPE_TO_MODALITY["audio_tts"] == "audio"

    def test_audio_sts_is_audio(self):
        assert _MODEL_TYPE_TO_MODALITY["audio_sts"] == "audio"

    def test_image_is_image(self):
        assert _MODEL_TYPE_TO_MODALITY["image"] == "image"

    def test_video_is_video(self):
        assert _MODEL_TYPE_TO_MODALITY["video"] == "video"


class TestResolveModality:
    def test_alias_profile_with_modality(self):
        profile = AliasProfile(name="flux-sdxl", hf_path="x/y", modality="image")
        with patch(
            "fusion_mlx.routes_internal.models.resolve_profile", return_value=profile
        ):
            assert _resolve_modality("flux-sdxl") == "image"

    def test_alias_profile_empty_modality_falls_through(self):
        profile = AliasProfile(name="qwen3", hf_path="x/y", modality="")
        with patch(
            "fusion_mlx.routes_internal.models.resolve_profile", return_value=profile
        ):
            assert _resolve_modality("qwen3") == "text"

    def test_no_profile_no_pool_defaults_text(self):
        with patch(
            "fusion_mlx.routes_internal.models.resolve_profile", return_value=None
        ):
            assert _resolve_modality("unknown-model") == "text"

    def test_pool_entry_model_type_resolved(self):
        entry = MagicMock()
        entry.model_type = "video"
        pool = MagicMock()
        pool.get_entry.return_value = entry
        with (
            patch(
                "fusion_mlx.routes_internal.models.resolve_profile", return_value=None
            ),
            patch("fusion_mlx.routes_internal.models._pool", pool),
        ):
            assert _resolve_modality("wan2-14b") == "video"

    def test_pool_entry_image_type(self):
        entry = MagicMock()
        entry.model_type = "image"
        pool = MagicMock()
        pool.get_entry.return_value = entry
        with (
            patch(
                "fusion_mlx.routes_internal.models.resolve_profile", return_value=None
            ),
            patch("fusion_mlx.routes_internal.models._pool", pool),
        ):
            assert _resolve_modality("flux-sdxl") == "image"

    def test_pool_entry_audio_type(self):
        entry = MagicMock()
        entry.model_type = "audio_stt"
        pool = MagicMock()
        pool.get_entry.return_value = entry
        with (
            patch(
                "fusion_mlx.routes_internal.models.resolve_profile", return_value=None
            ),
            patch("fusion_mlx.routes_internal.models._pool", pool),
        ):
            assert _resolve_modality("whisper-large") == "audio"

    def test_profile_modality_takes_priority_over_pool(self):
        profile = AliasProfile(name="flux-sdxl", hf_path="x/y", modality="image")
        entry = MagicMock()
        entry.model_type = "llm"
        pool = MagicMock()
        pool.get_entry.return_value = entry
        with (
            patch(
                "fusion_mlx.routes_internal.models.resolve_profile",
                return_value=profile,
            ),
            patch("fusion_mlx.routes_internal.models._pool", pool),
        ):
            assert _resolve_modality("flux-sdxl") == "image"

    def test_pool_entry_unknown_model_type_defaults_text(self):
        entry = MagicMock()
        entry.model_type = "unknown_future_type"
        pool = MagicMock()
        pool.get_entry.return_value = entry
        with (
            patch(
                "fusion_mlx.routes_internal.models.resolve_profile", return_value=None
            ),
            patch("fusion_mlx.routes_internal.models._pool", pool),
        ):
            assert _resolve_modality("future-model") == "text"


class TestEntryPayload:
    def test_payload_includes_modality(self):
        payload = _entry_payload("test", None, None, "video")
        assert payload["modality"] == "video"

    def test_payload_defaults_text(self):
        payload = _entry_payload("test", None, None)
        assert payload["modality"] == "text"

    def test_payload_all_fields(self):
        payload = _entry_payload("model-id", "hermes", "qwen", "image")
        assert payload == {
            "id": "model-id",
            "object": "model",
            "tool_call_parser": "hermes",
            "reasoning_parser": "qwen",
            "modality": "image",
            "loaded": True,
            "state": "loaded",
        }


class TestOpenaiRoutesModality:
    def test_resolve_modality_in_openai_routes(self):
        from fusion_mlx.api.openai_routes import _resolve_modality as or_modality

        profile = AliasProfile(name="wan2-14b", hf_path="x/y", modality="video")
        with patch("fusion_mlx.model_aliases.resolve_profile", return_value=profile):
            assert or_modality("wan2-14b") == "video"

    def test_resolve_modality_default_text(self):
        from fusion_mlx.api.openai_routes import _resolve_modality as or_modality

        with (
            patch("fusion_mlx.model_aliases.resolve_profile", return_value=None),
            patch("fusion_mlx.api.openai_routes._pool", None),
        ):
            assert or_modality("unknown") == "text"


class TestAliasesJsonModality:
    def test_all_aliases_have_modality(self):
        import json
        from pathlib import Path

        aliases_path = (
            Path(__file__).parent.parent.parent / "fusion_mlx" / "aliases.json"
        )
        data = json.loads(aliases_path.read_text())
        for name, entry in data.items():
            if isinstance(entry, dict):
                assert "modality" in entry, f"alias '{name}' missing modality"
                assert entry["modality"] in (
                    "text",
                    "image",
                    "video",
                    "audio",
                    "text-diffusion",
                    "embedding",
                ), f"alias '{name}' has invalid modality: {entry['modality']}"
            else:
                raise AssertionError(
                    f"alias '{name}' is still string form, expected dict"
                )

    def test_diffusion_gemma_is_text_diffusion(self):
        import json
        from pathlib import Path

        aliases_path = (
            Path(__file__).parent.parent.parent / "fusion_mlx" / "aliases.json"
        )
        data = json.loads(aliases_path.read_text())
        entry = data.get("diffusion-gemma-26b-4bit")
        assert entry is not None
        assert isinstance(entry, dict)
        assert entry["modality"] == "text-diffusion"
