# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1086.

#1086: Hunyuan3D-2.1-MLX-Serve-8bit (a 3D shape generator served by
POST /v1/3d/generate) and MuseTalk (a lip-sync model) were discovered by
the EnginePool and classified as ``llm`` because their config.json
carries no LLM architecture/model_type. /v1/models then advertised
``modality: "text"`` + ``text_generation: true``, so downstream clients
doing capability-based auto-selection (fusion-k12-teacher) routed
/v1/chat/completions traffic to them — 400 on every call.

Fix: detect_model_type classifies these specialty generative models as
``"image"`` (non-text, mirroring FLUX.1-dev) so modality is non-text and
text_generation stays false. _auto_detect_single_cached_model now only
considers llm/vlm dirs, so a lone specialty model on disk is never
auto-picked as the default chat model.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fusion_mlx._cli_base import _auto_detect_single_cached_model
from fusion_mlx.pool.model_discovery import (
    _is_specialty_generative_model,
    detect_model_type,
)


def _write_config(model_dir: Path, config: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))


class TestSpecialtyGenerativeClassification:
    """Hunyuan3D + MuseTalk classify as non-text (image), not llm."""

    def test_source_has_specialty_helper(self):
        src = inspect.getsource(detect_model_type)
        assert "_is_specialty_generative_model" in src
        assert inspect.isfunction(_is_specialty_generative_model)

    def test_hunyuan3d_classified_as_image(self, tmp_path):
        d = tmp_path / "Hunyuan3D-2.1-MLX-Serve-8bit"
        _write_config(
            d,
            {
                "model_type": "hunyuan3d_2_1",
                "quant": "8bit",
                "hidden_size": 2048,
                "depth": 21,
            },
        )
        assert detect_model_type(d) == "image"

    def test_musetalk_classified_as_image(self, tmp_path):
        d = tmp_path / "musetalk-mlx-native"
        _write_config(
            d,
            {
                "scaling_factor": 0.18215,
                "dtype": "bfloat16",
                "source": "TMElyralab/MuseTalk + stabilityai/sd-vae-ft-mse",
                "converted_by": "scripts/convert_musetalk_native.py",
            },
        )
        assert detect_model_type(d) == "image"

    def test_hunyuan3d_prefix_match(self, tmp_path):
        # any hunyuan3d* model_type prefix hits the specialty branch
        d = tmp_path / "Hunyuan3D-2.0"
        _write_config(d, {"model_type": "hunyuan3d_2"})
        assert _is_specialty_generative_model(d) is True
        assert detect_model_type(d) == "image"

    def test_plain_llm_unaffected(self, tmp_path):
        d = tmp_path / "qwen3-8b"
        _write_config(d, {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
        assert _is_specialty_generative_model(d) is False
        assert detect_model_type(d) == "llm"

    def test_no_config_json_not_specialty(self, tmp_path):
        d = tmp_path / "bare-weights"
        d.mkdir()
        (d / "weights.safetensors").write_bytes(b"\x00")
        assert _is_specialty_generative_model(d) is False

    def test_unrelated_config_not_specialty(self, tmp_path):
        d = tmp_path / "some-llm"
        _write_config(
            d, {"model_type": "gemma3", "architectures": ["Gemma3ForCausalLM"]}
        )
        assert _is_specialty_generative_model(d) is False


class TestCapabilitiesNonText:
    """An image-classified entry reports text_generation=false + non-text modality."""

    def test_openai_resolve_capabilities_image_is_non_text(self):
        from types import SimpleNamespace

        from fusion_mlx.api.openai._common import (
            _resolve_capabilities,
            _resolve_modality,
        )

        fake_pool = SimpleNamespace(
            get_entry=lambda mid: SimpleNamespace(model_type="image")
        )
        with patch("fusion_mlx.api.openai._common._pool", fake_pool):
            caps = _resolve_capabilities("Hunyuan3D-2.1-MLX-Serve-8bit")
            mod = _resolve_modality("Hunyuan3D-2.1-MLX-Serve-8bit")
        assert caps["text_generation"] is False
        assert mod == "image"

    def test_routes_internal_resolve_modality_image(self):
        from types import SimpleNamespace

        from fusion_mlx.routes_internal.models import _resolve_modality

        fake_pool = SimpleNamespace(
            get_entry=lambda mid: SimpleNamespace(model_type="image")
        )
        with patch("fusion_mlx.routes_internal.models._pool", fake_pool):
            mod = _resolve_modality("Hunyuan3D-2.1-MLX-Serve-8bit")
        assert mod == "image"


class TestAutoDetectSkipsNonText:
    """_auto_detect_single_cached_model only returns llm/vlm dirs."""

    def _models_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".fusion-mlx" / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_only_specialty_dir_returns_none(self, tmp_path, monkeypatch):
        # a lone Hunyuan3D dir must NOT be auto-picked as the chat default
        models = self._models_dir(tmp_path)
        _write_config(
            models / "Hunyuan3D-2.1-MLX-Serve-8bit",
            {"model_type": "hunyuan3d_2_1"},
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _auto_detect_single_cached_model() is None

    def test_one_llm_plus_specialty_returns_llm(self, tmp_path, monkeypatch):
        models = self._models_dir(tmp_path)
        _write_config(
            models / "Hunyuan3D-2.1-MLX-Serve-8bit",
            {"model_type": "hunyuan3d_2_1"},
        )
        _write_config(
            models / "Qwen3-8B-4bit",
            {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _auto_detect_single_cached_model() == "Qwen3-8B-4bit"

    def test_single_llm_returns_it(self, tmp_path, monkeypatch):
        models = self._models_dir(tmp_path)
        _write_config(
            models / "Qwen3-8B-4bit",
            {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _auto_detect_single_cached_model() == "Qwen3-8B-4bit"

    def test_two_llms_returns_none(self, tmp_path, monkeypatch):
        models = self._models_dir(tmp_path)
        _write_config(
            models / "Qwen3-8B-4bit",
            {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
        )
        _write_config(
            models / "Llama-3.2-1B",
            {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _auto_detect_single_cached_model() is None

    def test_non_model_dir_skipped(self, tmp_path, monkeypatch):
        # a dir with no config.json/gliner/image-video manifest is not a model
        models = self._models_dir(tmp_path)
        bare = models / "dwpose"
        bare.mkdir()
        (bare / "weights.safetensors").write_bytes(b"\x00")
        _write_config(
            models / "Qwen3-8B-4bit",
            {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _auto_detect_single_cached_model() == "Qwen3-8B-4bit"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
