# SPDX-License-Identifier: Apache-2.0
"""Tests for image model discovery (Flux / diffusers).

Verifies that Flux-style image-generation models are detected during model
discovery and routed to the ``image_gen`` engine type, instead of being
silently skipped (no top-level ``config.json``) or misclassified as ``llm``.

All tests run without mlx-examples / Flux installed — only manifest parsing
and discovery wiring are tested.
"""

import json
from pathlib import Path

from fusion_mlx.pool.engine_pool import EnginePool
from fusion_mlx.pool.model_discovery import (
    _is_image_model,
    _is_model_dir,
    detect_model_type,
    discover_models,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_image_manifest(path: Path, task: str = "text-to-image") -> None:
    (path / "configuration.json").write_text(json.dumps({"task": task}))


def _make_image_model(path: Path, task: str = "text-to-image") -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_image_manifest(path, task)
    (path / "transformer").mkdir(exist_ok=True)
    (path / "vae").mkdir(exist_ok=True)
    (path / "transformer" / "model.safetensors").write_bytes(b"0" * 1000)


def _write_model_index(path: Path, class_name: str) -> None:
    # Minimal diffusers model_index.json - raw diffusers image/video models
    # ship this instead of a configuration.json task manifest.
    (path / "model_index.json").write_text(
        json.dumps({"_class_name": class_name, "_diffusers_version": "0.37.0"})
    )


def _make_diffusers_image_model(
    path: Path, class_name: str = "Flux2KleinPipeline"
) -> None:
    # Raw diffusers image model: model_index.json + subdirs, NO configuration.json.
    path.mkdir(parents=True, exist_ok=True)
    _write_model_index(path, class_name)
    (path / "transformer").mkdir(exist_ok=True)
    (path / "vae").mkdir(exist_ok=True)
    (path / "transformer" / "model.safetensors").write_bytes(b"0" * 1000)


def _make_llm_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (path / "model.safetensors").write_bytes(b"0" * 1000)


# ---------------------------------------------------------------------------
# TestIsImageModel
# ---------------------------------------------------------------------------


class TestIsImageModel:
    """Tests for the _is_image_model helper."""

    def test_text_to_image_manifest_returns_true(self, tmp_path):
        _make_image_model(tmp_path)
        assert _is_image_model(tmp_path) is True

    def test_no_manifest_returns_false(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        assert _is_image_model(tmp_path) is False

    def test_non_image_task_returns_false(self, tmp_path):
        _make_image_model(tmp_path, task="text-to-speech")
        assert _is_image_model(tmp_path) is False

    def test_corrupt_manifest_returns_false(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "configuration.json").write_text("{not json")
        assert _is_image_model(tmp_path) is False

    def test_model_index_json_class_name_returns_true(self, tmp_path):
        # 方案B: raw diffusers model with model_index.json (no configuration.json)
        # must be detected as image via _class_name.
        _make_diffusers_image_model(tmp_path, "Flux2KleinPipeline")
        assert not (tmp_path / "configuration.json").exists()
        assert _is_image_model(tmp_path) is True

    def test_model_index_json_flux1_returns_true(self, tmp_path):
        _make_diffusers_image_model(tmp_path, "FluxPipeline")
        assert _is_image_model(tmp_path) is True

    def test_model_index_json_video_class_returns_false(self, tmp_path):
        # A video pipeline _class_name must not satisfy _is_image_model.
        _make_diffusers_image_model(tmp_path, "LTXVideoPipeline")
        assert _is_image_model(tmp_path) is False

    def test_model_index_json_unknown_class_returns_false(self, tmp_path):
        _make_diffusers_image_model(tmp_path, "SomeUnknownPipeline")
        assert _is_image_model(tmp_path) is False

    def test_model_index_json_missing_class_name_returns_false(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_diffusers_version": "0.37.0"})
        )
        assert _is_image_model(tmp_path) is False

    def test_corrupt_model_index_returns_false(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "model_index.json").write_text("{not json")
        assert _is_image_model(tmp_path) is False

    def test_configuration_json_takes_priority_over_model_index(self, tmp_path):
        # configuration.json task must win when both manifests exist.
        tmp_path.mkdir(exist_ok=True)
        _write_image_manifest(tmp_path, task="text-to-image")
        _write_model_index(tmp_path, "LTXVideoPipeline")
        assert _is_image_model(tmp_path) is True


# ---------------------------------------------------------------------------
# TestDetectImageModelType
# ---------------------------------------------------------------------------


class TestDetectImageModelType:
    """Tests that image models are detected as type 'image'."""

    def test_flux_dir_without_config_json_returns_image(self, tmp_path):
        _make_image_model(tmp_path)
        assert detect_model_type(tmp_path) == "image"

    def test_image_takes_priority_over_llm_fallback(self, tmp_path):
        _make_image_model(tmp_path)
        assert not (tmp_path / "config.json").exists()
        assert detect_model_type(tmp_path) == "image"

    def test_diffusers_model_index_only_returns_image(self, tmp_path):
        # 方案B: raw diffusers model with only model_index.json (no
        # configuration.json, no config.json) must detect as "image", not
        # fall through to "llm".
        _make_diffusers_image_model(tmp_path, "Flux2KleinPipeline")
        assert not (tmp_path / "configuration.json").exists()
        assert not (tmp_path / "config.json").exists()
        assert detect_model_type(tmp_path) == "image"

    def test_llm_dir_still_returns_llm(self, tmp_path):
        _make_llm_model(tmp_path)
        assert detect_model_type(tmp_path) == "llm"


# ---------------------------------------------------------------------------
# TestIsModelDir
# ---------------------------------------------------------------------------


class TestIsModelDir:
    """Tests that _is_model_dir accepts Flux directories."""

    def test_flux_dir_is_valid_model_dir(self, tmp_path):
        _make_image_model(tmp_path)
        assert _is_model_dir(tmp_path) is True

    def test_empty_dir_is_not_model_dir(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        assert _is_model_dir(tmp_path) is False


# ---------------------------------------------------------------------------
# TestDiscoverImageModel
# ---------------------------------------------------------------------------


class TestDiscoverImageModel:
    """End-to-end discovery: Flux model is registered as image_gen."""

    def test_flux_model_registered_with_image_gen_engine(self, tmp_path):
        _make_image_model(tmp_path / "Flux-1.lite-8B-MLX-Q4")
        models = discover_models(tmp_path)
        assert "Flux-1.lite-8B-MLX-Q4" in models
        entry = models["Flux-1.lite-8B-MLX-Q4"]
        assert entry.model_type == "image"
        assert entry.engine_type == "image_gen"

    def test_flux_and_llm_coexist_in_same_dir(self, tmp_path):
        _make_image_model(tmp_path / "Flux-1.lite-8B-MLX-Q4")
        _make_llm_model(tmp_path / "Qwen3-8B")
        models = discover_models(tmp_path)
        assert models["Flux-1.lite-8B-MLX-Q4"].model_type == "image"
        assert models["Qwen3-8B"].model_type == "llm"


# ---------------------------------------------------------------------------
# TestEnginePoolMapping
# ---------------------------------------------------------------------------


class TestEnginePoolMapping:
    """Tests that EnginePool maps image -> image_gen."""

    def test_model_type_to_engine_includes_image(self):
        assert EnginePool._MODEL_TYPE_TO_ENGINE.get("image") == "image_gen"
