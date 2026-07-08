# SPDX-License-Identifier: Apache-2.0
# Tests for video model discovery (mlx-video / LTX-2). Verifies that
# text-to-video models are detected and routed to the ``video_gen`` engine
# type. Runs without mlx-video installed - only manifest parsing + wiring.

import json

from fusion_mlx.pool.engine_pool import EnginePool
from fusion_mlx.pool.model_discovery import (
    _is_model_dir,
    _is_video_model,
    detect_model_type,
    discover_models,
)


def _write_video_manifest(path, task="text-to-video"):
    (path / "configuration.json").write_text(json.dumps({"task": task}))


def _make_video_model(path, task="text-to-video"):
    path.mkdir(parents=True, exist_ok=True)
    _write_video_manifest(path, task)
    (path / "transformer").mkdir(exist_ok=True)
    (path / "vae").mkdir(exist_ok=True)
    (path / "transformer" / "model.safetensors").write_bytes(b"0" * 1000)


def _make_llm_model(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (path / "model.safetensors").write_bytes(b"0" * 1000)


class TestIsVideoModel:
    def test_text_to_video_manifest_returns_true(self, tmp_path):
        _make_video_model(tmp_path)
        assert _is_video_model(tmp_path) is True

    def test_no_manifest_returns_false(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        assert _is_video_model(tmp_path) is False

    def test_non_video_task_returns_false(self, tmp_path):
        _make_video_model(tmp_path, task="text-to-image")
        assert _is_video_model(tmp_path) is False

    def test_corrupt_manifest_returns_false(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "configuration.json").write_text("{not json")
        assert _is_video_model(tmp_path) is False


class TestDetectVideoModelType:
    def test_video_dir_returns_video(self, tmp_path):
        _make_video_model(tmp_path)
        assert detect_model_type(tmp_path) == "video"

    def test_llm_dir_still_returns_llm(self, tmp_path):
        _make_llm_model(tmp_path)
        assert detect_model_type(tmp_path) == "llm"


class TestIsModelDir:
    def test_video_dir_is_valid_model_dir(self, tmp_path):
        _make_video_model(tmp_path)
        assert _is_model_dir(tmp_path) is True

    def test_empty_dir_is_not_model_dir(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        assert _is_model_dir(tmp_path) is False


class TestDiscoverVideoModel:
    def test_video_model_registered_with_video_gen_engine(self, tmp_path):
        _make_video_model(tmp_path / "ltx-2-mlx")
        models = discover_models(tmp_path)
        assert "ltx-2-mlx" in models
        entry = models["ltx-2-mlx"]
        assert entry.model_type == "video"
        assert entry.engine_type == "video_gen"

    def test_video_and_llm_coexist(self, tmp_path):
        _make_video_model(tmp_path / "ltx-2-mlx")
        _make_llm_model(tmp_path / "Qwen3-8B")
        models = discover_models(tmp_path)
        assert models["ltx-2-mlx"].model_type == "video"
        assert models["Qwen3-8B"].model_type == "llm"


class TestEnginePoolMapping:
    def test_model_type_to_engine_includes_video(self):
        assert EnginePool._MODEL_TYPE_TO_ENGINE.get("video") == "video_gen"
