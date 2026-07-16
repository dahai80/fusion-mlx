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


def _make_wan2_config_only_model(path, model_type="ti2v"):
    # Wan2.2 models that ship config.json with model_type in {t2v, i2v, ti2v}
    # but NO configuration.json task manifest. Reproduces issue #95.
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (path / "transformer").mkdir(exist_ok=True)
    (path / "vae").mkdir(exist_ok=True)
    (path / "transformer" / "model.safetensors").write_bytes(b"0" * 1000)


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

    def test_stray_video_manifest_without_diffusers_subdir_returns_false(
        self, tmp_path
    ):
        # An LLM dir that happens to carry a stray text-to-video configuration.json
        # must NOT be misclassified as a video engine - real LTX-2/Wan models ship
        # vae/transformer subdirs; LLM dirs do not.
        tmp_path.mkdir(exist_ok=True)
        _write_video_manifest(tmp_path, task="text-to-video")
        assert _is_video_model(tmp_path) is False

    def test_ti2v_config_without_manifest_returns_true(self, tmp_path):
        # Wan2.2 ti2v model shipping config.json model_type="ti2v" but no
        # configuration.json task manifest (issue #95). With diffusers subdirs
        # present, it must be recognized as a video model.
        _make_wan2_config_only_model(tmp_path, model_type="ti2v")
        assert _is_video_model(tmp_path) is True

    def test_t2v_and_i2v_config_without_manifest_return_true(self, tmp_path):
        _make_wan2_config_only_model(tmp_path / "t2v", model_type="t2v")
        _make_wan2_config_only_model(tmp_path / "i2v", model_type="i2v")
        assert _is_video_model(tmp_path / "t2v") is True
        assert _is_video_model(tmp_path / "i2v") is True

    def test_ti2v_config_without_subdirs_returns_false(self, tmp_path):
        # config.json model_type="ti2v" but no vae/transformer subdirs -> not a
        # loadable video model; must not be misclassified.
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "ti2v"}))
        assert _is_video_model(tmp_path) is False


class TestDetectVideoModelType:
    def test_video_dir_returns_video(self, tmp_path):
        _make_video_model(tmp_path)
        assert detect_model_type(tmp_path) == "video"

    def test_llm_dir_still_returns_llm(self, tmp_path):
        _make_llm_model(tmp_path)
        assert detect_model_type(tmp_path) == "llm"

    def test_stray_manifest_dir_falls_through_to_llm(self, tmp_path):
        # text-to-video manifest but no diffusers subdir -> not video; with no
        # config.json either, detect_model_type falls through to "llm".
        tmp_path.mkdir(exist_ok=True)
        _write_video_manifest(tmp_path, task="text-to-video")
        assert detect_model_type(tmp_path) == "llm"

    def test_ti2v_config_only_returns_video(self, tmp_path):
        # Issue #95: a Wan2.2 ti2v model with config.json model_type="ti2v" and
        # diffusers subdirs but no task manifest must be detected as "video"
        # (not "llm", which would route to BatchedEngine and fail mlx-lm load).
        _make_wan2_config_only_model(tmp_path, model_type="ti2v")
        assert detect_model_type(tmp_path) == "video"


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

    def test_ti2v_config_only_model_registered_with_video_gen(self, tmp_path):
        # Issue #95: a ti2v model without a task manifest must still register
        # with engine_type="video_gen" so /v1/videos/generate routes to the
        # VideoGenEngine (wan2 backend), not BatchedEngine (mlx-lm).
        _make_wan2_config_only_model(tmp_path / "wan22-ti2v-5b", model_type="ti2v")
        models = discover_models(tmp_path)
        entry = models["wan22-ti2v-5b"]
        assert entry.model_type == "video"
        assert entry.engine_type == "video_gen"


class TestEnginePoolMapping:
    def test_model_type_to_engine_includes_video(self):
        assert EnginePool._MODEL_TYPE_TO_ENGINE.get("video") == "video_gen"

    def test_model_type_to_engine_includes_ti2v(self):
        # Override-path consistency (issue #95): a model_type_override="ti2v"
        # must map to "video_gen", not fall back to "batched".
        assert EnginePool._MODEL_TYPE_TO_ENGINE.get("ti2v") == "video_gen"
