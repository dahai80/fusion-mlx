# SPDX-License-Identifier: Apache-2.0
# Tests for model_registry.list_available_models (issue #172).
# Filesystem + backend resolution are monkeypatched so no real model dir
# scan or backend construction runs.
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fusion_mlx import model_registry


def _fake_info(model_type: str, engine_type: str, size: int = 1024) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=f"/fake/{model_type}-model",
        model_type=model_type,
        engine_type=engine_type,
        estimated_size=size,
        config_model_type="",
        thinking_default=False,
        preserve_thinking_default=False,
        model_context_length=0,
        source_type="local",
        source_repo_id=None,
    )


def _fake_discovered(monkeypatch, mapping: dict[str, SimpleNamespace]) -> None:
    monkeypatch.setattr(
        "fusion_mlx.pool.model_discovery.discover_models_from_dirs",
        lambda dirs: dict(mapping),
    )


def _fake_video_backend(monkeypatch, *, name: str, supports_i2v: bool) -> None:
    from fusion_mlx.engines.video_backends import VideoConstraints

    fake_backend = SimpleNamespace(name=name, supports_i2v=supports_i2v)
    fake_constraints = VideoConstraints(
        supports_i2v=supports_i2v,
        max_n=4,
        dim_divisibility=16,
        num_frames_validator=None,
        num_frames_hint="1 + 8k",
        dim_hint="w/h % 16 == 0",
    )
    monkeypatch.setattr(
        "fusion_mlx.engines.video_backends.resolve_backend",
        lambda model_name, **kw: fake_backend,
    )
    monkeypatch.setattr(
        "fusion_mlx.engines.video_backends.constraints_for",
        lambda model_name, **kw: fake_constraints,
    )


def test_invalid_model_type_raises(monkeypatch):
    _fake_discovered(monkeypatch, {})
    with pytest.raises(ValueError):
        model_registry.list_available_models("audio")


def test_image_filter_returns_only_image(monkeypatch):
    _fake_discovered(
        monkeypatch,
        {
            "flux-2-klein-9b": _fake_info("image", "image_gen"),
            "wan2.2-5b": _fake_info("video", "video_gen"),
            "qwen-3b": _fake_info("llm", "batched"),
        },
    )
    models = model_registry.list_available_models("image")
    assert len(models) == 1
    entry = models[0]
    assert entry["name"] == "flux-2-klein-9b"
    assert entry["type"] == "image"
    assert entry["backend"] == "mflux"
    assert entry["quantize"] == [None, 4, 8]
    assert entry["constraints"]["dim_divisibility"] == 16
    assert entry["constraints"]["max_resolution"] == 2048
    assert entry["path"] == "/fake/image-model"
    assert entry["size_bytes"] == 1024


def test_video_filter_returns_video_with_capabilities(monkeypatch):
    _fake_discovered(
        monkeypatch,
        {
            "flux-2-klein-9b": _fake_info("image", "image_gen"),
            "wan2.2-5b": _fake_info("video", "video_gen"),
        },
    )
    _fake_video_backend(monkeypatch, name="wan2", supports_i2v=True)
    models = model_registry.list_available_models("video")
    assert len(models) == 1
    entry = models[0]
    assert entry["name"] == "wan2.2-5b"
    assert entry["type"] == "video"
    assert entry["backend"] == "wan2"
    assert entry["supports_i2v"] is True
    assert entry["constraints"]["dim_divisibility"] == 16
    assert entry["constraints"]["max_n"] == 4
    assert entry["constraints"]["frame_pattern"] == "1 + 8k"
    assert entry["constraints"]["dim_hint"] == "w/h % 16 == 0"


def test_no_filter_returns_all(monkeypatch):
    _fake_discovered(
        monkeypatch,
        {
            "flux-2-klein-9b": _fake_info("image", "image_gen"),
            "wan2.2-5b": _fake_info("video", "video_gen"),
            "qwen-3b": _fake_info("llm", "batched"),
        },
    )
    _fake_video_backend(monkeypatch, name="wan2", supports_i2v=False)
    models = model_registry.list_available_models()
    assert len(models) == 3
    types = {m["type"] for m in models}
    assert types == {"image", "video", "llm"}


def test_results_sorted_by_name(monkeypatch):
    _fake_discovered(
        monkeypatch,
        {
            "zeta-model": _fake_info("image", "image_gen"),
            "alpha-model": _fake_info("image", "image_gen"),
            "mid-model": _fake_info("image", "image_gen"),
        },
    )
    models = model_registry.list_available_models("image")
    names = [m["name"] for m in models]
    assert names == ["alpha-model", "mid-model", "zeta-model"]


def test_default_model_dirs_env_override(monkeypatch, tmp_path):
    # FUSION_MLX_MODELS should override the default ~/.fusion-mlx/models.
    monkeypatch.setenv("FUSION_MLX_MODELS", str(tmp_path))
    captured: dict[str, object] = {}

    def fake_discover(dirs):
        captured["dirs"] = list(dirs)
        return {}

    monkeypatch.setattr(
        "fusion_mlx.pool.model_discovery.discover_models_from_dirs", fake_discover
    )
    model_registry.list_available_models()
    assert captured["dirs"] == [tmp_path]


def test_video_backend_resolve_failure_is_safe(monkeypatch):
    _fake_discovered(
        monkeypatch,
        {"bad-video": _fake_info("video", "video_gen")},
    )

    def boom(model_name, **kw):
        raise RuntimeError("backend explode")

    monkeypatch.setattr("fusion_mlx.engines.video_backends.resolve_backend", boom)
    monkeypatch.setattr("fusion_mlx.engines.video_backends.constraints_for", boom)
    models = model_registry.list_available_models("video")
    assert len(models) == 1
    entry = models[0]
    assert entry["backend"] == "unknown"
    assert entry["supports_i2v"] is False
    assert entry["constraints"] == {}
