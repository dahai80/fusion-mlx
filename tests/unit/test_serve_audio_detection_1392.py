# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from fusion_mlx._cli_base import (
    _detect_local_audio_entry,
    _resolve_audio_model_for_serve,
)


def _make_cache_model(root, repo_id, config, commit="a" * 40):
    cache_dir = root / ("models--" + repo_id.replace("/", "--"))
    snapshot = cache_dir / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (cache_dir / "refs").mkdir(exist_ok=True)
    (cache_dir / "refs" / "main").write_text(commit)
    (snapshot / "config.json").write_text(json.dumps(config))
    return snapshot


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("FUSION_MLX_MODELS", str(root))
    return root


def test_registry_entry_still_wins():
    entry = _resolve_audio_model_for_serve("kokoro")

    assert entry is not None
    assert entry.alias == "kokoro"
    assert entry.hf_id == "mlx-community/Kokoro-82M-bf16"


def test_unregistered_tts_checkpoint_on_disk_routes_to_audio_mode(cache_root):
    repo_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    _make_cache_model(cache_root, repo_id, {"model_type": "qwen3_tts"})

    entry = _resolve_audio_model_for_serve(repo_id)

    assert entry is not None
    assert entry.type == "tts"
    assert entry.hf_id == repo_id
    assert entry.family == "qwen3_tts"


def test_unregistered_stt_checkpoint_on_disk_routes_to_audio_mode(cache_root):
    repo_id = "mlx-community/whisper-large-v3-turbo-4bit"
    _make_cache_model(cache_root, repo_id, {"model_type": "whisper"})

    entry = _resolve_audio_model_for_serve(repo_id)

    assert entry is not None
    assert entry.type == "stt"


def test_local_directory_path_is_detected(tmp_path):
    model_dir = tmp_path / "my-tts"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_tts"}))

    entry = _resolve_audio_model_for_serve(str(model_dir))

    assert entry is not None
    assert entry.type == "tts"


def test_text_checkpoint_stays_on_text_path(cache_root):
    repo_id = "mlx-community/Qwen3.8-27B-4bit"
    _make_cache_model(cache_root, repo_id, {"model_type": "qwen3"})

    assert _resolve_audio_model_for_serve(repo_id) is None


def test_downloaded_nowhere_returns_none(cache_root):
    assert _resolve_audio_model_for_serve("mlx-community/Nope-4bit") is None
    assert _detect_local_audio_entry("Nope") is None
    assert _detect_local_audio_entry("") is None
    assert _detect_local_audio_entry(None) is None
