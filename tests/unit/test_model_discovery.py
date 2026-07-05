import json

import pytest

from fusion_mlx.pool.model_discovery import (
    DiscoveredModel,
    _is_adapter_dir,
    _is_unsupported_model,
    _read_model_context_length,
    _resolve_hf_cache_entry,
    detect_model_type,
    discover_models,
    discover_models_from_dirs,
    estimate_model_size,
    format_size,
    model_directory_access_error,
)


class TestDetectModelType:
    def test_detect_llm_basic(self, tmp_path):
        model_dir = tmp_path / "test-llm"
        model_dir.mkdir()
        config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
        (model_dir / "config.json").write_text(json.dumps(config))
        result = detect_model_type(model_dir)
        assert result == "llm"

    def test_detect_vlm_with_vision(self, tmp_path):
        model_dir = tmp_path / "test-vlm"
        model_dir.mkdir()
        config = {
            "model_type": "llava",
            "architectures": ["LlavaForConditionalGeneration"],
            "vision_config": {},
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        result = detect_model_type(model_dir)
        assert result == "vlm"

    def test_detect_embedding_model(self, tmp_path):
        model_dir = tmp_path / "test-emb"
        model_dir.mkdir()
        config = {"model_type": "bert", "architectures": ["BertModel"]}
        (model_dir / "config.json").write_text(json.dumps(config))
        result = detect_model_type(model_dir)
        assert result == "embedding"

    def test_detect_reranker_model(self, tmp_path):
        model_dir = tmp_path / "test-rerank"
        model_dir.mkdir()
        config = {
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaForSequenceClassification"],
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        result = detect_model_type(model_dir)
        assert result == "reranker"

    def test_detect_audio_stt_model(self, tmp_path):
        model_dir = tmp_path / "test-audio"
        model_dir.mkdir()
        config = {
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        result = detect_model_type(model_dir)
        assert result in ("audio_stt", "audio_tts", "audio_sts", "llm")

    def test_unknown_model_type_defaults_to_llm(self, tmp_path):
        model_dir = tmp_path / "test-unknown"
        model_dir.mkdir()
        config = {"model_type": "unknown", "architectures": ["UnknownModel"]}
        (model_dir / "config.json").write_text(json.dumps(config))
        result = detect_model_type(model_dir)
        assert result == "llm"

    def test_no_config_json(self, tmp_path):
        model_dir = tmp_path / "no-config"
        model_dir.mkdir()
        result = detect_model_type(model_dir)
        assert result == "llm"


class TestEstimateModelSize:
    def test_estimate_with_safetensors(self, tmp_path):
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
        result = estimate_model_size(model_dir)
        assert result >= 1024

    def test_estimate_empty_directory_raises(self, tmp_path):
        model_dir = tmp_path / "empty"
        model_dir.mkdir()
        with pytest.raises(ValueError, match="No model weights"):
            estimate_model_size(model_dir)


class TestFormatSize:
    def test_format_bytes(self):
        result = format_size(512)
        assert "512" in result
        assert "B" in result

    def test_format_kilobytes(self):
        result = format_size(1024)
        assert "K" in result or "k" in result

    def test_format_megabytes(self):
        result = format_size(1024**2)
        assert "M" in result or "m" in result

    def test_format_gigabytes(self):
        result = format_size(1024**3)
        assert "G" in result or "g" in result

    def test_format_zero(self):
        result = format_size(0)
        assert "0" in result


class TestAdapterDetection:
    def test_is_adapter_dir_true(self, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        result = _is_adapter_dir(adapter_dir)
        assert result is True

    def test_is_adapter_dir_false(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        result = _is_adapter_dir(model_dir)
        assert result is False

    def test_is_adapter_dir_not_directory(self, tmp_path):
        result = _is_adapter_dir(tmp_path / "nonexistent")
        assert result is False


class TestDiscoveredModel:
    def test_discovered_model_creation(self):
        model = DiscoveredModel(
            model_id="test-model",
            model_path="/models/test-model",
            model_type="llm",
            engine_type="batched",
            estimated_size=1024**3,
        )
        assert model.model_id == "test-model"
        assert model.model_path == "/models/test-model"
        assert model.model_type == "llm"
        assert model.engine_type == "batched"
        assert model.estimated_size == 1024**3

    def test_discovered_model_with_context_length(self):
        model = DiscoveredModel(
            model_id="test-model",
            model_path="/models/test-model",
            model_type="llm",
            engine_type="batched",
            estimated_size=1024**3,
            model_context_length=4096,
        )
        assert model.model_context_length == 4096

    def test_discovered_model_defaults(self):
        model = DiscoveredModel(
            model_id="test-model",
            model_path="/models/test-model",
            model_type="llm",
            engine_type="batched",
            estimated_size=0,
        )
        assert model.model_context_length is None
        assert model.source_type == "local"
        assert model.source_repo_id is None


class TestReadModelContextLength:
    def test_read_from_config(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config = {"max_position_embeddings": 8192}
        (model_dir / "config.json").write_text(json.dumps(config))
        result = _read_model_context_length(model_dir)
        assert result == 8192

    def test_read_from_text_config(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config = {"text_config": {"max_position_embeddings": 4096}}
        (model_dir / "config.json").write_text(json.dumps(config))
        result = _read_model_context_length(model_dir)
        assert result == 4096

    def test_no_context_length(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config = {}
        (model_dir / "config.json").write_text(json.dumps(config))
        result = _read_model_context_length(model_dir)
        assert result is None


class TestTwoLevelDiscovery:
    def test_empty_directory(self, tmp_path):
        models = discover_models_from_dirs([tmp_path])
        assert models == {}

    def test_two_level_org_structure(self, tmp_path):
        org_dir = tmp_path / "org"
        org_dir.mkdir()
        model_dir = org_dir / "model"
        model_dir.mkdir()
        config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
        (model_dir / "config.json").write_text(json.dumps(config))
        models = discover_models_from_dirs([tmp_path])
        assert isinstance(models, dict)


class TestDiscoverModels:
    def test_discover_models_returns_dict(self, tmp_path):
        result = discover_models(tmp_path)
        assert isinstance(result, dict)


class TestUnsupportedModels:
    def test_is_unsupported_model_false_for_normal(self, tmp_path):
        model_dir = tmp_path / "normal-model"
        model_dir.mkdir()
        result = _is_unsupported_model(model_dir)
        assert result is False


class TestHfCacheDiscovery:
    def test_resolve_hf_cache_entry_valid(self, tmp_path):
        cache_path = tmp_path / "models--org--model"
        cache_path.mkdir()
        snapshots_dir = cache_path / "snapshots"
        snapshots_dir.mkdir()
        snapshot = snapshots_dir / "abc123"
        snapshot.mkdir()
        result = _resolve_hf_cache_entry(cache_path)
        assert result is not None

    def test_resolve_hf_cache_entry_wrong_format(self, tmp_path):
        cache_path = tmp_path / "invalid-name"
        cache_path.mkdir()
        result = _resolve_hf_cache_entry(cache_path)
        assert result is None


class TestModelDirectoryAccessError:
    def test_error_on_nonexistent(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        error = model_directory_access_error(nonexistent)
        assert error is not None

    def test_no_error_on_existing(self, tmp_path):
        error = model_directory_access_error(tmp_path)
        assert error is None
