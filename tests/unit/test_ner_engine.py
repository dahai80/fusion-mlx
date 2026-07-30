# SPDX-License-Identifier: Apache-2.0
"""Unit tests for NER engine and models."""

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.api.ner_models import NEREntity, NERRequest, NERResponse, NERUsage


class TestNERModels:
    def test_ner_request_single_text(self):
        req = NERRequest(
            text="Apple is based in Cupertino",
            labels=["company", "location"],
            model="gliner-test",
        )
        assert req.text == "Apple is based in Cupertino"
        assert req.labels == ["company", "location"]
        assert req.model == "gliner-test"
        assert req.threshold == 0.5
        assert req.flat_ner is True
        assert req.multi_label is False

    def test_ner_request_list_text(self):
        req = NERRequest(text=["text1", "text2"], labels=["org"], model="gliner-test")
        assert isinstance(req.text, list)
        assert len(req.text) == 2

    def test_ner_request_custom_threshold(self):
        req = NERRequest(
            text="hello",
            labels=["per"],
            model="m",
            threshold=0.3,
            flat_ner=False,
            multi_label=True,
        )
        assert req.threshold == 0.3
        assert req.flat_ner is False
        assert req.multi_label is True

    def test_ner_entity(self):
        e = NEREntity(start=0, end=5, text="Apple", label="company", score=0.95)
        assert e.start == 0
        assert e.end == 5
        assert e.text == "Apple"
        assert e.label == "company"
        assert e.score == 0.95

    def test_ner_response(self):
        resp = NERResponse(
            data=[[NEREntity(start=0, end=5, text="Apple", label="org", score=0.9)]],
            model="gliner-test",
            usage=NERUsage(prompt_tokens=10, total_tokens=10),
        )
        assert resp.object == "list"
        assert len(resp.data) == 1
        assert len(resp.data[0]) == 1
        assert resp.data[0][0].label == "org"
        assert resp.usage.prompt_tokens == 10


class TestMLXNERModel:
    def test_ner_architectures_frozenset(self):
        from fusion_mlx.engines.ner import MLXNERModel

        assert isinstance(MLXNERModel._NER_ARCHITECTURES, frozenset)
        assert "GLiNERModel" in MLXNERModel._NER_ARCHITECTURES
        assert "SpaModel" in MLXNERModel._NER_ARCHITECTURES

    def test_validate_architecture_gliner(self, tmp_path):
        from fusion_mlx.engines.ner import MLXNERModel

        config = tmp_path / "config.json"
        config.write_text(json.dumps({"architectures": ["GLiNERModel"]}))
        model = MLXNERModel(str(tmp_path))
        model._validate_architecture()

    def test_validate_architecture_spamodel(self, tmp_path):
        from fusion_mlx.engines.ner import MLXNERModel

        config = tmp_path / "config.json"
        config.write_text(json.dumps({"architectures": ["SpaModel"]}))
        model = MLXNERModel(str(tmp_path))
        model._validate_architecture()

    def test_validate_architecture_unknown(self, tmp_path):
        from fusion_mlx.engines.ner import MLXNERModel

        config = tmp_path / "config.json"
        config.write_text(json.dumps({"architectures": ["SomeOtherModel"]}))
        model = MLXNERModel(str(tmp_path))
        with pytest.raises(ValueError, match="not a supported NER model"):
            model._validate_architecture()

    def test_validate_architecture_empty(self, tmp_path):
        from fusion_mlx.engines.ner import MLXNERModel

        config = tmp_path / "config.json"
        config.write_text(json.dumps({"architectures": []}))
        model = MLXNERModel(str(tmp_path))
        model._validate_architecture()

    def test_validate_architecture_no_config(self, tmp_path):
        from fusion_mlx.engines.ner import MLXNERModel

        model = MLXNERModel(str(tmp_path))
        model._validate_architecture()

    def test_predict_entities_not_loaded(self):
        from fusion_mlx.engines.ner import MLXNERModel

        model = MLXNERModel("fake-model")
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict_entities("text", ["org"])

    def test_batch_predict_entities_not_loaded(self):
        from fusion_mlx.engines.ner import MLXNERModel

        model = MLXNERModel("fake-model")
        with pytest.raises(RuntimeError, match="not loaded"):
            model.batch_predict_entities(["text"], ["org"])

    def test_ner_output_dataclass(self):
        from fusion_mlx.engines.ner import NEROutput

        output = NEROutput(
            entities=[
                [{"start": 0, "end": 5, "text": "Apple", "label": "org", "score": 0.9}]
            ],
            total_tokens=10,
        )
        assert len(output.entities) == 1
        assert output.total_tokens == 10


class TestNEREngine:
    def test_ner_engine_init(self):
        from fusion_mlx.engines.ner import NEREngine

        engine = NEREngine(model_name="test-ner", trust_remote_code=False)
        assert engine.model_name == "test-ner"
        assert engine._model is None

    async def test_ner_single_text(self):
        from fusion_mlx.engines.ner import NEREngine, NEROutput

        engine = NEREngine(model_name="test-ner", trust_remote_code=False)
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"start": 0, "end": 5, "text": "Apple", "label": "org", "score": 0.92}
        ]
        mock_model.loaded = True
        engine._model = mock_model

        executor = ThreadPoolExecutor(max_workers=1)
        with patch("fusion_mlx.engine_core.get_executor", return_value=executor):
            result = await engine.ner(texts=["Apple is great"], labels=["org"])
        executor.shutdown(wait=False)
        assert isinstance(result, NEROutput)
        assert len(result.entities) == 1
        assert result.entities[0][0]["text"] == "Apple"
        assert result.total_tokens > 0

    async def test_ner_batch_text(self):
        from fusion_mlx.engines.ner import NEREngine, NEROutput

        engine = NEREngine(model_name="test-ner", trust_remote_code=False)
        mock_model = MagicMock()
        mock_model.batch_predict_entities.return_value = [
            [{"start": 0, "end": 5, "text": "Apple", "label": "org", "score": 0.9}],
            [{"start": 0, "end": 6, "text": "Google", "label": "org", "score": 0.88}],
        ]
        mock_model.loaded = True
        engine._model = mock_model

        executor = ThreadPoolExecutor(max_workers=1)
        with patch("fusion_mlx.engine_core.get_executor", return_value=executor):
            result = await engine.ner(texts=["Apple HQ", "Google HQ"], labels=["org"])
        executor.shutdown(wait=False)
        assert isinstance(result, NEROutput)
        assert len(result.entities) == 2

    async def test_ner_empty_texts(self):
        from fusion_mlx.engines.ner import NEREngine, NEROutput

        engine = NEREngine(model_name="test-ner", trust_remote_code=False)
        result = await engine.ner(texts=[], labels=["org"])
        assert isinstance(result, NEROutput)
        assert result.entities == []
        assert result.total_tokens == 0

    def test_ner_stats(self):
        from fusion_mlx.engines.ner import NEREngine

        engine = NEREngine(model_name="test-ner", trust_remote_code=False)
        stats = engine.get_stats()
        assert stats["model_name"] == "test-ner"
        assert stats["loaded"] is False

    async def test_ner_not_started_raises(self):
        from fusion_mlx.engines.ner import NEREngine

        engine = NEREngine(model_name="test-ner", trust_remote_code=False)
        with pytest.raises(RuntimeError, match="Engine not started"):
            await engine.ner(texts=["text"], labels=["org"])

    async def test_ner_empty_entities(self):
        from fusion_mlx.engines.ner import NEREngine, NEROutput

        engine = NEREngine(model_name="test-ner", trust_remote_code=False)
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = []
        mock_model.loaded = True
        engine._model = mock_model

        executor = ThreadPoolExecutor(max_workers=1)
        with patch("fusion_mlx.engine_core.get_executor", return_value=executor):
            result = await engine.ner(texts=["no entities here"], labels=["org"])
        executor.shutdown(wait=False)
        assert isinstance(result, NEROutput)
        assert result.entities == [[]]
