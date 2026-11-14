# SPDX-License-Identifier: Apache-2.0
"""Token-level embeddings on /v1/embeddings (#1624, late chunking).

Covers: MLXEmbeddingModel native token-state extraction with
attention-mask trimming, fail-visible behavior for models without
last_hidden_state, EmbeddingEngine aggregation, and the opt-in
return_token_embeddings wire field on the route.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import mlx.core as mx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.engines.embedding import EmbeddingOutput, MLXEmbeddingModel


def _native_model_with_mock(processor_fn, model_outputs):
    model = MLXEmbeddingModel("test-embed-model")
    model._loaded = True
    model._using_native = True
    model._hidden_size = 4
    model._processor = processor_fn
    model._model = MagicMock(return_value=model_outputs)
    return model


def _encoded_batch():
    return {
        "input_ids": np.array([[5, 6, 7, 0], [8, 9, 0, 0]], dtype=np.int32),
        "attention_mask": np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.int32),
        "offset_mapping": np.array(
            [[[0, 1], [1, 2], [2, 3], [0, 0]], [[0, 1], [1, 2], [0, 0], [0, 0]]],
            dtype=np.int32,
        ),
    }


class TestNativeTokenEmbeddings:
    def test_token_states_trimmed_by_attention_mask(self):
        encoded = _encoded_batch()
        hidden = mx.array(
            [
                [[0.1, 0.1, 0.1, 0.1]] * 4,
                [[0.2, 0.2, 0.2, 0.2]] * 4,
            ]
        )
        calls = []

        def processor(*args, **kwargs):
            calls.append(kwargs)
            return encoded

        model = _native_model_with_mock(
            processor, SimpleNamespace(last_hidden_state=hidden)
        )
        out = model.embed(["ab", "cd"], return_token_embeddings=True)

        assert out.token_embeddings is not None
        assert len(out.token_embeddings) == 2
        assert len(out.token_embeddings[0]) == 3
        assert len(out.token_embeddings[1]) == 2
        assert out.token_embeddings[0][0] == pytest.approx([0.1, 0.1, 0.1, 0.1])
        assert out.token_ids == [[5, 6, 7], [8, 9]]
        assert out.token_offsets == [
            [[0, 1], [1, 2], [2, 3]],
            [[0, 1], [1, 2]],
        ]
        # Pooled path still returned alongside
        assert len(out.embeddings) == 2
        assert len(out.embeddings[0]) == 4

    def test_flag_off_returns_no_token_fields(self):
        encoded = {
            "input_ids": np.array([[5, 6, 0]], dtype=np.int32),
            "attention_mask": np.array([[1, 1, 0]], dtype=np.int32),
        }
        hidden = mx.array([[[0.1, 0.1, 0.1, 0.1]] * 3])

        def processor(*args, **kwargs):
            return encoded

        model = _native_model_with_mock(
            processor, SimpleNamespace(last_hidden_state=hidden)
        )
        out = model.embed(["ab"])

        assert out.token_embeddings is None
        assert out.token_ids is None
        assert out.token_offsets is None
        # plain-function processor was invoked once for the batch
        assert out.embeddings == [pytest.approx([0.1, 0.1, 0.1, 0.1])]

    def test_missing_last_hidden_state_raises(self):
        encoded = {
            "input_ids": np.array([[5, 6, 0]], dtype=np.int32),
            "attention_mask": np.array([[1, 1, 0]], dtype=np.int32),
        }

        def processor(*args, **kwargs):
            return encoded

        model = _native_model_with_mock(
            processor,
            SimpleNamespace(
                text_embeds=mx.array([[0.1, 0.1, 0.1, 0.1]]),
                last_hidden_state=None,
            ),
        )
        with pytest.raises(ValueError, match="token-level"):
            model.embed(["ab"], return_token_embeddings=True)

    def test_offset_mapping_failure_degrades_to_none(self):
        encoded = {
            "input_ids": np.array([[5, 6, 0]], dtype=np.int32),
            "attention_mask": np.array([[1, 1, 0]], dtype=np.int32),
        }
        hidden = mx.array([[[0.1, 0.1, 0.1, 0.1]] * 3])

        def processor_fail(*args, **kwargs):
            if kwargs.get("return_offsets_mapping"):
                raise TypeError("no offset mapping support")
            return encoded

        model = _native_model_with_mock(
            processor_fail, SimpleNamespace(last_hidden_state=hidden)
        )
        out = model.embed(["ab"], return_token_embeddings=True)

        assert out.token_embeddings is not None
        assert out.token_offsets is None
        assert out.token_ids == [[5, 6]]


class TestEngineAggregation:
    @pytest.mark.asyncio
    async def test_engine_merges_token_fields_across_batches(self):
        from fusion_mlx.engines.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-embed-model", batch_size=1)
        engine._model = MagicMock()
        engine._model.embed.side_effect = [
            EmbeddingOutput(
                embeddings=[[0.1, 0.2]],
                total_tokens=2,
                dimensions=2,
                token_embeddings=[[[0.1, 0.2], [0.3, 0.4]]],
                token_ids=[[5, 6]],
                token_offsets=None,
            ),
            EmbeddingOutput(
                embeddings=[[0.3, 0.4]],
                total_tokens=1,
                dimensions=2,
                token_embeddings=[[[0.5, 0.6]]],
                token_ids=[[7]],
                token_offsets=None,
            ),
        ]
        out = await engine.embed(["a", "b"], return_token_embeddings=True)

        assert out.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert out.token_embeddings == [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6]],
        ]
        assert out.token_ids == [[5, 6], [7]]
        assert out.token_offsets is None
        assert engine._model.embed.call_args.kwargs["return_token_embeddings"] is True


class TestRouteWireField:
    def _client(self, monkeypatch, engine):
        from fusion_mlx.api import embeddings_routes
        from fusion_mlx.middleware.auth import check_rate_limit, verify_api_key

        app = FastAPI()
        app.include_router(embeddings_routes.router)
        app.dependency_overrides[verify_api_key] = lambda: True
        app.dependency_overrides[check_rate_limit] = lambda: True

        async def _fake_engine(model_id):
            return engine

        monkeypatch.setattr(embeddings_routes, "get_embedding_engine", _fake_engine)
        return TestClient(app)

    def test_route_returns_token_fields_when_opted_in(self, monkeypatch):
        engine = MagicMock()
        engine.embed = AsyncMock(
            return_value=EmbeddingOutput(
                embeddings=[[0.1, 0.2]],
                total_tokens=2,
                dimensions=2,
                token_embeddings=[[[0.1, 0.2], [0.3, 0.4]]],
                token_ids=[[5, 6]],
                token_offsets=[[[0, 1], [1, 2]]],
            )
        )
        client = self._client(monkeypatch, engine)
        r = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-m3",
                "input": "hello world",
                "return_token_embeddings": True,
            },
        )
        assert r.status_code == 200, r.text
        item = r.json()["data"][0]
        assert item["token_embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
        assert item["token_ids"] == [5, 6]
        assert item["token_offsets"] == [[0, 1], [1, 2]]
        assert len(item["embedding"]) == 2
        assert engine.embed.call_args.kwargs["return_token_embeddings"] is True

    def test_route_omits_token_fields_by_default(self, monkeypatch):
        engine = MagicMock()
        engine.embed = AsyncMock(
            return_value=EmbeddingOutput(
                embeddings=[[0.1, 0.2]],
                total_tokens=2,
                dimensions=2,
            )
        )
        client = self._client(monkeypatch, engine)
        r = client.post(
            "/v1/embeddings",
            json={"model": "bge-m3", "input": "hello"},
        )
        assert r.status_code == 200, r.text
        item = r.json()["data"][0]
        assert item["token_embeddings"] is None
        assert item["token_ids"] is None
        assert engine.embed.call_args.kwargs["return_token_embeddings"] is False

    def test_route_dedup_maps_token_fields(self, monkeypatch):
        engine = MagicMock()
        engine.embed = AsyncMock(
            return_value=EmbeddingOutput(
                embeddings=[[0.1, 0.2]],
                total_tokens=2,
                dimensions=2,
                token_embeddings=[[[0.1, 0.2], [0.3, 0.4]]],
                token_ids=[[5, 6]],
                token_offsets=None,
            )
        )
        client = self._client(monkeypatch, engine)
        r = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-m3",
                "input": ["same text", "same text"],
                "return_token_embeddings": True,
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert len(data) == 2
        assert data[0]["token_ids"] == [5, 6]
        assert data[1]["token_ids"] == [5, 6]

    def test_route_rejects_unsupported_model_400(self, monkeypatch):
        engine = MagicMock()
        engine.embed = AsyncMock(side_effect=ValueError("token-level"))
        client = self._client(monkeypatch, engine)
        r = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-m3",
                "input": "hello",
                "return_token_embeddings": True,
            },
        )
        assert r.status_code == 400
        assert "token-level" in r.text
