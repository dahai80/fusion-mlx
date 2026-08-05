# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OCR API routes.

Standalone pytest, no importers. Tests fusion_mlx.api.ocr_routes.
User instruction: "完成所有剩余工作，全部未完成，defer和遗留的工作"
"""

from unittest.mock import MagicMock

import pytest

from fusion_mlx.api.ocr_routes import (
    OCRRequest,
    OCRResponse,
    OCRResult,
    OCRUsage,
    _resolve_image_url,
    set_ocr_context,
)


class TestResolveImageUrl:
    def test_data_uri_passthrough(self):
        uri = "data:image/png;base64,iVBORw0KGgo="
        assert _resolve_image_url(uri) == uri

    def test_http_url_passthrough(self):
        url = "https://example.com/image.png"
        assert _resolve_image_url(url) == url

    def test_https_url_passthrough(self):
        url = "https://example.com/image.png"
        assert _resolve_image_url(url) == url

    def test_local_file_converted(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        result = _resolve_image_url(str(img))
        assert result.startswith("data:image/png;base64,")

    def test_missing_file_raises(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _resolve_image_url("/nonexistent/path/image.png")
        assert exc_info.value.status_code == 400


class TestOCRModels:
    def test_ocr_request_defaults(self):
        req = OCRRequest(model="glm-ocr", image="data:image/png;base64,abc")
        assert req.output_format == "markdown"
        assert req.temperature is None
        assert req.max_tokens is None

    def test_ocr_response_structure(self):
        resp = OCRResponse(
            id="ocr-123",
            model="glm-ocr",
            results=[OCRResult(text="hello", format="text")],
            usage=OCRUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert resp.object == "ocr_result"
        assert len(resp.results) == 1
        assert resp.results[0].text == "hello"


class TestSetOcrContext:
    def test_set_pool(self):
        mock_pool = MagicMock()
        set_ocr_context(mock_pool)
        from fusion_mlx.api.ocr_routes import _pool

        assert _pool is mock_pool


class TestListOcrModelsRoute:
    # #359: GET /v1/ocr/models crashed with AttributeError 'EnginePool' has no
    # attribute 'engines' after the pool refactor moved engines to _entries.
    # Regression: route must use get_loaded_model_ids() + get_entry().

    def _make_engine(self, is_ocr: bool, model_id: str):
        # spec=VLMBatchedEngine so isinstance(engine, VLMBatchedEngine) in the
        # route passes (the real filter the #359 regression hinges on).
        from fusion_mlx.engines.vlm import VLMBatchedEngine

        engine = MagicMock(spec=VLMBatchedEngine)
        engine.is_ocr_model = is_ocr
        engine.model_id = model_id
        engine.model_type = "vlm_ocr"
        return engine

    def _make_pool(self, engines: list):
        # Build a mock pool honoring the public accessor contract (#359 fix).
        ids = [e.model_id for e in engines]
        entries = {e.model_id: MagicMock(engine=e) for e in engines}

        pool = MagicMock()
        pool.get_loaded_model_ids.return_value = ids
        pool.get_entry.side_effect = lambda mid: entries.get(mid)
        return pool

    def test_lists_only_ocr_engines(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from fusion_mlx.api.ocr_routes import router, set_ocr_context

        engines = [
            self._make_engine(is_ocr=True, model_id="ocr-a"),
            self._make_engine(is_ocr=False, model_id="vlm-b"),
        ]
        set_ocr_context(self._make_pool(engines))

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        r = client.get("/v1/ocr/models")
        assert r.status_code == 200
        body = r.json()
        ids = [m["id"] for m in body["models"]]
        assert ids == ["ocr-a"]
        assert body["models"][0]["capabilities"] == ["chat", "vision", "ocr"]

    def test_no_crash_on_empty_pool(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from fusion_mlx.api.ocr_routes import router, set_ocr_context

        set_ocr_context(self._make_pool([]))
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        r = client.get("/v1/ocr/models")
        assert r.status_code == 200
        assert r.json() == {"models": []}
