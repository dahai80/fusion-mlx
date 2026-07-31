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
