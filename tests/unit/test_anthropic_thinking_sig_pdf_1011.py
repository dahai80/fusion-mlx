# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1011.

#1011: Anthropic compat layer fake data — (1) thinking blocks used a fixed
placeholder signature string "fusion-mlx-reasoning" (identical for every
response, fails strict clients that verify); (2) PDF/non-text document blocks
returned a fixed placeholder text injected into model context, making the
model answer the placeholder as if it were document content.

Fix (#1011): (1) signature is now an HMAC of the thinking content keyed by a
per-server-startup secret (self-signed, self-verifiable); (2) non-text
document blocks raise 400 with a clear "unsupported media_type" message.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from fusion_mlx.api.anthropic_models import (
    AnthropicMessage,
    ContentBlockDocument,
    MessagesRequest,
)
from fusion_mlx.api.anthropic_utils import (
    _decode_document_block,
    _sign_thinking,
    convert_anthropic_to_internal,
    convert_internal_to_anthropic_response,
    create_content_block_start_event,
)


class TestThinkingSignature:
    def test_signature_is_hex_digest(self):
        sig = _sign_thinking("hello world")
        assert len(sig) == 64  # SHA-256 hex digest
        int(sig, 16)  # valid hex

    def test_signature_differs_for_different_content(self):
        sig_a = _sign_thinking("thinking A")
        sig_b = _sign_thinking("thinking B")
        assert sig_a != sig_b

    def test_signature_same_for_same_content(self):
        sig_a = _sign_thinking("same thinking")
        sig_b = _sign_thinking("same thinking")
        assert sig_a == sig_b

    def test_signature_not_placeholder(self):
        sig = _sign_thinking("test")
        assert sig != "fusion-mlx-reasoning"
        assert len(sig) > 16  # not a short placeholder

    def test_non_stream_response_uses_hmac(self):
        resp = convert_internal_to_anthropic_response(
            text="answer",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
            thinking="reasoning here",
        )
        thinking_block = next(
            b for b in resp.content if getattr(b, "type", None) == "thinking"
        )
        assert thinking_block.signature == _sign_thinking("reasoning here")
        assert thinking_block.signature != "fusion-mlx-reasoning"

    def test_stream_event_uses_hmac(self):
        import json

        event = create_content_block_start_event(0, "thinking", thinking="stream think")
        data = json.loads(event.removeprefix("event: content_block_start\ndata: "))
        assert data["content_block"]["signature"] == _sign_thinking("stream think")
        assert data["content_block"]["signature"] != "fusion-mlx-reasoning"


class TestPdfDocumentBlock400:
    def _pdf_block(self, title: str = "doc.pdf") -> dict:
        return {
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0xLjQ=",
            },
            "title": title,
        }

    def test_pdf_raises_400(self):
        with pytest.raises(HTTPException) as exc:
            _decode_document_block(self._pdf_block())
        assert exc.value.status_code == 400
        assert "application/pdf" in exc.value.detail

    def test_pdf_400_in_convert(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockDocument(
                            source={
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQ=",
                            },
                            title="manual.pdf",
                        ),
                    ],
                ),
            ],
        )
        with pytest.raises(HTTPException) as exc:
            convert_anthropic_to_internal(request)
        assert exc.value.status_code == 400
        assert "manual.pdf" in exc.value.detail

    def test_text_plain_still_works(self):
        import base64

        text = "Hello from document"
        block = {
            "source": {
                "type": "base64",
                "media_type": "text/plain",
                "data": base64.b64encode(text.encode()).decode(),
            },
            "title": "notes.txt",
        }
        result = _decode_document_block(block)
        assert "Hello from document" in result
        assert "[Document: notes.txt]" in result

    def test_error_message_mentions_text_plain(self):
        with pytest.raises(HTTPException) as exc:
            _decode_document_block(self._pdf_block("report.pdf"))
        assert "text/plain" in exc.value.detail


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
