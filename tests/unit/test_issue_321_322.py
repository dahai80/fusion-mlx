"""Issue #321/#322 verification: streaming SSE, JSON mode, embedding API."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# -- Issue #321: Streaming SSE confirmation --


class TestStreamingSSE:
    def test_streaming_encoder_formats_sse(self):
        from fusion_mlx.api.streaming import StreamingJSONEncoder

        enc = StreamingJSONEncoder(
            response_id="test-id",
            model="test-model",
            object_type="chat.completion.chunk",
        )
        chunk = enc.encode_chat_chunk(content="hello")
        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")
        payload = json.loads(chunk[len("data: ") :].strip())
        assert payload["object"] == "chat.completion.chunk"
        assert payload["choices"][0]["delta"]["content"] == "hello"
        logger.info("Issue #321: Streaming SSE format verified")

    def test_streaming_done_message(self):
        from fusion_mlx.api.streaming import StreamingJSONEncoder

        enc = StreamingJSONEncoder(
            response_id="test-id",
            model="test-model",
            object_type="chat.completion.chunk",
        )
        done = enc.encode_done()
        assert done == "data: [DONE]\n\n"


# -- Issue #321: JSON Mode --


class TestJSONMode:
    def test_response_format_json_object_dict(self):
        from fusion_mlx.api.openai_models import ResponseFormat

        rf = ResponseFormat(type="json_object")
        assert rf.type == "json_object"
        logger.info("Issue #321: JSON mode response_format verified via dict")

    def test_response_format_json_object_pydantic(self):
        from fusion_mlx.api.openai_models import ResponseFormat

        rf = ResponseFormat(type="json_object")
        assert rf.type == "json_object"


# -- Issue #322: Embedding API --


class TestEmbeddingAPI:
    def test_embedding_request_model(self):
        from fusion_mlx.api.embedding_models import EmbeddingRequest

        req = EmbeddingRequest(model="bge-m3", input="test")
        assert req.model == "bge-m3"
        assert req.input == "test"

    def test_embedding_request_multi_input(self):
        from fusion_mlx.api.embedding_models import EmbeddingRequest

        req = EmbeddingRequest(model="bge-m3", input=["text1", "text2"])
        assert req.model == "bge-m3"
        assert req.input == ["text1", "text2"]

    def test_embedding_response_model(self):
        from fusion_mlx.api.embedding_models import (
            EmbeddingData,
            EmbeddingResponse,
            EmbeddingUsage,
        )

        resp = EmbeddingResponse(
            model="bge-m3",
            data=[EmbeddingData(embedding=[0.1], index=0)],
            usage=EmbeddingUsage(prompt_tokens=1, total_tokens=1),
        )
        assert resp.object == "list"
        assert resp.model == "bge-m3"
        assert len(resp.data) == 1
        logger.info("Issue #322: Embedding API response model verified")

    def test_bge_m3_alias(self):
        import json

        with open(
            Path(__file__).parent.parent.parent / "fusion_mlx" / "aliases.json"
        ) as f:
            aliases = json.load(f)
        assert "bge-m3" in aliases
        assert "bge-m3" in aliases
        assert aliases["bge-m3"]["modality"] == "embedding"
        assert aliases["bge-m3"]["modality"] == "embedding"
        logger.info("Issue #322: bge-m3 alias verified")
