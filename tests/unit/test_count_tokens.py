# SPDX-License-Identifier: Apache-2.0
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_mlx.api.anthropic_models import (
    AnthropicMessage,
    AnthropicTool,
    ContentBlockText,
    SystemContent,
    ThinkingConfig,
    TokenCountRequest,
    TokenCountResponse,
)
from fusion_mlx.api.anthropic_routes import (
    _encode_token_count,
    _extract_request_text,
)
from fusion_mlx.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


class TestExtractRequestText:
    def test_simple_messages(self):
        req = TokenCountRequest(
            model="test-model",
            messages=[AnthropicMessage(role="user", content="Hello world")],
        )
        assert "Hello world" in _extract_request_text(req)

    def test_system_string(self):
        req = TokenCountRequest(
            model="test-model",
            messages=[AnthropicMessage(role="user", content="Hi")],
            system="You are helpful",
        )
        text = _extract_request_text(req)
        assert "You are helpful" in text
        assert "Hi" in text

    def test_system_list(self):
        req = TokenCountRequest(
            model="test-model",
            messages=[AnthropicMessage(role="user", content="Hi")],
            system=[SystemContent(text="Be concise")],
        )
        text = _extract_request_text(req)
        assert "Be concise" in text

    def test_tools_extracted(self):
        req = TokenCountRequest(
            model="test-model",
            messages=[AnthropicMessage(role="user", content="Go")],
            tools=[AnthropicTool(name="run", description="Run it", input_schema={})],
        )
        text = _extract_request_text(req)
        assert "run" in text

    def test_content_blocks(self):
        req = TokenCountRequest(
            model="test-model",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[ContentBlockText(text="Block text")],
                )
            ],
        )
        text = _extract_request_text(req)
        assert "Block text" in text

    def test_empty_messages(self):
        req = TokenCountRequest(model="test-model", messages=[])
        text = _extract_request_text(req)
        assert text == ""

    def test_dict_content_blocks(self):
        req = TokenCountRequest(
            model="test-model",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[{"type": "text", "text": "Dict block"}],
                )
            ],
        )
        text = _extract_request_text(req)
        assert "Dict block" in text


class TestEncodeTokenCount:
    def test_list_encode(self):
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        assert _encode_token_count(tok, "hello") == 5

    def test_nested_tokenizer(self):
        inner = MagicMock()
        inner.encode.return_value = [10, 20]
        tok = MagicMock(spec=["tokenizer"])
        tok.tokenizer = inner
        assert _encode_token_count(tok, "hi") == 2

    def test_fallback_estimate(self):
        tok = MagicMock(spec=[])
        result = _encode_token_count(tok, "abcdefgh")
        assert result == 2  # 8 // 4


class TestCountTokensRoute:
    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        engine.tokenizer = MagicMock()
        engine.tokenizer.encode.return_value = list(range(12))
        engine.build_prompt.return_value = "<|im_start|>user\nHello<|im_end|>"
        return engine

    @pytest.mark.asyncio
    async def test_model_not_found(self):
        from fusion_mlx.api.anthropic_routes import count_tokens

        req = TokenCountRequest(
            model="nonexistent",
            messages=[AnthropicMessage(role="user", content="Hi")],
        )
        with patch(
            "fusion_mlx.api.anthropic_routes._resolve_engine",
            new_callable=AsyncMock,
            side_effect=ModelNotFoundError("nonexistent"),
        ):
            with pytest.raises(Exception) as exc_info:
                await count_tokens(req, _auth=True)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_real_tokenizer_count(self, mock_engine):
        from fusion_mlx.api.anthropic_routes import count_tokens

        req = TokenCountRequest(
            model="test-model",
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        with (
            patch(
                "fusion_mlx.api.anthropic_routes._resolve_engine",
                new_callable=AsyncMock,
                return_value=mock_engine,
            ),
            patch(
                "fusion_mlx.api.anthropic_routes._release_engine",
                new_callable=AsyncMock,
            ),
        ):
            resp = await count_tokens(req, _auth=True)
            assert isinstance(resp, TokenCountResponse)
            assert resp.input_tokens == 12

    @pytest.mark.asyncio
    async def test_no_tokenizer_fallback(self):
        from fusion_mlx.api.anthropic_routes import count_tokens

        engine = MagicMock()
        engine.tokenizer = None
        req = TokenCountRequest(
            model="test-model",
            messages=[AnthropicMessage(role="user", content="Hello world test")],
        )
        with (
            patch(
                "fusion_mlx.api.anthropic_routes._resolve_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ),
            patch(
                "fusion_mlx.api.anthropic_routes._release_engine",
                new_callable=AsyncMock,
            ),
        ):
            resp = await count_tokens(req, _auth=True)
            assert isinstance(resp, TokenCountResponse)
            assert resp.input_tokens >= 1

    @pytest.mark.asyncio
    async def test_build_prompt_fallback(self):
        from fusion_mlx.api.anthropic_routes import count_tokens

        engine = MagicMock()
        engine.tokenizer = MagicMock()
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.build_prompt.side_effect = RuntimeError("no template")
        req = TokenCountRequest(
            model="test-model",
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        with (
            patch(
                "fusion_mlx.api.anthropic_routes._resolve_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ),
            patch(
                "fusion_mlx.api.anthropic_routes._release_engine",
                new_callable=AsyncMock,
            ),
        ):
            resp = await count_tokens(req, _auth=True)
            assert isinstance(resp, TokenCountResponse)
            assert resp.input_tokens >= 1

    @pytest.mark.asyncio
    async def test_with_system_and_tools(self, mock_engine):
        from fusion_mlx.api.anthropic_routes import count_tokens

        req = TokenCountRequest(
            model="test-model",
            messages=[AnthropicMessage(role="user", content="Go")],
            system="You are helpful",
            tools=[AnthropicTool(name="run", description="Run", input_schema={})],
        )
        with (
            patch(
                "fusion_mlx.api.anthropic_routes._resolve_engine",
                new_callable=AsyncMock,
                return_value=mock_engine,
            ),
            patch(
                "fusion_mlx.api.anthropic_routes._release_engine",
                new_callable=AsyncMock,
            ),
        ):
            resp = await count_tokens(req, _auth=True)
            assert isinstance(resp, TokenCountResponse)
            assert resp.input_tokens >= 1

    @pytest.mark.asyncio
    async def test_with_thinking_enabled(self, mock_engine):
        from fusion_mlx.api.anthropic_routes import count_tokens

        req = TokenCountRequest(
            model="test-model",
            messages=[AnthropicMessage(role="user", content="Think")],
            thinking=ThinkingConfig(type="enabled", budget_tokens=5000),
        )
        with (
            patch(
                "fusion_mlx.api.anthropic_routes._resolve_engine",
                new_callable=AsyncMock,
                return_value=mock_engine,
            ),
            patch(
                "fusion_mlx.api.anthropic_routes._release_engine",
                new_callable=AsyncMock,
            ),
        ):
            resp = await count_tokens(req, _auth=True)
            assert isinstance(resp, TokenCountResponse)
            assert resp.input_tokens >= 1
