# SPDX-License-Identifier: Apache-2.0
"""E2E integration tests — full request flow through router -> engine -> response.

These tests require running services and are skipped by default.
Run with: pytest tests/integration/ --integration
"""

import pytest

pytestmark = pytest.mark.integration


class TestChatRequestFlow:

    @pytest.mark.skip(reason="Requires running MLX engine")
    @pytest.mark.asyncio
    async def test_chat_request_full_flow(self):
        """Test: HTTP request -> router -> BatchedEngine.chat -> response."""
        from fusion_mlx.router.smart_router import SmartRouter, RouterConfig
        from unittest.mock import AsyncMock, MagicMock

        engine = AsyncMock()
        engine.chat = AsyncMock(return_value=MagicMock(
             output_text="Hello! How can I help?",
             prompt_tokens=10, completion_tokens=8, finish_reason="stop",
             tool_calls=[], cached_tokens=0,
           ))
        engine.stream_chat = AsyncMock(return_value=iter([
            MagicMock(text="Hello", new_text="Hello", finished=False),
            MagicMock(text="!", new_text="!", finished=True),
         ]))

        router = SmartRouter(llm_engine=engine, rapid_engine=engine)
        result = await router.route_chat(
             [{"role": "user", "content": "Hello"}], {}, prompt_length=3,
           )
        assert result is not None

    @pytest.mark.skip(reason="Requires running MLX engine")
    @pytest.mark.asyncio
    async def test_stream_chat_full_flow(self):
        """Test: streaming request -> router -> engine.stream_chat -> SSE chunks."""
        from fusion_mlx.router.smart_router import SmartRouter, RouterConfig
        from unittest.mock import AsyncMock

        engine = AsyncMock()

        async def fake_stream():
             yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
             yield 'data: {"choices":[{"delta":{"content":"!"}}]}\n\n'
             yield 'data: [DONE]\n\n'

        engine.stream_chat = fake_stream

        router = SmartRouter(llm_engine=engine, rapid_engine=engine)
        chunks = []
        async for chunk in router.route_stream_chat(
             [{"role": "user", "content": "Hello"}], {}, prompt_length=3,
           ):
            chunks.append(chunk)
        assert len(chunks) == 3


class TestCloudFallbackFlow:

    @pytest.mark.skip(reason="Requires cloud API key")
    @pytest.mark.asyncio
    async def test_cloud_fallback_for_large_prompt(self):
        """Test: large prompt -> cloud router -> cloud API -> response."""
        from fusion_mlx.router.router import RequestRouter
        from unittest.mock import MagicMock, AsyncMock

        llm = AsyncMock()
        llm.prefix_cache_enabled = True
        llm.count_chat_tokens = MagicMock(return_value=50000)

        cloud = MagicMock()
        cloud.should_route_to_cloud = MagicMock(return_value=True)
        cloud.completion = AsyncMock(return_value={
             "choices": [{"message": {"content": "cloud response"}}]
           })
        cloud.cloud_model = "gpt-4"

        router = RequestRouter(llm_engine=llm, cloud_router=cloud)
        result = await router.route_chat(
             [{"role": "user", "content": "x" * 10000}], {},
           )
        cloud.completion.assert_called_once()
        assert "cloud response" in str(result)


class TestPhaseSplitFlow:

    @pytest.mark.skip(reason="Requires two running MLX engines")
    @pytest.mark.asyncio
    async def test_phase_split_prefill_decode(self):
        """Test: long prompt -> prefill on omlx -> KV handoff -> decode on Rapid."""
        from fusion_mlx.router.smart_router import SmartRouter, RouterConfig
        from unittest.mock import AsyncMock, MagicMock

        prefill_engine = AsyncMock()
        prefill_engine.supports_prefill_only = True
        prefill_engine.chat = AsyncMock(return_value=MagicMock(
             kv_state={"block_table": MagicMock(), "num_computed_tokens": 1000},
           ))

        decode_engine = AsyncMock()
        decode_engine.supports_kv_handoff = True

        async def fake_decode_stream():
             yield 'data: {"choices":[{"delta":{"content":"decoded"}}]}\n\n'
             yield 'data: [DONE]\n\n'

        decode_engine.stream_chat = fake_decode_stream

           # Set up router with low split threshold to trigger phase split
        config = RouterConfig(phase_split_threshold=10, cloud_fallback_threshold=999999)
        router = SmartRouter(
             config=config, llm_engine=prefill_engine, rapid_engine=decode_engine,
           )
        chunks = []
        async for chunk in router.route_stream_chat(
             [{"role": "user", "content": "x" * 100}], {},
             prompt_length=100, cache_hit_rate=0.0,
           ):
            chunks.append(chunk)
        assert len(chunks) >= 1
