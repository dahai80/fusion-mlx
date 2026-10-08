# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1049.

#1049: streaming paths did not接入 circuit breaker stats.
- CloudRouter.stream_completion: mid-stream exception yielded an error frame
  but no [DONE] and no report_cloud_failure() → EF-4 cloud breaker never
  tripped on persistent mid-stream断流.
- RequestRouter.route_stream_chat: returned the raw engine.stream_chat
  iterator → local mid-stream failures invisible to report_local_failure
  (only non-stream route_chat covered).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fusion_mlx.dispatch.cloud_router import CloudRouter
from fusion_mlx.dispatch.router import RequestRouter


async def _aiter_chunks(chunks, fail_at=None, exc=None):
    for i, c in enumerate(chunks):
        if fail_at is not None and i == fail_at:
            raise exc
        yield c


def _litellm_chunk(content="hi", finish_reason=None):
    chunk = MagicMock()
    delta = MagicMock()
    delta.role = None
    delta.content = content
    delta.tool_calls = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


class TestLocalStreamCircuitBreaker:
    """route_stream_chat wraps the local stream with failure tracking."""

    @pytest.mark.asyncio
    async def test_local_stream_failure_reports_local_failure(self):
        llm = AsyncMock()
        llm.prefix_cache_enabled = True
        llm.count_chat_tokens = MagicMock(return_value=100)
        llm.stream_chat = MagicMock(
            return_value=_aiter_chunks(["a", "b"], fail_at=1, exc=RuntimeError("boom"))
        )
        cloud = MagicMock()
        cloud.should_route_to_cloud = MagicMock(return_value=False)

        router = RequestRouter(llm_engine=llm, cloud_router=cloud)
        stream = await router.route_stream_chat([{"role": "user", "content": "hi"}], {})
        collected = []
        with pytest.raises(RuntimeError, match="boom"):
            async for chunk in stream:
                collected.append(chunk)
        assert collected == ["a"]
        cloud.report_local_failure.assert_called_once()
        cloud.report_local_success.assert_not_called()

    @pytest.mark.asyncio
    async def test_local_stream_success_reports_local_success(self):
        llm = AsyncMock()
        llm.prefix_cache_enabled = True
        llm.count_chat_tokens = MagicMock(return_value=100)
        llm.stream_chat = MagicMock(return_value=_aiter_chunks(["a", "b"]))
        cloud = MagicMock()
        cloud.should_route_to_cloud = MagicMock(return_value=False)

        router = RequestRouter(llm_engine=llm, cloud_router=cloud)
        stream = await router.route_stream_chat([{"role": "user", "content": "hi"}], {})
        collected = [c async for c in stream]
        assert collected == ["a", "b"]
        cloud.report_local_success.assert_called_once()
        cloud.report_local_failure.assert_not_called()


class TestCloudStreamCircuitBreaker:
    """CloudRouter.stream_completion reports cloud failure + [DONE] on break."""

    @pytest.mark.asyncio
    async def test_cloud_stream_failure_reports_and_done(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=1000)
        assert cr._cloud_failure_count == 0

        fake_response = _aiter_chunks(
            [_litellm_chunk("hi"), _litellm_chunk("there")],
            fail_at=1,
            exc=RuntimeError("upstream dropped"),
        )
        cr._get_litellm = MagicMock(return_value=MagicMock())

        async def fake_call(litellm_mod, call_kwargs, is_stream):
            return fake_response

        cr._call_cloud = fake_call

        chunks = []
        async for chunk in cr.stream_completion([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert cr._cloud_failure_count == 1
        joined = "".join(chunks)
        assert "cloud_stream_error" in joined
        assert "data: [DONE]\n\n" in joined

    @pytest.mark.asyncio
    async def test_cloud_stream_success_no_cloud_failure(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=1000)
        assert cr._cloud_failure_count == 0

        fake_response = _aiter_chunks(
            [_litellm_chunk("hi"), _litellm_chunk(finish_reason="stop")]
        )
        cr._get_litellm = MagicMock(return_value=MagicMock())

        async def fake_call(litellm_mod, call_kwargs, is_stream):
            return fake_response

        cr._call_cloud = fake_call

        chunks = [
            c async for c in cr.stream_completion([{"role": "user", "content": "hi"}])
        ]
        assert cr._cloud_failure_count == 0
        assert chunks[-1] == "data: [DONE]\n\n"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
