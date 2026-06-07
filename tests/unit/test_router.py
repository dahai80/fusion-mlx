from unittest.mock import MagicMock, AsyncMock
import pytest

from fusion_mlx.router.router import RequestRouter


class TestRequestRouter:
     def _make_router(self, **overrides):
        engines = {
            "llm_engine": AsyncMock(),
            "vlm_engine": AsyncMock(),
            "stt_engine": AsyncMock(),
            "tts_engine": AsyncMock(),
            "sts_engine": AsyncMock(),
            "image_gen_engine": AsyncMock(),
            "embedding_engine": AsyncMock(),
            "reranker_engine": AsyncMock(),
        }
        engines.update(overrides)
        return RequestRouter(**engines)

     def test_select_llm_engine_for_text(self):
        router = self._make_router()
        engine, etype = router.select_engine([{"role": "user", "content": "hello"}], {})
        assert etype == "llm"

     def test_select_vlm_engine_for_images(self):
        router = self._make_router()
        messages = [{"role": "user", "content": [{"type": "image_url", "url": "http://x.png"}]}]
        engine, etype = router.select_engine(messages, {})
        assert etype == "vlm"

     def test_select_embedding_engine(self):
        router = self._make_router()
        engine, etype = router.select_engine([], {"task": "embedding"})
        assert etype == "embedding"

     def test_select_reranker_engine(self):
        router = self._make_router()
        engine, etype = router.select_engine([], {"task": "rerank"})
        assert etype == "reranker"

     def test_fallback_to_llm_when_no_vlm(self):
        router = self._make_router(vlm_engine=None)
        messages = [{"role": "user", "content": [{"type": "image_url", "url": "http://x.png"}]}]
        engine, etype = router.select_engine(messages, {})
        assert etype == "llm"

     def test_no_engine_raises(self):
        router = self._make_router(llm_engine=None, vlm_engine=None)
        messages = [{"role": "user", "content": [{"type": "image_url", "url": "http://x.png"}]}]
        with pytest.raises(RuntimeError, match="No suitable engine"):
            router.select_engine(messages, {})


class TestRequestRouterCloudRouting:
     @pytest.mark.asyncio
     async def test_cloud_routing_on_large_context(self):
        llm = AsyncMock()
        llm.prefix_cache_enabled = True
        llm.count_chat_tokens = MagicMock(return_value=30000)
        cloud = MagicMock()
        cloud.should_route_to_cloud = MagicMock(return_value=True)
        cloud.completion = AsyncMock(return_value={"choices": [{"message": {"content": "cloud"}}]})
        cloud.cloud_model = "gpt-4"

        router = RequestRouter(llm_engine=llm, cloud_router=cloud)
        result = await router.route_chat([{"role": "user", "content": "big prompt"}], {})
        cloud.completion.assert_called_once()

     @pytest.mark.asyncio
     async def test_local_routing_on_small_context(self):
        llm = AsyncMock()
        llm.prefix_cache_enabled = True
        llm.count_chat_tokens = MagicMock(return_value=100)
        llm.chat = AsyncMock(return_value={"choices": [{"message": {"content": "local"}}]})
        cloud = MagicMock()
        cloud.should_route_to_cloud = MagicMock(return_value=False)

        router = RequestRouter(llm_engine=llm, cloud_router=cloud)
        result = await router.route_chat([{"role": "user", "content": "small"}], {})
        llm.chat.assert_called_once()
        cloud.report_local_success.assert_called_once()

     @pytest.mark.asyncio
     async def test_circuit_breaker_on_failure(self):
        llm = AsyncMock()
        llm.prefix_cache_enabled = True
        llm.count_chat_tokens = MagicMock(return_value=100)
        llm.chat = AsyncMock(side_effect=RuntimeError("crash"))
        cloud = MagicMock()
        cloud.should_route_to_cloud = MagicMock(return_value=False)

        router = RequestRouter(llm_engine=llm, cloud_router=cloud)
        with pytest.raises(RuntimeError, match="crash"):
            await router.route_chat([{"role": "user", "content": "fail"}], {})
        cloud.report_local_failure.assert_called_once()
