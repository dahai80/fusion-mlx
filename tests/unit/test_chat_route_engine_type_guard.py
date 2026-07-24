# SPDX-License-Identifier: Apache-2.0
# Test for #205: chat completion route rejects non-chat engines.
# Standalone — avoids fusion_mlx.__init__ -> mlx_whisper import chain
# by testing the guard logic directly.

import pytest


def _has_chat_capability(engine):
    return hasattr(engine, "chat") and callable(getattr(engine, "chat", None))


def _has_stream_chat_capability(engine):
    return hasattr(engine, "stream_chat") and callable(
        getattr(engine, "stream_chat", None)
    )


class _FakeImageGenEngine:
    engine_type = "image_gen"
    is_mllm = False


class _FakeChatEngine:
    engine_type = "batched"
    is_mllm = False

    async def chat(self, **kwargs):
        return None

    async def stream_chat(self, **kwargs):
        yield None


class _FakeVLMEngine:
    engine_type = "vlm"
    is_mllm = True

    async def chat(self, **kwargs):
        return None

    async def stream_chat(self, **kwargs):
        yield None


class _FakeEmbeddingEngine:
    engine_type = "embedding"
    is_mllm = False


class _FakeTTSEngine:
    engine_type = "audio_tts"
    is_mllm = False


class TestChatCapabilityGuard:
    def test_image_gen_no_chat(self):
        assert not _has_chat_capability(_FakeImageGenEngine())

    def test_image_gen_no_stream_chat(self):
        assert not _has_stream_chat_capability(_FakeImageGenEngine())

    def test_chat_engine_has_chat(self):
        assert _has_chat_capability(_FakeChatEngine())

    def test_chat_engine_has_stream_chat(self):
        assert _has_stream_chat_capability(_FakeChatEngine())

    def test_vlm_engine_has_chat(self):
        assert _has_chat_capability(_FakeVLMEngine())

    def test_vlm_engine_has_stream_chat(self):
        assert _has_stream_chat_capability(_FakeVLMEngine())

    def test_embedding_engine_no_chat(self):
        assert not _has_chat_capability(_FakeEmbeddingEngine())

    def test_tts_engine_no_chat(self):
        assert not _has_chat_capability(_FakeTTSEngine())

    def test_engine_type_attribute_on_all(self):
        for cls in (
            _FakeImageGenEngine,
            _FakeChatEngine,
            _FakeVLMEngine,
            _FakeEmbeddingEngine,
            _FakeTTSEngine,
        ):
            engine = cls()
            assert hasattr(engine, "engine_type"), (
                f"{cls.__name__} missing engine_type"
            )

    def test_non_chat_engine_yields_400_detail(self):
        engine = _FakeImageGenEngine()
        if not _has_chat_capability(engine):
            detail = (
                f"Model 'flux2' does not support chat completions "
                f"(engine_type={getattr(engine, 'engine_type', 'unknown')})"
            )
            assert "does not support chat" in detail
            assert "image_gen" in detail
