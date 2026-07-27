# SPDX-License-Identifier: Apache-2.0
# Tests for output_format='raw' (F8) and T5 embed cache (F9).

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.engines.video_backends.base import VideoGenParams


# ---------------------------------------------------------------------------
# F8: output_format='raw' tests
# ---------------------------------------------------------------------------


class TestVideoGenParamsOutputFormat:
    def test_default_is_mp4(self):
        p = VideoGenParams(prompt="test")
        assert p.output_format == "mp4"

    def test_raw_accepted(self):
        p = VideoGenParams(prompt="test", output_format="raw")
        assert p.output_format == "raw"

    def test_mp4_explicit(self):
        p = VideoGenParams(prompt="test", output_format="mp4")
        assert p.output_format == "mp4"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="output_format must be one of"):
            VideoGenParams(prompt="test", output_format="avi")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="output_format must be one of"):
            VideoGenParams(prompt="test", output_format="")

    def test_case_sensitive(self):
        with pytest.raises(ValueError, match="output_format must be one of"):
            VideoGenParams(prompt="test", output_format="RAW")


class TestWan2RawOutput:
    def test_raw_path_passes_output_format_to_generate(self, monkeypatch):
        from fusion_mlx.engines.video_backends.wan2 import Wan2Backend

        backend = Wan2Backend("wan2-test")
        backend._loaded = True
        backend._model_dir = "/tmp/fake-wan2"

        captured = {}

        def fake_generate_video(model_dir, prompt, **kwargs):
            captured.update(kwargs)
            captured["model_dir"] = model_dir
            captured["prompt"] = prompt
            import numpy as np
            return np.zeros((10, 480, 854, 3), dtype=np.uint8)

        monkeypatch.setattr(
            "fusion_mlx.video.wan2.generate.generate_video",
            fake_generate_video,
        )

        params = VideoGenParams(prompt="test raw", output_format="raw")
        result = asyncio.run(backend.generate(params))

        assert captured.get("output_format") == "raw"
        assert len(result) == 1
        assert result[0].shape == (10, 480, 854, 3)

    def test_mp4_path_does_not_pass_raw(self, monkeypatch):
        from fusion_mlx.engines.video_backends.wan2 import Wan2Backend

        backend = Wan2Backend("wan2-test")
        backend._loaded = True
        backend._model_dir = "/tmp/fake-wan2"

        captured = {}

        def fake_generate_video(model_dir, prompt, **kwargs):
            captured.update(kwargs)
            captured["model_dir"] = model_dir
            captured["prompt"] = prompt
            output_path = kwargs.get("output_path", "/tmp/fake.mp4")
            Path(output_path).write_bytes(b"FAKEMP4")

        monkeypatch.setattr(
            "fusion_mlx.video.wan2.generate.generate_video",
            fake_generate_video,
        )

        params = VideoGenParams(prompt="test mp4", output_format="mp4")
        result = asyncio.run(backend.generate(params))

        assert captured.get("output_format") is None
        assert len(result) == 1
        assert isinstance(result[0], bytes)


# ---------------------------------------------------------------------------
# F9: T5 embed cache tests
# ---------------------------------------------------------------------------


class TestT5EmbedCache:
    def _make_backend(self):
        from fusion_mlx.engines.video_backends.wan2 import Wan2Backend

        backend = Wan2Backend("wan2-test")
        backend._loaded = True
        backend._model_dir = "/tmp/fake-wan2"
        return backend

    def test_cache_miss_calls_encode(self, monkeypatch):
        backend = self._make_backend()

        import mlx.core as mx

        fake_context = mx.zeros((1, 512, 4096))

        encode_calls = {"count": 0}

        def fake_encode_text(encoder, tokenizer, prompt, text_len):
            encode_calls["count"] += 1
            return fake_context

        monkeypatch.setattr(mx, "eval", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fusion_mlx.video.wan2.utils.encode_text", fake_encode_text
        )
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda x: MagicMock(),
        )

        backend._t5_encoder = MagicMock()
        backend._t5_config = MagicMock()
        backend._t5_config.text_len = 512
        backend._t5_config.sample_neg_prompt = ""

        result = backend._get_cached_embeds("hello", None, 512)

        assert result is not None
        context, context_null = result
        assert context is not None
        assert context_null is None
        assert encode_calls["count"] == 1

    def test_cache_hit_returns_cached(self):
        backend = self._make_backend()

        import mlx.core as mx

        cached_context = mx.ones((1, 512, 4096))
        cached_null = mx.zeros((1, 512, 4096))

        backend._t5_encoder = MagicMock()
        backend._t5_config = MagicMock()
        backend._t5_config.text_len = 512
        backend._t5_config.sample_neg_prompt = ""

        with backend._embed_cache_lock:
            backend._embed_cache[("hello", "", 512)] = (cached_context, cached_null)

        result = backend._get_cached_embeds("hello", "", 512)

        assert result is not None
        context, null = result
        assert mx.allclose(context, cached_context)

    def test_lru_eviction_via_get_cached_embeds(self, monkeypatch):
        import mlx.core as mx
        from fusion_mlx.engines.video_backends.wan2 import _T5_EMBED_CACHE_MAX

        backend = self._make_backend()
        backend._t5_encoder = MagicMock()
        backend._t5_config = MagicMock()
        backend._t5_config.text_len = 512
        backend._t5_config.sample_neg_prompt = ""

        fake_context = mx.zeros((1, 512, 4096))

        monkeypatch.setattr(mx, "eval", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fusion_mlx.video.wan2.utils.encode_text",
            lambda enc, tok, p, tl: fake_context,
        )
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda x: MagicMock(),
        )

        for i in range(_T5_EMBED_CACHE_MAX + 2):
            backend._get_cached_embeds(f"prompt_{i}", "", 512)

        assert len(backend._embed_cache) <= _T5_EMBED_CACHE_MAX

    def test_cache_thread_safety(self):
        backend = self._make_backend()
        backend._t5_encoder = MagicMock()
        backend._t5_config = MagicMock()
        backend._t5_config.text_len = 512

        import mlx.core as mx

        fake_val = (mx.zeros((1, 512, 4096)), None)
        errors = []

        def write_cache():
            try:
                for i in range(200):
                    key = (f"prompt_{i}", "", 512)
                    with backend._embed_cache_lock:
                        if key in backend._embed_cache:
                            v = backend._embed_cache.pop(key)
                            backend._embed_cache[key] = v
                        else:
                            if len(backend._embed_cache) >= 16:
                                oldest = next(iter(backend._embed_cache))
                                del backend._embed_cache[oldest]
                            backend._embed_cache[key] = fake_val
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_cache) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread-safety errors: {errors}"
        assert len(backend._embed_cache) <= 16

    def test_cache_cleared_on_stop(self):
        backend = self._make_backend()
        with backend._embed_cache_lock:
            backend._embed_cache[("hello", "", 512)] = ("val", None)

        asyncio.run(backend.stop())
        assert len(backend._embed_cache) == 0

    def test_precomputed_context_wired_to_generate(self, monkeypatch):
        backend = self._make_backend()

        import mlx.core as mx

        cached_context = mx.ones((1, 512, 4096))
        cached_null = mx.zeros((1, 512, 4096))

        backend._t5_encoder = MagicMock()
        backend._t5_config = MagicMock()
        backend._t5_config.text_len = 512
        backend._t5_config.sample_neg_prompt = ""
        with backend._embed_cache_lock:
            backend._embed_cache[("hello", "", 512)] = (cached_context, cached_null)

        captured = {}

        def fake_generate_video(model_dir, prompt, **kwargs):
            captured.update(kwargs)
            output_path = kwargs.get("output_path", "/tmp/fake.mp4")
            Path(output_path).write_bytes(b"FAKEMP4")

        monkeypatch.setattr(
            "fusion_mlx.video.wan2.generate.generate_video",
            fake_generate_video,
        )

        params = VideoGenParams(prompt="hello", output_format="mp4")
        asyncio.run(backend.generate(params))

        assert captured.get("precomputed_context") is not None
        assert captured.get("keep_t5") is True

    def test_no_precomputed_without_t5(self, monkeypatch):
        backend = self._make_backend()
        backend._t5_encoder = None
        backend._t5_config = None

        captured = {}

        def fake_generate_video(model_dir, prompt, **kwargs):
            captured.update(kwargs)
            output_path = kwargs.get("output_path", "/tmp/fake.mp4")
            Path(output_path).write_bytes(b"FAKEMP4")

        monkeypatch.setattr(
            "fusion_mlx.video.wan2.generate.generate_video",
            fake_generate_video,
        )

        params = VideoGenParams(prompt="hello", output_format="mp4")
        asyncio.run(backend.generate(params))

        assert captured.get("precomputed_context") is None
        assert captured.get("keep_t5") is None
