# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1005.

#1005: ``/v1/count_tokens`` acquired an engine lease (_lease=True via
_resolve_engine) but the body after it — convert_anthropic_to_internal,
build_prompt, _encode_token_count — was not in a try/finally. Only the
happy path and two specific branches manually released. A malformed
request (e.g. multimodal convert failure) raised mid-body, leaving
entry.in_use permanently +1 so the model could never be evicted (LRU +
TTL skip in_use>0).

Fix (#1005): wrapped the entire post-resolve body in try/finally that
calls _release_engine on EVERY path (return + exception), matching the
_release closure pattern in the messages/stream paths.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from fusion_mlx.api.anthropic_models import TokenCountRequest
from fusion_mlx.api.anthropic_routes import count_tokens


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text)))


class _FakeEngine:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "".join(
            m.get("content", "") if isinstance(m, dict) else str(m) for m in messages
        )


def _make_request():
    return TokenCountRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello world"}],
    )


def _make_pool(engine):
    pool = MagicMock()
    pool.get_engine = AsyncMock(return_value=engine)
    pool.release_engine = AsyncMock(return_value=None)
    return pool


class TestCountTokensLeaseRelease:
    def _setup(self, monkeypatch, engine=None):
        import fusion_mlx.api.anthropic_routes as ar

        eng = engine or _FakeEngine()
        pool = _make_pool(eng)
        monkeypatch.setattr(ar, "_pool", pool)
        return ar, pool

    def test_happy_path_releases_once(self, monkeypatch):
        ar, pool = self._setup(monkeypatch)
        resp = asyncio.run(count_tokens(_make_request()))
        assert resp.input_tokens > 0
        pool.release_engine.assert_awaited_once()

    def test_convert_exception_still_releases(self, monkeypatch):
        # #1005 core: convert_anthropic_to_internal raises mid-body. The
        # lease MUST still be released (previously leaked in_use+1 forever).
        ar, pool = self._setup(monkeypatch)

        def _boom(*a, **k):
            raise ValueError("malformed multimodal content")

        monkeypatch.setattr(
            "fusion_mlx.api.anthropic_utils.convert_anthropic_to_internal", _boom
        )
        with pytest.raises(ValueError, match="malformed"):
            asyncio.run(count_tokens(_make_request()))
        pool.release_engine.assert_awaited_once()

    def test_build_prompt_exception_still_releases(self, monkeypatch):
        # build_prompt failure is caught by the handler (graceful fallback
        # to text encoding) — but the lease must still be released on the
        # fallback return path.
        ar, pool = self._setup(monkeypatch)
        eng = pool.get_engine.return_value
        eng.build_prompt = MagicMock(side_effect=RuntimeError("bp boom"))
        resp = asyncio.run(count_tokens(_make_request()))
        assert resp.input_tokens >= 1
        pool.release_engine.assert_awaited_once()

    def test_encode_exception_still_releases(self, monkeypatch):
        ar, pool = self._setup(monkeypatch)
        monkeypatch.setattr(
            "fusion_mlx.api.anthropic_routes._encode_token_count",
            lambda tok, text: (_ for _ in ()).throw(RuntimeError("encode boom")),
        )
        with pytest.raises(RuntimeError, match="encode boom"):
            asyncio.run(count_tokens(_make_request()))
        pool.release_engine.assert_awaited_once()

    def test_no_tokenizer_path_releases_once(self, monkeypatch):
        ar, pool = self._setup(monkeypatch, engine=_FakeEngine())
        pool.get_engine.return_value.tokenizer = None
        resp = asyncio.run(count_tokens(_make_request()))
        assert resp.input_tokens >= 1
        pool.release_engine.assert_awaited_once()

    def test_not_found_does_not_acquire_lease(self, monkeypatch):
        import fusion_mlx.api.anthropic_routes as ar
        from fusion_mlx.exceptions import ModelNotFoundError

        pool = MagicMock()
        pool.get_engine = AsyncMock(side_effect=ModelNotFoundError("nope"))
        pool.release_engine = AsyncMock(return_value=None)
        monkeypatch.setattr(ar, "_pool", pool)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(count_tokens(_make_request()))
        assert exc_info.value.status_code == 404
        # No engine was acquired -> no release needed.
        pool.release_engine.assert_not_awaited()


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
