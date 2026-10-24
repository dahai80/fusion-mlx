# SPDX-License-Identifier: Apache-2.0
"""Regression tests for audit 1019 L-series offline findings.

L6  — the process-global ``_waiting_consumers`` counter on the collector
      (semantics globalized across engines) is gone.
L7  — a missing ``max_tokens`` still defaults to 2048 but logs why.
L10 — the request semaphore is bounded, so an over-release is refused
      instead of permanently widening the admission budget.
L19 — a failed seq_len probe is logged at the point it returns a guess.
L20 — the last-resort cache reconstruction warns about dropped blocks.
L35 — expired resumable sessions are reaped before the per-principal cap.
"""

import asyncio
import logging

import pytest

from fusion_mlx.cache.prefix_cache import BlockAwarePrefixCache

_CACHE_LOGGER = "fusion_mlx.cache.prefix_cache"


class _Unreadable:
    @property
    def shape(self):
        raise RuntimeError("unreadable cache state")


def _bare_cache():
    return object.__new__(BlockAwarePrefixCache)


class TestCacheSeqLenProbeVisibility:
    def test_unreadable_states_log_warning(self, caplog):
        cache = _bare_cache()
        with caplog.at_level(logging.WARNING, logger=_CACHE_LOGGER):
            assert (
                cache._get_cache_seq_len([{"state": (_Unreadable(), _Unreadable())}])
                == 0
            )
        assert "seq_len probe failed" in caplog.text

    def test_sliceable_layer_returns_len_without_warning(self, caplog):
        mx = pytest.importorskip("mlx.core")
        cache = _bare_cache()
        keys = mx.zeros((1, 2, 7, 4))
        with caplog.at_level(logging.WARNING, logger=_CACHE_LOGGER):
            assert cache._get_cache_seq_len([{"state": (keys, keys)}]) == 7
        assert "seq_len probe failed" not in caplog.text


class TestFallbackReconstructVisibility:
    def test_incompatible_shapes_warn(self, caplog):
        mx = pytest.importorskip("mlx.core")
        cache = _bare_cache()
        states = [
            {"keys": mx.zeros((1, 1, 4, 8)), "values": mx.zeros((1, 1, 4, 8))},
            {"keys": mx.zeros((2, 3, 5, 9)), "values": mx.zeros((2, 3, 5, 9))},
        ]
        with caplog.at_level(logging.WARNING, logger=_CACHE_LOGGER):
            cache._fallback_reconstruct_layer(states, "KVCache")
        assert "using the last block only" in caplog.text

    def test_concatable_shapes_do_not_warn(self, caplog):
        mx = pytest.importorskip("mlx.core")
        cache = _bare_cache()
        states = [
            {"keys": mx.zeros((1, 1, 4, 8)), "values": mx.zeros((1, 1, 4, 8))},
            {"keys": mx.zeros((1, 1, 4, 8)), "values": mx.zeros((1, 1, 4, 8))},
        ]
        with caplog.at_level(logging.WARNING, logger=_CACHE_LOGGER):
            out = cache._fallback_reconstruct_layer(states, "KVCache")
        assert out is not None
        assert "using the last block only" not in caplog.text


class TestAdapterMaxTokensNotice:
    @staticmethod
    def _request(**overrides):
        from fusion_mlx.api.openai_models import ChatCompletionRequest

        return ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            **overrides,
        )

    def test_missing_max_tokens_defaults_and_warns(self, caplog):
        from fusion_mlx.api.adapters.openai import OpenAIAdapter

        with caplog.at_level(logging.WARNING, logger="fusion_mlx.api.adapters.openai"):
            internal = OpenAIAdapter().parse_request(self._request())
        assert internal.max_tokens == 2048
        assert "defaulting to 2048" in caplog.text

    def test_explicit_max_tokens_is_silent(self, caplog):
        from fusion_mlx.api.adapters.openai import OpenAIAdapter

        with caplog.at_level(logging.WARNING, logger="fusion_mlx.api.adapters.openai"):
            internal = OpenAIAdapter().parse_request(self._request(max_tokens=64))
        assert internal.max_tokens == 64
        assert "defaulting to 2048" not in caplog.text


class TestBoundedRequestSemaphore:
    def test_over_release_refused(self, monkeypatch, caplog):
        import fusion_mlx.api._concurrency as conc

        monkeypatch.setattr(conc, "_request_sem", None)
        monkeypatch.setenv("FUSION_MAX_CONCURRENT_REQUESTS", "2")
        conc.init_request_semaphore(max_num_seqs=1)
        sem = conc.get_request_semaphore()
        assert isinstance(sem, asyncio.BoundedSemaphore)

        async def scenario():
            await conc.acquire_request_slot()
            conc.release_request_slot()
            with caplog.at_level(logging.WARNING, logger="fusion_mlx.api._concurrency"):
                conc.release_request_slot()

        asyncio.run(scenario())
        assert sem._value == 2
        assert "over-release" in caplog.text


class TestExpiredSessionsPruned:
    def test_expired_sessions_do_not_count(self, monkeypatch):
        from fusion_mlx import stream_session

        store = stream_session.StreamSessionStore()
        monkeypatch.setattr(stream_session, "_TTL", -1.0)
        store.create("sess-a", principal="alice")
        store.create("sess-b", principal="alice")

        assert store.count_for_principal("alice") == 0
        assert store._sessions == {}


class TestNoGlobalWaitingCounter:
    def test_collector_has_no_process_global_counter(self):
        from fusion_mlx.output_collector import RequestOutputCollector

        assert not hasattr(RequestOutputCollector, "_waiting_consumers")
        assert not hasattr(RequestOutputCollector, "has_waiting_consumers")
