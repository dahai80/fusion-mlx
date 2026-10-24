# SPDX-License-Identifier: Apache-2.0
# Regression tests for the M-series findings of
# audit/fusion-mlx-codeing-audit-report-1019.md:
#   M13 pin TTL sweep, M14 unlocked eviction fallback, M16 per-model KV
#   admission headroom, M17 stale last_access in the TTL sweep,
#   M21 context preflight before the SSE response is created.

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.responses import StreamingResponse

from fusion_mlx.cache.prefix_cache import BlockAwarePrefixCache
from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool
from fusion_mlx.pool.memory_enforcer import ProcessMemoryEnforcer


class _FakePaged:
    def __init__(self):
        self.freed = []

    def free_block(self, bid):
        self.freed.append(bid)


def _bare_prefix_cache(pinned, last_check=0.0):
    pc = object.__new__(BlockAwarePrefixCache)
    pc._pinned_blocks = pinned
    pc._last_pin_expiry_check = last_check
    pc._cache_lock = threading.RLock()
    pc.paged_cache = _FakePaged()
    return pc


class TestPinTtlSweep:
    # M13: expiry must not depend on a later pin_prefix() call.

    def test_permanent_pin_survives(self):
        pc = _bare_prefix_cache({7: 0.0})
        pc._maybe_expire_pinned_blocks()
        assert pc._pinned_blocks == {7: 0.0}
        assert pc.paged_cache.freed == []

    def test_expired_pin_released_without_new_pin(self):
        pc = _bare_prefix_cache({11: time.time() - 1.0})
        pc._maybe_expire_pinned_blocks()
        assert 11 not in pc._pinned_blocks
        assert pc.paged_cache.freed == [11]

    def test_sweep_is_throttled(self):
        now = time.monotonic()
        pc = _bare_prefix_cache({12: time.time() - 1.0}, last_check=now)
        pc._maybe_expire_pinned_blocks()
        # Inside the interval: nothing scanned, pin intact.
        assert 12 in pc._pinned_blocks
        pc._last_pin_expiry_check = now - 3600
        pc._maybe_expire_pinned_blocks()
        assert 12 not in pc._pinned_blocks

    def test_empty_pin_set_is_a_noop(self):
        pc = _bare_prefix_cache({})
        pc._maybe_expire_pinned_blocks()
        assert pc.paged_cache.freed == []


def _make_pool():
    pool = Mock()
    pool._entries = {}
    pool._find_lru_victim = Mock(return_value=None)
    pool._find_pending_unload_ready_locked = Mock(return_value=None)
    pool._entry_has_active_requests = Mock(return_value=False)
    return pool


def _make_enforcer(pool):
    enforcer = ProcessMemoryEnforcer(
        engine_pool=pool,
        memory_guard_tier="balanced",
        memory_guard_custom_ceiling_gb=4.0,
        poll_interval=1.0,
        prefill_memory_guard=True,
    )
    enforcer._propagate_memory_limit = lambda: None
    enforcer._walk_store_cache_caps = lambda: None
    return enforcer


class _TimeoutLock:
    def acquire(self):
        async def _blocked():
            raise AssertionError("acquire() coroutine must be closed, not awaited")

        return _blocked()


class TestUnlockedEvictionFallback:
    # M14: on lock timeout the fallback must not touch eviction bookkeeping
    # (_find_lru_victim / _eviction_marked) without the pool lock.

    def _run_timeout_poll(self, monkeypatch, entry):
        import fusion_mlx.pool.memory_enforcer as mem_mod

        pool = _make_pool()
        pool._entries = {"m": entry}
        pool._lock = _TimeoutLock()
        enforcer = _make_enforcer(pool)

        ceiling = 10 * 1024**3
        monkeypatch.setattr(enforcer, "_get_hard_limit_bytes", lambda: ceiling)
        # Between soft (0.75 * ceiling for balanced) and hard threshold.
        # balanced tier: soft = 0.90 * ceiling, hard = 0.95 * ceiling.
        monkeypatch.setattr(
            enforcer, "_current_usage_bytes", lambda: int(ceiling * 0.92)
        )

        class _FakeAsyncio:
            @staticmethod
            async def wait_for(coro, timeout):
                coro.close()
                raise TimeoutError

            @staticmethod
            async def to_thread(fn, *a, **k):
                return fn(*a, **k)

        monkeypatch.setattr(mem_mod, "asyncio", _FakeAsyncio)
        asyncio.run(enforcer._check_and_enforce())
        return pool, enforcer

    def test_timeout_does_not_pick_victim_unlocked(self, monkeypatch):
        entry = SimpleNamespace(
            is_loading=True,
            abort_loading=False,
            model_id="m",
            engine=None,
            is_pinned=False,
        )
        pool, enforcer = self._run_timeout_poll(monkeypatch, entry)
        pool._find_lru_victim.assert_not_called()
        assert enforcer._eviction_marked == set()
        assert enforcer._deferred_victim_pick is True

    def test_timeout_still_aborts_loading_models(self, monkeypatch):
        entry = SimpleNamespace(
            is_loading=True,
            abort_loading=False,
            model_id="m",
            engine=None,
            is_pinned=False,
        )
        _pool, _enforcer = self._run_timeout_poll(monkeypatch, entry)
        assert entry.abort_loading is True

    def test_timeout_skips_already_marked_entries(self, monkeypatch):
        entry = SimpleNamespace(
            is_loading=True,
            abort_loading=True,
            model_id="m",
            engine=None,
            is_pinned=False,
        )
        _pool, _enforcer = self._run_timeout_poll(monkeypatch, entry)
        assert entry.abort_loading is True


class TestKvHeadroomPerModel:
    # M16: the reservation scales with the incoming checkpoint footprint.

    def test_floor_for_tiny_model(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_ADMISSION_KV_HEADROOM_GB", raising=False)
        pool = EnginePool()
        hr = pool._kv_admission_headroom(1 * 1024**3)
        assert hr == 512 * 1024**2

    def test_scales_with_large_model(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_ADMISSION_KV_HEADROOM_GB", raising=False)
        pool = EnginePool()
        hr = pool._kv_admission_headroom(6 * 1024**3)
        assert hr == int(6 * 1024**3 * 0.25)

    def test_capped_by_scheduler_kv_budget(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_ADMISSION_KV_HEADROOM_GB", raising=False)
        pool = EnginePool()
        hr = pool._kv_admission_headroom(40 * 1024**3)
        # SchedulerConfig carries no max_kv_cache_memory here, so the
        # legacy 4 GiB default applies and the cap is 2 GiB.
        assert hr == 2 * 1024**3

    def test_no_model_size_keeps_legacy_cap(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_ADMISSION_KV_HEADROOM_GB", raising=False)
        pool = EnginePool()
        assert pool._kv_admission_headroom(None) == pool._kv_admission_headroom(0)


def _entry(model_id="m", last_access=0.0):
    return EngineEntry(
        model_id=model_id,
        model_path=f"/tmp/{model_id}",
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
        last_access=last_access,
    )


class TestTtlFreshLastAccess:
    # M17: a request served during the settings disk read must reset the TTL.

    def test_touch_during_settings_io_prevents_expiry(self, monkeypatch):
        pool = EnginePool()
        ent = _entry(last_access=time.time() - 3600)
        ent.engine = Mock()
        ent.engine.has_active_requests = Mock(return_value=False)
        pool._entries["m"] = ent

        def _settings(_model_id):
            # Simulate another coroutine serving this model while the TTL
            # sweep is blocked in settings I/O.
            ent.last_access = time.time()
            return SimpleNamespace(ttl_seconds=60)

        pool.unload_engine_async = AsyncMock()

        expired = asyncio.run(
            pool.check_ttl_expirations(SimpleNamespace(get_settings=_settings))
        )
        assert expired == []
        pool.unload_engine_async.assert_not_awaited()

    def test_stale_entry_still_expires(self, monkeypatch):
        pool = EnginePool()
        ent = _entry(last_access=time.time() - 3600)
        ent.engine = Mock()
        ent.engine.has_active_requests = Mock(return_value=False)
        pool._entries["m"] = ent

        pool.unload_engine_async = AsyncMock()

        expired = asyncio.run(
            pool.check_ttl_expirations(
                SimpleNamespace(get_settings=lambda _m: SimpleNamespace(ttl_seconds=60))
            )
        )
        assert expired == ["m"]


class _StubTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text)))


class _StubEngine:
    is_mllm = False

    def __init__(self, max_position_embeddings=32768):
        self._model = SimpleNamespace(
            args=SimpleNamespace(max_position_embeddings=max_position_embeddings)
        )
        self._tokenizer = _StubTokenizer()
        self.stream_chat = Mock()

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "".join(
            m.get("content", "") if isinstance(m, dict) else str(m) for m in messages
        )


def _chat_request(text="hi"):
    from fusion_mlx.api.openai_models import ChatCompletionRequest

    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": text}], stream=True
    )


def _patch_stream_chat_deps(monkeypatch, engine, released):
    import fusion_mlx.fd_watchdog as fd_mod
    import fusion_mlx.server as server_mod
    import fusion_mlx.service.helpers as helpers_mod
    from fusion_mlx.api.openai import streaming as streaming_mod

    async def _noop_web_search(_request):
        return None

    async def _resolve(model_name, adapter_path=None):
        return engine

    async def _release(model_name, adapter_path=None):
        released.append(model_name)

    class _NoOp:
        def __call__(self, *a, **k):
            return None

    monkeypatch.setattr(fd_mod, "ensure_stream_fd_capacity", _NoOp())
    monkeypatch.setattr(server_mod, "resolve_model_with_profile", lambda m: (m, {}))
    monkeypatch.setattr(streaming_mod, "_resolve_engine", _resolve)
    monkeypatch.setattr(streaming_mod, "_release_engine", _release)
    monkeypatch.setattr(streaming_mod, "_inject_web_search", _noop_web_search)
    monkeypatch.setattr(streaming_mod, "check_chat_capability", _NoOp())
    monkeypatch.setattr(streaming_mod, "check_multimodal_content", _NoOp())
    monkeypatch.setattr(streaming_mod, "check_tool_choice_support", _NoOp())
    monkeypatch.setattr(
        helpers_mod,
        "compute_prompt_tokens_for_messages",
        lambda *a, **k: 5000,
    )
    return streaming_mod


class TestContextPreflightBeforeStream:
    # M21: a rejected prompt must raise before StreamingResponse exists, and
    # must release the engine ref it was holding.

    def test_oversized_prompt_raises_and_releases(self, monkeypatch):
        from fastapi import HTTPException

        engine = _StubEngine()
        released = []
        _patch_stream_chat_deps(monkeypatch, engine, released)
        import fusion_mlx.server as server_mod

        monkeypatch.setattr(server_mod, "get_max_context_window", lambda _m: 1000)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_chat_stream(monkeypatch))
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"]["code"] == "context_length_exceeded"
        assert released == ["m"]

    def test_preflight_result_passed_to_generator(self, monkeypatch):
        engine = _StubEngine(max_position_embeddings=100000)
        released = []
        streaming_mod = _patch_stream_chat_deps(monkeypatch, engine, released)
        import fusion_mlx.server as server_mod

        monkeypatch.setattr(server_mod, "get_max_context_window", lambda _m: 100000)

        captured = {}

        async def _fake_gen(request, eng, model_name, adapter_path, **kwargs):
            captured.update(kwargs)
            yield "data: {}\n\n"

        monkeypatch.setattr(streaming_mod, "_stream_chat_generator", _fake_gen)

        resp = asyncio.run(_chat_stream(monkeypatch))
        assert isinstance(resp, StreamingResponse)

        async def _first_chunk():
            async for chunk in resp.body_iterator:
                return chunk
            return None

        assert asyncio.run(_first_chunk()) is not None
        assert captured["preflight"][1] == 5000
        assert released == []


async def _chat_stream(monkeypatch):
    from fusion_mlx.api.openai import streaming as streaming_mod

    return await streaming_mod._stream_chat(_chat_request(), _skip_cap_check=True)


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
