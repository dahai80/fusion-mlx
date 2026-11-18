# SPDX-License-Identifier: Apache-2.0
"""M9 step 2: deferred KV-drop delivery — scheduler queue drained inside
step() context, engine API queues + wakes, unknown requests skipped."""

import asyncio
import logging

from fusion_mlx.cache.paged_cache import PagedCacheManager
from fusion_mlx.cache.prefix_cache import BlockAwarePrefixCache
from fusion_mlx.scheduler import sched_trim

logger = logging.getLogger(__name__)


class _MockModel:
    def __init__(self):
        import unittest.mock as _m

        self.layers = [_m.MagicMock() for _ in range(4)]

    @property
    def args(self):
        import unittest.mock as _m

        a = _m.MagicMock()
        a.num_hidden_layers = 4
        return a


def _make_cache():
    pc = PagedCacheManager(
        block_size=4,
        max_blocks=100,
        model_name="test-model",
        initial_blocks=100,
    )
    bac = BlockAwarePrefixCache(
        model=_MockModel(), paged_cache_manager=pc, paged_ssd_cache_manager=None
    )
    tokens = list(range(24))
    table = pc.get_or_create_block_table("req-a")
    parent = None
    for i in range(0, 24, 4):
        b = pc.allocate_block()
        b.token_count = 4
        pc.register_block_hash(b, tokens[i : i + 4], parent)
        parent = b.block_hash
        table.add_block(b.block_id, 4)
    bac._update_prefix_index(tokens, table.block_ids)
    return pc, bac, tokens


class _FakeSched:
    def __init__(self, bac, requests):
        self.block_aware_cache = bac
        self.requests = requests
        self._pending_kv_drops = {}


class TestSchedulerDelivery:
    def test_queue_then_process_applies_drop(self):
        pc, bac, tokens = _make_cache()
        sched = _FakeSched(bac, {"req-a": object()})
        sched_trim.queue_kv_drop(sched, "req-a", 4, 12)
        assert len(sched._pending_kv_drops["req-a"]) == 1
        sched_trim._process_pending_kv_drops(sched)
        assert sched._pending_kv_drops == {}
        # Drop applied: later fetch must not match past block 0.
        table, remaining = bac.fetch_cache("other", tokens)
        matched = 0 if table is None else table.num_tokens
        assert matched <= 4

    def test_unknown_request_skipped_and_drained(self):
        pc, bac, _ = _make_cache()
        sched = _FakeSched(bac, {})
        sched_trim.queue_kv_drop(sched, "missing", 0, 8)
        sched_trim._process_pending_kv_drops(sched)
        assert sched._pending_kv_drops == {}
        # Cache untouched: full match still works for a fresh fetch.
        tokens = list(range(24))
        table, _ = bac.fetch_cache("x", tokens)
        assert table is not None

    def test_no_cache_manager_skipped(self):
        sched = _FakeSched(None, {"req": object()})
        sched_trim.queue_kv_drop(sched, "req", 0, 8)
        sched_trim._process_pending_kv_drops(sched)
        assert sched._pending_kv_drops == {}

    def test_bad_cache_manager_does_not_raise(self):
        class _Boom:
            def drop_request_segments(self, *a):
                raise RuntimeError("boom")

        sched = _FakeSched(_Boom(), {"req": object()})
        sched_trim.queue_kv_drop(sched, "req", 0, 8)
        sched_trim._process_pending_kv_drops(sched)
        assert sched._pending_kv_drops == {}


class TestEngineApi:
    def test_engine_core_queues_and_wakes(self):
        from fusion_mlx.engine_core import EngineCore

        queued = []

        class _Sched:
            def queue_kv_drop(self, rid, s, e):
                queued.append((rid, s, e))

        core = object.__new__(EngineCore)
        core.scheduler = _Sched()
        core._closed = False
        wakes = []
        core._wake_engine_loop = lambda: wakes.append(1)
        n = asyncio.new_event_loop().run_until_complete(
            core.drop_cache_segments("req-a", [(4, 12), (16, 20)])
        )
        assert n == 2
        assert queued == [("req-a", 4, 12), ("req-a", 16, 20)]
        assert wakes == [1]

    def test_engine_core_closed_returns_zero(self):
        from fusion_mlx.engine_core import EngineCore

        core = object.__new__(EngineCore)
        core.scheduler = None
        core._closed = True
        n = asyncio.new_event_loop().run_until_complete(
            core.drop_cache_segments("req-a", [(4, 12)])
        )
        assert n == 0

    def test_async_engine_core_delegates(self):
        from fusion_mlx.engine_core import AsyncEngineCore

        calls = []

        class _Engine:
            async def drop_cache_segments(self, rid, segs):
                calls.append((rid, segs))
                return len(segs)

        core = object.__new__(AsyncEngineCore)
        core.engine = _Engine()
        n = asyncio.new_event_loop().run_until_complete(
            core.drop_cache_segments("req-a", [(4, 12)])
        )
        assert n == 1
        assert calls == [("req-a", [(4, 12)])]

    def test_async_engine_core_no_engine_returns_zero(self):
        from fusion_mlx.engine_core import AsyncEngineCore

        core = object.__new__(AsyncEngineCore)
        core.engine = None
        n = asyncio.new_event_loop().run_until_complete(
            core.drop_cache_segments("req-a", [(4, 12)])
        )
        assert n == 0


class TestDropRoute:
    def _call(self, monkeypatch, *, pool, env, body):
        import types

        from fastapi import HTTPException

        import fusion_mlx.routes_internal.cache as cache_routes
        from fusion_mlx.routes_internal.cache import DropSegmentsRequest

        state = {"engine_pool": pool}
        fake_server = types.SimpleNamespace(_server_state=state)
        monkeypatch.setattr(cache_routes, "_server_state", state, raising=False)
        monkeypatch.setitem(__import__("sys").modules, "fusion_mlx.server", fake_server)
        monkeypatch.setenv("FUSION_MLX_KV_DROP_API", env)
        req = DropSegmentsRequest(**body)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(cache_routes.drop_cache_segments(req))
        except HTTPException as e:
            return {"http_error": e.status_code, "detail": e.detail}
        finally:
            loop.close()

    def test_disabled_by_default(self, monkeypatch):
        out = self._call(
            monkeypatch,
            pool=object(),
            env="0",
            body={"request_id": "r", "segments": [[0, 8]]},
        )
        assert out == {"http_error": 404, "detail": "KV drop API is disabled"}

    def test_enabled_no_pool_503(self, monkeypatch):
        out = self._call(
            monkeypatch,
            pool=None,
            env="1",
            body={"request_id": "r", "segments": [[0, 8]]},
        )
        assert out == {"http_error": 503, "detail": "Engine not loaded"}

    def test_enabled_unknown_request_404(self, monkeypatch):
        class _Pool:
            _entries = {}

        out = self._call(
            monkeypatch,
            pool=_Pool(),
            env="1",
            body={"request_id": "r", "segments": [[0, 8]]},
        )
        assert out == {
            "http_error": 404,
            "detail": "Request not found or already finished",
        }

    def test_enabled_queues_on_owning_engine(self, monkeypatch):
        calls = []

        class _Engine:
            def __init__(self):
                import types

                self.scheduler = types.SimpleNamespace(requests={"req-a": object()})

            async def drop_cache_segments(self, rid, segs):
                calls.append((rid, segs))
                return len(segs)

        class _Entry:
            engine = _Engine()

        class _Pool:
            _entries = {"m": _Entry()}

        out = self._call(
            monkeypatch,
            pool=_Pool(),
            env="1",
            body={"request_id": "req-a", "segments": [[4, 12], [20, 10]]},
        )
        assert out == {"status": "queued", "request_id": "req-a", "queued_segments": 1}
        assert calls == [("req-a", [(4, 12)])]
