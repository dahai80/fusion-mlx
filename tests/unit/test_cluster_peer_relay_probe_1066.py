# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1066.

#1066: two bugs.
1. peer_lb._relay returned a bare async generator — a consumer that broke
   out of the stream early left the httpx response + client hanging (the
   generator's finally only runs on exhaustion/raise/GC). Fix: wrap the
   relay in contextlib.aclosing so __aexit__ drives gen.aclose().
2. router.Backend.maybe_revive claimed "one probe request" but returned
   True on every select() past cooldown — a recovering bad node took 1/N
   of real traffic continuously. Fix: single-probe mode via a ``probing``
   flag — select() returns the backend once, then skips it until the probe
   resolves (record_success revives; record_failure/mark_dead restart cooldown).
"""

from __future__ import annotations

import inspect

import pytest

from fusion_mlx.cluster import peer_lb
from fusion_mlx.cluster.peer_lb import forward_to_peer
from fusion_mlx.cluster.registry import ClusterNode
from fusion_mlx.cluster.router import Backend, ClusterRouter


class _FakeStreamResp:
    def __init__(self, chunks):
        self.status_code = 200
        self._chunks = chunks
        self.closed = False

    async def aiter_raw(self):
        for c in self._chunks:
            yield c

    async def aclose(self):
        self.closed = True


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp
        self.closed = False

    def build_request(self, *a, **kw):
        return object()

    async def send(self, req, stream=False):
        return self._resp

    async def aclose(self):
        self.closed = True


class TestRelayAclosing1066:
    """Stream relay is wrapped in aclosing — early break releases resources."""

    def test_source_uses_aclosing(self):
        source = inspect.getsource(peer_lb)
        assert "contextlib.aclosing" in source

    @pytest.mark.asyncio
    async def test_early_break_releases_resp_and_client(self, monkeypatch):
        import httpx as _real_httpx

        resp = _FakeStreamResp([b"a", b"b", b"c"])
        client = _FakeClient(resp)
        monkeypatch.setattr(_real_httpx, "AsyncClient", lambda *a, **kw: client)
        node = ClusterNode(node_id="p", host="p", port=1)
        node.base_url = "http://p:1"  # type: ignore[attr-defined]
        relay = await forward_to_peer(node, "POST", "/v1/chat", stream=True)
        # consume one chunk then break — aclosing.__aexit__ must close resp+client
        async with relay as it:
            async for chunk in it:
                assert chunk == b"a"
                break
        assert resp.closed, "resp not closed after early break"
        assert client.closed, "client not closed after early break"

    @pytest.mark.asyncio
    async def test_full_consumption_releases(self, monkeypatch):
        import httpx as _real_httpx

        resp = _FakeStreamResp([b"x", b"y"])
        client = _FakeClient(resp)
        monkeypatch.setattr(_real_httpx, "AsyncClient", lambda *a, **kw: client)
        node = ClusterNode(node_id="p", host="p", port=1)
        node.base_url = "http://p:1"  # type: ignore[attr-defined]
        relay = await forward_to_peer(node, "POST", "/v1/chat", stream=True)
        out = []
        async with relay as it:
            async for chunk in it:
                out.append(chunk)
        assert out == [b"x", b"y"]
        assert resp.closed and client.closed


class TestSingleProbeMode1066:
    """maybe_revive returns a dead backend for exactly one probe, not N."""

    def test_probing_field_exists(self):
        b = Backend(name="x", base_url="http://x", weight=1)
        assert hasattr(b, "probing")
        assert b.probing is False

    @pytest.mark.asyncio
    async def test_dead_backend_probed_once_then_skipped(self):
        router = ClusterRouter([Backend(name="a", base_url="http://a", weight=1)])
        b = router.get_backend("a")
        # drive it dead
        for _ in range(10):
            b.record_failure()
        assert not b.alive
        # advance cooldown: monkeypatch time.monotonic via last_failure_ts in past
        import time

        b.last_failure_ts = time.monotonic() - 999
        first = await router.select()
        assert first is b
        assert b.probing is True
        # second select BEFORE probe resolves — must skip (not re-probe)
        second = await router.select()
        assert second is None
        assert b.probing is True

    @pytest.mark.asyncio
    async def test_probe_success_revives(self):
        router = ClusterRouter([Backend(name="a", base_url="http://a", weight=1)])
        b = router.get_backend("a")
        for _ in range(10):
            b.record_failure()
        import time

        b.last_failure_ts = time.monotonic() - 999
        await router.select()
        assert b.probing is True
        b.record_success()
        assert b.probing is False
        assert b.alive is True

    @pytest.mark.asyncio
    async def test_probe_failure_restarts_cooldown(self):
        router = ClusterRouter([Backend(name="a", base_url="http://a", weight=1)])
        b = router.get_backend("a")
        for _ in range(10):
            b.record_failure()
        import time

        b.last_failure_ts = time.monotonic() - 999
        await router.select()
        assert b.probing is True
        before_ts = b.last_failure_ts
        b.record_failure()
        assert b.probing is False
        # cooldown clock restarted (last_failure_ts bumped)
        assert b.last_failure_ts >= before_ts
        # immediately after a failed probe, select() must skip (within cooldown)
        assert await router.select() is None

    @pytest.mark.asyncio
    async def test_within_cooldown_not_probed(self):
        router = ClusterRouter([Backend(name="a", base_url="http://a", weight=1)])
        b = router.get_backend("a")
        b.mark_dead()
        # just marked dead — within cooldown
        assert await router.select() is None
        assert b.probing is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
