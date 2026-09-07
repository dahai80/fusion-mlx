# SPDX-License-Identifier: Apache-2.0
# Tests for issue #811: config-driven multi-instance load balancing.
# Exercises peer URL parsing, peer bootstrap, health-monitor mark-dead,
# least-loaded selection, and the httpx failover relay — all headless
# (no real server; httpx is monkeypatched where a live peer is needed).

import asyncio
import types

import pytest

from fusion_mlx.cluster import peer_lb, registry
from fusion_mlx.cluster.registry import (
    ClusterLoadBalancer,
    ClusterNode,
    FailoverRouter,
    NodeUnavailableError,
    PartialStreamError,
    get_registry,
)


def test_parse_peer_url_host_port():
    base, host, port = peer_lb._parse_peer_url("127.0.0.1:11435")
    assert base == "http://127.0.0.1:11435"
    assert host == "127.0.0.1"
    assert port == 11435


def test_parse_peer_url_with_scheme():
    base, host, port = peer_lb._parse_peer_url("http://10.0.0.5:11436")
    assert base == "http://10.0.0.5:11436"
    assert port == 11436


def test_parse_peer_url_https_default_port():
    base, host, port = peer_lb._parse_peer_url("https://node.example.com")
    assert base == "https://node.example.com:443"
    assert port == 443


def test_parse_peer_url_strips_path():
    base, host, port = peer_lb._parse_peer_url("http://host:1234/v1/chat")
    assert base == "http://host:1234"
    assert port == 1234


def test_parse_peer_url_rejects_empty():
    with pytest.raises(ValueError):
        peer_lb._parse_peer_url("   ")
    with pytest.raises(ValueError):
        peer_lb._parse_peer_url("http://:1234")


def _fresh_registry():
    # Force a fresh singleton so tests don't leak peers across each other.
    registry._registry = None
    return get_registry()


async def test_bootstrap_registers_peers():
    reg = _fresh_registry()
    cfg = types.SimpleNamespace(
        cluster_peers=["127.0.0.1:11435", "http://127.0.0.1:11436"],
        platform="mac",
    )
    count = await peer_lb.bootstrap_peers(cfg)
    assert count == 2
    n1 = await reg.get("127.0.0.1:11435")
    n2 = await reg.get("127.0.0.1:11436")
    assert n1 is not None and n2 is not None
    assert n1.base_url == "http://127.0.0.1:11435"  # type: ignore[attr-defined]
    assert n2.base_url == "http://127.0.0.1:11436"  # type: ignore[attr-defined]
    assert n1.is_alive()


async def test_bootstrap_skips_invalid_peers():
    reg = _fresh_registry()
    cfg = types.SimpleNamespace(
        cluster_peers=["127.0.0.1:11435", "   ", "http://:1234"],
        platform="mac",
    )
    count = await peer_lb.bootstrap_peers(cfg)
    assert count == 1
    assert await reg.get("127.0.0.1:11435") is not None


async def test_bootstrap_no_peers_is_noop():
    reg = _fresh_registry()
    cfg = types.SimpleNamespace(cluster_peers=[], platform="mac")
    count = await peer_lb.bootstrap_peers(cfg)
    assert count == 0
    assert await reg.alive_nodes() == []


async def test_health_monitor_marks_dead_peer():
    # beat_fn always returns False -> node evicted after max_missed beats.
    reg = _fresh_registry()
    node = ClusterNode(node_id="dead:1", host="dead", port=1)
    node.base_url = "http://dead:1"  # type: ignore[attr-defined]
    await reg.register(node)

    async def always_fail(n):
        return False

    monitor = registry.ClusterHealthMonitor(
        registry=reg, beat_fn=always_fail, interval=0.01, max_missed=2
    )
    await monitor.start()
    # Wait long enough for >max_missed beats.
    await asyncio.sleep(0.1)
    await monitor.stop()
    alive = await reg.alive_nodes()
    assert all(n.node_id != "dead:1" for n in alive), "dead peer not evicted"


async def test_load_balancer_selects_least_loaded():
    reg = _fresh_registry()
    a = ClusterNode(node_id="a:1", host="a", port=1)
    b = ClusterNode(node_id="b:1", host="b", port=1)
    await reg.register(a)
    await reg.register(b)
    # Make 'a' busier than 'b'.
    await reg.update_load("a:1", active_requests=5, available_percent=50.0)
    await reg.update_load("b:1", active_requests=0, available_percent=100.0)
    lb = ClusterLoadBalancer(registry=reg)
    picked = await lb.select()
    assert picked is not None
    assert picked.node_id == "b:1", "should pick least-loaded peer"


async def test_load_balancer_returns_none_when_all_dead():
    reg = _fresh_registry()
    node = ClusterNode(node_id="only:1", host="only", port=1)
    await reg.register(node)
    await reg.mark_dead("only:1", "test")
    lb = ClusterLoadBalancer(registry=reg)
    assert await lb.select() is None


async def test_failover_router_retries_then_succeeds():
    # call_fn fails on first node, succeeds on second. Non-stream -> retries.
    reg = _fresh_registry()
    a = ClusterNode(node_id="a:1", host="a", port=1)
    b = ClusterNode(node_id="b:1", host="b", port=1)
    await reg.register(a)
    await reg.register(b)
    lb = ClusterLoadBalancer(registry=reg)
    calls = []

    async def call_fn(node):
        calls.append(node.node_id)
        if node.node_id == "a:1":
            raise NodeUnavailableError(node.node_id, "down")
        return "OK"

    router = FailoverRouter(registry=reg, lb=lb, max_retries=2)
    result = await router.route(call_fn, stream=False)
    assert result == "OK"
    assert "a:1" in calls and "b:1" in calls


async def test_failover_router_stream_does_not_retry():
    # Streaming: a failure detected DURING call_fn's await (before the gen is
    # handed back) is wrapped in PartialStreamError by the router and NOT
    # retried. This mirrors a relay that detects a broken peer mid-setup.
    reg = _fresh_registry()
    a = ClusterNode(node_id="a:1", host="a", port=1)
    b = ClusterNode(node_id="b:1", host="b", port=1)
    await reg.register(a)
    await reg.register(b)
    lb = ClusterLoadBalancer(registry=reg)
    attempts = []

    async def call_fn(node):
        attempts.append(node.node_id)
        # Simulate a peer that accepted the stream then died — surfaces as
        # a NodeUnavailableError from within the await, after one node tried.
        raise NodeUnavailableError(node.node_id, "stream peer died")

    router = FailoverRouter(registry=reg, lb=lb, max_retries=2)
    with pytest.raises(PartialStreamError):
        await router.route(call_fn, stream=True)
    # Stream path does NOT retry: only one peer should have been attempted.
    assert len(attempts) == 1, "stream failure must not retry across peers"


async def test_forward_to_peer_non_stream_success(monkeypatch):
    # Monkeypatch httpx.AsyncClient to a stub returning a canned response.
    reg = _fresh_registry()
    node = ClusterNode(node_id="peer:1", host="peer", port=1)
    node.base_url = "http://peer:1"  # type: ignore[attr-defined]
    await reg.register(node)

    class FakeResp:
        status_code = 200
        text = "hello"

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def request(self, method, url, **kw):
            assert url == "http://peer:1/v1/models"
            return FakeResp()

        async def aclose(self):
            pass

    import httpx as _real_httpx

    monkeypatch.setattr(_real_httpx, "AsyncClient", FakeClient)
    resp = await peer_lb.forward_to_peer(node, "GET", "/v1/models")
    assert resp.status_code == 200
    assert resp.text == "hello"


async def test_forward_to_peer_raises_on_connect_failure(monkeypatch):
    reg = _fresh_registry()
    node = ClusterNode(node_id="down:1", host="down", port=1)
    node.base_url = "http://down:1"  # type: ignore[attr-defined]
    await reg.register(node)

    import httpx as _real_httpx

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def request(self, *a, **kw):
            raise _real_httpx.ConnectError("connection refused")

        async def aclose(self):
            pass

    monkeypatch.setattr(_real_httpx, "AsyncClient", FakeClient)
    with pytest.raises(NodeUnavailableError):
        await peer_lb.forward_to_peer(node, "GET", "/v1/models")


async def test_forward_to_peer_stream_relays_bytes(monkeypatch):
    reg = _fresh_registry()
    node = ClusterNode(node_id="peer:1", host="peer", port=1)
    node.base_url = "http://peer:1"  # type: ignore[attr-defined]
    await reg.register(node)

    class FakeStreamResp:
        status_code = 200

        async def aiter_raw(self):
            for c in (b"chunk1", b"chunk2"):
                yield c

        async def aclose(self):
            pass

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def build_request(self, *a, **kw):
            return object()

        async def send(self, req, stream=False):
            return FakeStreamResp()

        async def aclose(self):
            pass

    import httpx as _real_httpx

    monkeypatch.setattr(_real_httpx, "AsyncClient", FakeClient)
    it = await peer_lb.forward_to_peer(node, "POST", "/v1/chat", stream=True)
    out = []
    async for chunk in it:
        out.append(chunk)
    assert out == [b"chunk1", b"chunk2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
