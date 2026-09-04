# SPDX-License-Identifier: Apache-2.0
"""Unit tests for cluster self-healing (#6): dead-node detect/evict,
dynamic least-loaded LB, mid-request failover, + /v1/cluster/health route."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.cluster.registry import (
    ClusterHealthMonitor,
    ClusterLoadBalancer,
    ClusterNode,
    FailoverRouter,
    NodeRegistry,
    NodeState,
    NodeUnavailableError,
    set_registry,
)
from fusion_mlx.cluster.routes import router as cluster_router
from fusion_mlx.middleware.auth import verify_management_access


def _node(
    node_id: str, active_requests: int = 0, available_percent: float = 100.0
) -> ClusterNode:
    return ClusterNode(
        node_id=node_id,
        host="127.0.0.1",
        port=8000,
        active_requests=active_requests,
        available_percent=available_percent,
    )


@pytest.fixture(autouse=True)
def fresh_registry():
    reg = NodeRegistry()
    set_registry(reg)
    yield reg
    set_registry(None)


class TestDeadNodeDetection:
    @pytest.mark.asyncio
    async def test_mark_dead_after_missed_beats(self, fresh_registry):
        await fresh_registry.register(_node("a"))
        beat_results = {"a": False}

        async def beat(node):
            return beat_results.get(node.node_id, False)

        monitor = ClusterHealthMonitor(fresh_registry, beat, interval=999, max_missed=3)
        for _ in range(3):
            await monitor.check_once()
        node = await fresh_registry.get("a")
        assert node.state == NodeState.EVICTED
        assert node.missed_beats >= 3

    @pytest.mark.asyncio
    async def test_alive_node_stays_alive(self, fresh_registry):
        await fresh_registry.register(_node("a"))

        async def beat(node):
            return True

        monitor = ClusterHealthMonitor(fresh_registry, beat, interval=999, max_missed=2)
        await monitor.check_once()
        node = await fresh_registry.get("a")
        assert node.is_alive()

    @pytest.mark.asyncio
    async def test_revived_node_rejoins(self, fresh_registry):
        await fresh_registry.register(_node("a"))
        beat_results = {"a": False}

        async def beat(node):
            return beat_results.get(node.node_id, False)

        monitor = ClusterHealthMonitor(fresh_registry, beat, interval=999, max_missed=5)
        await monitor.check_once()
        node = await fresh_registry.get("a")
        assert node.state == NodeState.DEAD
        # Node comes back online.
        beat_results["a"] = True
        await monitor.check_once()
        node = await fresh_registry.get("a")
        assert node.is_alive()
        assert node.missed_beats == 0


class TestEviction:
    @pytest.mark.asyncio
    async def test_evicted_excluded_from_alive(self, fresh_registry):
        await fresh_registry.register(_node("a"))
        await fresh_registry.register(_node("b"))
        await fresh_registry.evict("a", "manual")
        alive = await fresh_registry.alive_nodes()
        assert {n.node_id for n in alive} == {"b"}

    @pytest.mark.asyncio
    async def test_evict_unknown_returns_false(self, fresh_registry):
        assert await fresh_registry.evict("nope") is False

    @pytest.mark.asyncio
    async def test_evicted_sticky_until_reregister(self, fresh_registry):
        await fresh_registry.register(_node("a"))
        await fresh_registry.evict("a")
        # mark_alive must NOT revive an evicted node.
        assert await fresh_registry.mark_alive("a") is False
        node = await fresh_registry.get("a")
        assert node.state == NodeState.EVICTED
        # Re-register clears eviction.
        await fresh_registry.register(_node("a"))
        node = await fresh_registry.get("a")
        assert node.is_alive()


class TestLoadBalancer:
    @pytest.mark.asyncio
    async def test_picks_least_loaded(self, fresh_registry):
        await fresh_registry.register(_node("light", active_requests=0))
        await fresh_registry.register(_node("medium", active_requests=3))
        await fresh_registry.register(_node("heavy", active_requests=10))
        lb = ClusterLoadBalancer(fresh_registry)
        chosen = await lb.select()
        assert chosen.node_id == "light"

    @pytest.mark.asyncio
    async def test_excludes_dead(self, fresh_registry):
        await fresh_registry.register(_node("light", active_requests=0))
        await fresh_registry.register(_node("dead", active_requests=0))
        await fresh_registry.mark_dead("dead")
        lb = ClusterLoadBalancer(fresh_registry)
        chosen = await lb.select()
        assert chosen.node_id == "light"

    @pytest.mark.asyncio
    async def test_none_when_no_alive(self, fresh_registry):
        await fresh_registry.register(_node("a"))
        await fresh_registry.evict("a")
        lb = ClusterLoadBalancer(fresh_registry)
        assert await lb.select() is None

    @pytest.mark.asyncio
    async def test_round_robin_on_tie(self, fresh_registry):
        await fresh_registry.register(_node("a", active_requests=1))
        await fresh_registry.register(_node("b", active_requests=1))
        lb = ClusterLoadBalancer(fresh_registry)
        first = await lb.select()
        second = await lb.select()
        assert first.node_id != second.node_id
        assert {first.node_id, second.node_id} == {"a", "b"}


class TestFailover:
    @pytest.mark.asyncio
    async def test_nonstream_retries_on_dead_node(self, fresh_registry):
        await fresh_registry.register(_node("good", active_requests=0))
        await fresh_registry.register(_node("bad", active_requests=0))
        # Force LB to pick "bad" first by giving it lower load, then "good".
        await fresh_registry.update_load("bad", 0, 100.0)
        await fresh_registry.update_load("good", 1, 100.0)
        lb = ClusterLoadBalancer(fresh_registry)
        calls: list[str] = []

        async def call_fn(node):
            calls.append(node.node_id)
            if node.node_id == "bad":
                raise NodeUnavailableError("bad", "connection reset")
            return {"ok": True, "served_by": node.node_id}

        router = FailoverRouter(fresh_registry, lb, max_retries=2)
        result = await router.route(call_fn, stream=False)
        assert result["served_by"] == "good"
        assert "bad" in calls

    @pytest.mark.asyncio
    async def test_stream_not_retried(self, fresh_registry):
        await fresh_registry.register(_node("bad", active_requests=0))
        await fresh_registry.register(_node("good", active_requests=1))
        lb = ClusterLoadBalancer(fresh_registry)
        calls: list[str] = []

        async def call_fn(node):
            calls.append(node.node_id)
            raise NodeUnavailableError(node.node_id, "mid-stream death")

        router = FailoverRouter(fresh_registry, lb, max_retries=3)
        with pytest.raises(NodeUnavailableError):
            await router.route(call_fn, stream=True)
        # Only one node attempted — no duplication.
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self, fresh_registry):
        await fresh_registry.register(_node("only", active_requests=0))
        lb = ClusterLoadBalancer(fresh_registry)

        async def call_fn(node):
            raise NodeUnavailableError(node.node_id, "always down")

        router = FailoverRouter(fresh_registry, lb, max_retries=1)
        with pytest.raises(NodeUnavailableError):
            await router.route(call_fn, stream=False)

    @pytest.mark.asyncio
    async def test_no_peers_raises_local_unavailable(self, fresh_registry):
        lb = ClusterLoadBalancer(fresh_registry)

        async def call_fn(node):
            return None

        router = FailoverRouter(fresh_registry, lb, max_retries=2)
        with pytest.raises(NodeUnavailableError) as exc:
            await router.route(call_fn, stream=False)
        assert exc.value.node_id == "local"


@pytest.fixture
def app_client(fresh_registry):
    app = FastAPI()

    async def _fake_auth():
        return True

    app.dependency_overrides[verify_management_access] = _fake_auth
    app.include_router(cluster_router)
    return TestClient(app)


class TestClusterRoutes:
    def test_health_empty(self, app_client):
        r = app_client.get("/v1/cluster/health")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 0
        assert body["nodes"] == []

    def test_register_then_health(self, app_client):
        r = app_client.post(
            "/v1/cluster/register",
            json={
                "node_id": "peer-1",
                "host": "10.0.0.2",
                "port": 11434,
                "platform": "mac",
                "active_requests": 2,
            },
        )
        assert r.status_code == 200
        r = app_client.get("/v1/cluster/health")
        body = r.json()
        assert body["total"] == 1
        assert body["alive"] == 1
        assert body["nodes"][0]["node_id"] == "peer-1"
        assert body["nodes"][0]["state"] == "alive"

    def test_evict_then_health(self, app_client):
        app_client.post(
            "/v1/cluster/register",
            json={"node_id": "peer-1", "host": "10.0.0.2", "port": 11434},
        )
        r = app_client.post(
            "/v1/cluster/evict",
            json={"node_id": "peer-1", "reason": "draining"},
        )
        assert r.status_code == 200
        r = app_client.get("/v1/cluster/health")
        body = r.json()
        assert body["evicted"] == 1
        assert body["alive"] == 0
        assert body["nodes"][0]["last_error"] == "draining"

    def test_evict_unknown_404(self, app_client):
        r = app_client.post("/v1/cluster/evict", json={"node_id": "ghost"})
        assert r.status_code == 404
