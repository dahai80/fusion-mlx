# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1065.

#1065: FailoverRouter had a dead if/else (both branches mark_dead) and, with
no ClusterHealthMonitor wired, a DEAD node was never revived — one transient
network blip permanently kicked a peer out of routing.
Fix: deleted the duplicate branch; added passive revival (revive_cooled_nodes)
called at the top of route() when no monitor is present.
"""

from __future__ import annotations

import inspect
import time

import pytest

from fusion_mlx.cluster.registry import (
    ClusterLoadBalancer,
    ClusterNode,
    FailoverRouter,
    NodeRegistry,
    NodeState,
)


def _node(node_id: str = "a") -> ClusterNode:
    return ClusterNode(node_id=node_id, host="127.0.0.1", port=8000)


@pytest.fixture()
def registry():
    reg = NodeRegistry()
    reg._dead_node_ttl = 9999.0  # don't let prune interfere with revive tests
    return reg


class TestFailoverDeadRevive1065:
    """Dead if/else removed; passive revival works for no-monitor path."""

    def test_source_no_dead_if_else(self):
        source = inspect.getsource(FailoverRouter.route)
        # the pre-fix dead branch was:
        #   if self.monitor is not None:
        #       await self.registry.mark_dead(...)
        #   else:
        #       await self.registry.mark_dead(...)
        # post-fix: a single unconditional mark_dead + a comment citing #1065.
        assert "if self.monitor is not None:" not in source or (
            "revive_cooled_nodes" in source
        )
        # exactly one mark_dead call in the except handler now
        assert source.count("await self.registry.mark_dead") == 1

    @pytest.mark.asyncio
    async def test_revive_cooled_dead_node(self, registry):
        await registry.register(_node("a"))
        await registry.mark_dead("a", "blip")
        node = await registry.get("a")
        assert node.state == NodeState.DEAD
        # push past cooldown
        node.state_changed_at = time.time() - 61
        revived = await registry.revive_cooled_nodes(60.0)
        assert revived == 1
        node = await registry.get("a")
        assert node.state == NodeState.ALIVE
        assert node.state_changed_at == 0.0
        assert node.missed_beats == 0

    @pytest.mark.asyncio
    async def test_dead_within_cooldown_not_revived(self, registry):
        await registry.register(_node("b"))
        await registry.mark_dead("b", "blip")
        # just marked dead — within cooldown
        revived = await registry.revive_cooled_nodes(60.0)
        assert revived == 0
        node = await registry.get("b")
        assert node.state == NodeState.DEAD

    @pytest.mark.asyncio
    async def test_evicted_not_revived(self, registry):
        await registry.register(_node("c"))
        await registry.evict("c", "manual")
        node = await registry.get("c")
        node.state_changed_at = time.time() - 9999
        revived = await registry.revive_cooled_nodes(1.0)
        assert revived == 0
        node = await registry.get("c")
        assert node.state == NodeState.EVICTED

    @pytest.mark.asyncio
    async def test_cooldown_zero_disables(self, registry):
        await registry.register(_node("d"))
        await registry.mark_dead("d", "blip")
        node = await registry.get("d")
        node.state_changed_at = time.time() - 9999
        revived = await registry.revive_cooled_nodes(0.0)
        assert revived == 0
        assert (await registry.get("d")).state == NodeState.DEAD

    @pytest.mark.asyncio
    async def test_route_revives_before_select_no_monitor(self, registry):
        # A dead node that has cooled down should become selectable again when
        # route() runs without a monitor.
        await registry.register(_node("e"))
        await registry.mark_dead("e", "blip")
        node = await registry.get("e")
        node.state_changed_at = time.time() - 61
        lb = ClusterLoadBalancer(registry)
        router = FailoverRouter(
            registry, lb, monitor=None, max_retries=0, dead_cooldown=60.0
        )

        called_with: list[str] = []

        async def call_fn(n):
            called_with.append(n.node_id)
            return "ok"

        result = await router.route(call_fn)
        assert result == "ok"
        assert called_with == ["e"]
        assert (await registry.get("e")).state == NodeState.ALIVE

    @pytest.mark.asyncio
    async def test_default_dead_cooldown_30(self, monkeypatch):
        monkeypatch.delenv("FUSION_CLUSTER_DEAD_COOLDOWN", raising=False)
        reg = NodeRegistry()
        lb = ClusterLoadBalancer(reg)
        router = FailoverRouter(reg, lb, monitor=None)
        assert router.dead_cooldown == 30.0

    @pytest.mark.asyncio
    async def test_env_dead_cooldown_override(self, monkeypatch):
        monkeypatch.setenv("FUSION_CLUSTER_DEAD_COOLDOWN", "120")
        reg = NodeRegistry()
        lb = ClusterLoadBalancer(reg)
        router = FailoverRouter(reg, lb, monitor=None)
        assert router.dead_cooldown == 120.0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
