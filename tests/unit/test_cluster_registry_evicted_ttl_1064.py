# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1064.

#1064: NodeRegistry.evict() only flipped state=EVICTED — the entry stayed in
_nodes forever (no TTL/expiry). Long-running gateway registries (and
/v1/cluster/health listings) grew monotonically with dead/evicted nodes.
Fix: DEAD/EVICTED nodes are pruned after FUSION_CLUSTER_NODE_TTL seconds
(default 300) on the list read paths; re-register re-adds a pruned node.
"""

from __future__ import annotations

import time

import pytest

from fusion_mlx.cluster.registry import (
    ClusterNode,
    NodeRegistry,
    NodeState,
)


def _node(node_id: str = "a") -> ClusterNode:
    return ClusterNode(node_id=node_id, host="127.0.0.1", port=8000)


@pytest.fixture()
def registry():
    reg = NodeRegistry()
    reg._dead_node_ttl = 60.0
    return reg


class TestRegistryEvictedTtl1064:
    """DEAD/EVICTED nodes are pruned after TTL; alive nodes are not."""

    @pytest.mark.asyncio
    async def test_evicted_node_pruned_after_ttl(self, registry):
        await registry.register(_node("a"))
        await registry.evict("a", "manual")
        node = await registry.get("a")
        assert node is not None
        assert node.state == NodeState.EVICTED
        # push the state-change clock into the past beyond TTL
        node.state_changed_at = time.time() - registry._dead_node_ttl - 1
        nodes = await registry.all_nodes()
        assert nodes == []
        assert await registry.get("a") is None

    @pytest.mark.asyncio
    async def test_dead_node_pruned_after_ttl(self, registry):
        await registry.register(_node("b"))
        await registry.mark_dead("b", "heartbeat failed")
        node = await registry.get("b")
        assert node is not None
        assert node.state == NodeState.DEAD
        node.state_changed_at = time.time() - registry._dead_node_ttl - 1
        alive = await registry.alive_nodes()
        assert alive == []
        assert await registry.get("b") is None

    @pytest.mark.asyncio
    async def test_alive_node_not_pruned(self, registry):
        await registry.register(_node("c"))
        nodes = await registry.all_nodes()
        assert len(nodes) == 1
        assert nodes[0].state == NodeState.ALIVE

    @pytest.mark.asyncio
    async def test_node_not_pruned_within_ttl(self, registry):
        await registry.register(_node("d"))
        await registry.evict("d", "manual")
        # state_changed_at is now (within TTL) — should still be present
        nodes = await registry.all_nodes()
        assert len(nodes) == 1
        assert nodes[0].state == NodeState.EVICTED

    @pytest.mark.asyncio
    async def test_reregister_readds_pruned_node(self, registry):
        await registry.register(_node("e"))
        await registry.evict("e", "manual")
        node = await registry.get("e")
        node.state_changed_at = time.time() - registry._dead_node_ttl - 1
        await registry.all_nodes()  # triggers prune
        assert await registry.get("e") is None
        # re-register brings it back fresh + alive
        await registry.register(_node("e"))
        node = await registry.get("e")
        assert node is not None
        assert node.state == NodeState.ALIVE

    @pytest.mark.asyncio
    async def test_ttl_zero_disables_prune(self, monkeypatch):
        monkeypatch.setenv("FUSION_CLUSTER_NODE_TTL", "0")
        reg = NodeRegistry()
        assert reg._dead_node_ttl == 0.0
        await reg.register(_node("f"))
        await reg.evict("f", "manual")
        node = await reg.get("f")
        node.state_changed_at = time.time() - 9999
        nodes = await reg.all_nodes()
        assert len(nodes) == 1

    @pytest.mark.asyncio
    async def test_mark_alive_resets_state_changed_at(self, registry):
        await registry.register(_node("g"))
        await reg_mark_dead_and_revive(registry, "g")
        node = await registry.get("g")
        assert node.state == NodeState.ALIVE
        assert node.state_changed_at == 0.0

    @pytest.mark.asyncio
    async def test_snapshot_includes_state_changed_at(self, registry):
        await registry.register(_node("h"))
        await registry.evict("h", "manual")
        node = await registry.get("h")
        snap = node.snapshot()
        assert "state_changed_at" in snap
        assert snap["state_changed_at"] > 0

    @pytest.mark.asyncio
    async def test_default_ttl_300(self, monkeypatch):
        monkeypatch.delenv("FUSION_CLUSTER_NODE_TTL", raising=False)
        reg = NodeRegistry()
        assert reg._dead_node_ttl == 300.0


async def reg_mark_dead_and_revive(reg: NodeRegistry, node_id: str):
    await reg.mark_dead(node_id, "down")
    await reg.mark_alive(node_id, 0, 100.0)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
