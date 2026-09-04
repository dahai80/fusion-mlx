# SPDX-License-Identifier: Apache-2.0
"""Cluster node registry + self-healing (dead-node detect/evict, dynamic
least-loaded load balancing, mid-request failover).

This node advertises itself via mDNS (mdns.py) and emits a load snapshot
(server._node_load_snapshot). Peers in a fusion-mlx cluster are discovered
out-of-band (fusion-gateway / fusion-multi-node) and registered here. This
module owns the in-repo health model:

- ``NodeRegistry`` holds peer ``ClusterNode`` entries keyed by node_id.
- ``ClusterHealthMonitor`` runs periodic heartbeats; a node missing
  ``max_missed`` consecutive beats is marked DEAD and evicted from the
  active routing set (no new requests route there). A revived node
  rejoins on the next successful beat.
- ``ClusterLoadBalancer`` picks the least-loaded ALIVE node for a new
  request (falls back to round-robin among alive peers, then to the local
  node when no peers are alive).
- ``FailoverRouter`` wraps a per-node call: if the chosen node dies
  mid-flight (raises ``NodeUnavailableError`` or is observed DEAD), the
  request is retried on the next healthy node. Non-streaming requests are
  retried idempotently up to ``max_retries``; streaming requests are NOT
  retried (would duplicate partial output) — the error surfaces per
  OpenAI semantics.

All state is process-local and lock-guarded; the gateway remains the
authoritative cross-cluster router. This is the in-repo self-heal layer
the gateway + multi-node project consume.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class NodeState(str, Enum):
    ALIVE = "alive"
    DEAD = "dead"
    EVICTED = "evicted"

    def __str__(self) -> str:
        return self.value


class NodeUnavailableError(Exception):
    """Raised by a per-node call when the node is unreachable mid-request.

    ``FailoverRouter`` catches this and retries on the next healthy node.
    Carries the node_id that failed so the monitor can mark it suspect.
    """

    def __init__(self, node_id: str, reason: str = ""):
        self.node_id = node_id
        self.reason = reason
        super().__init__(f"node {node_id} unavailable: {reason}")


@dataclass
class ClusterNode:
    node_id: str
    host: str
    port: int
    platform: str = "mac"
    state: NodeState = NodeState.ALIVE
    last_heartbeat: float = 0.0
    missed_beats: int = 0
    active_requests: int = 0
    available_percent: float = 100.0
    models_loaded: list[str] = field(default_factory=list)
    last_error: str = ""

    def is_alive(self) -> bool:
        return self.state == NodeState.ALIVE

    def load_score(self) -> float:
        # Lower is better. Active requests dominate; memory pressure breaks ties.
        return self.active_requests * 1000.0 + (100.0 - self.available_percent)

    def snapshot(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "platform": self.platform,
            "state": str(self.state),
            "last_heartbeat": round(self.last_heartbeat, 3),
            "missed_beats": self.missed_beats,
            "active_requests": self.active_requests,
            "available_percent": round(self.available_percent, 1),
            "models_loaded": list(self.models_loaded),
            "last_error": self.last_error,
        }


class NodeRegistry:
    """Thread/coroutine-safe registry of cluster peer nodes.

    The local node is NOT stored here — it is the fallback target when no
    alive peers are suitable. Peers are added/removed by the discovery
    layer (gateway handshake, mDNS browse) and health-checked by
    ``ClusterHealthMonitor``.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, ClusterNode] = {}
        self._lock = asyncio.Lock()

    async def register(self, node: ClusterNode) -> None:
        async with self._lock:
            existing = self._nodes.get(node.node_id)
            if existing is not None:
                # Re-register refreshes metadata but preserves liveness state
                # unless the node was EVICTED (manual eviction is sticky until
                # an explicit re-add).
                existing.host = node.host
                existing.port = node.port
                existing.platform = node.platform
                if existing.state == NodeState.EVICTED:
                    logger.info("cluster: re-adding evicted node %s", node.node_id)
                    existing.state = NodeState.ALIVE
                    existing.missed_beats = 0
                logger.debug("cluster: refreshed peer %s", node.node_id)
                return
            self._nodes[node.node_id] = node
            logger.info(
                "cluster: registered peer %s (%s:%d)",
                node.node_id,
                node.host,
                node.port,
            )

    async def remove(self, node_id: str) -> bool:
        async with self._lock:
            return self._nodes.pop(node_id, None) is not None

    async def evict(self, node_id: str, reason: str = "") -> bool:
        async with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return False
            node.state = NodeState.EVICTED
            node.last_error = reason or "manual eviction"
            logger.warning("cluster: evicted node %s (%s)", node_id, node.last_error)
            return True

    async def get(self, node_id: str) -> ClusterNode | None:
        async with self._lock:
            return self._nodes.get(node_id)

    async def alive_nodes(self) -> list[ClusterNode]:
        async with self._lock:
            return [n for n in self._nodes.values() if n.is_alive()]

    async def all_nodes(self) -> list[ClusterNode]:
        async with self._lock:
            return list(self._nodes.values())

    async def mark_dead(self, node_id: str, reason: str = "") -> bool:
        async with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return False
            if node.state == NodeState.EVICTED:
                return False
            node.state = NodeState.DEAD
            node.missed_beats += 1
            node.last_error = reason or f"missed {node.missed_beats} heartbeats"
            logger.warning(
                "cluster: node %s marked DEAD (%s)", node_id, node.last_error
            )
            return True

    async def mark_alive(
        self,
        node_id: str,
        active_requests: int = 0,
        available_percent: float = 100.0,
    ) -> bool:
        async with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return False
            if node.state == NodeState.EVICTED:
                return False
            was_dead = not node.is_alive()
            node.state = NodeState.ALIVE
            node.missed_beats = 0
            node.last_heartbeat = time.time()
            node.active_requests = active_requests
            node.available_percent = available_percent
            if was_dead:
                logger.info("cluster: node %s revived — rejoining active set", node_id)
            return True

    async def update_load(
        self, node_id: str, active_requests: int, available_percent: float
    ) -> bool:
        async with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return False
            node.active_requests = active_requests
            node.available_percent = available_percent
            return True


class ClusterHealthMonitor:
    """Periodic heartbeat checker that marks dead nodes and evicts them.

    ``beat_fn(node)`` is an awaitable returning True on a successful
    heartbeat (peer responded), False on failure. The monitor calls it on
    every alive node every ``interval`` seconds. After ``max_missed``
    consecutive failures the node is marked DEAD and evicted from the
    routing set. A later successful beat revives it.

    Run with ``await monitor.start()`` inside the server lifespan; stop
    with ``await monitor.stop()``.
    """

    def __init__(
        self,
        registry: NodeRegistry,
        beat_fn: Callable[[ClusterNode], Coroutine[Any, Any, bool]],
        interval: float = 5.0,
        max_missed: int = 3,
    ) -> None:
        self.registry = registry
        self.beat_fn = beat_fn
        self.interval = interval
        self.max_missed = max_missed
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopped.clear()
        self._task = asyncio.ensure_future(self._loop())
        logger.info(
            "cluster health monitor started (interval=%.1fs max_missed=%d)",
            self.interval,
            self.max_missed,
        )

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
            logger.info("cluster health monitor stopped")

    async def check_once(self) -> None:
        """Single heartbeat sweep — also used by tests for deterministic checks."""
        nodes = await self.registry.all_nodes()
        for node in nodes:
            if node.state == NodeState.EVICTED:
                continue
            try:
                ok = await self.beat_fn(node)
            except Exception as exc:
                logger.debug("cluster: heartbeat %s raised %s", node.node_id, exc)
                ok = False
            if ok:
                await self.registry.mark_alive(
                    node.node_id, node.active_requests, node.available_percent
                )
            else:
                await self.registry.mark_dead(node.node_id, "heartbeat failed")
                fresh = await self.registry.get(node.node_id)
                if (
                    fresh is not None
                    and fresh.missed_beats >= self.max_missed
                    and fresh.state != NodeState.EVICTED
                ):
                    await self.registry.evict(
                        node.node_id,
                        f"exceeded {self.max_missed} missed heartbeats",
                    )

    async def _loop(self) -> None:
        while not self._stopped.is_set():
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("cluster: health sweep failed", exc_info=True)
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self.interval)
            except (TimeoutError, asyncio.CancelledError):
                pass


class ClusterLoadBalancer:
    """Picks the least-loaded ALIVE node for a new request.

    Strategy: among alive nodes, choose the one with the lowest
    ``load_score`` (active requests dominate, memory pressure breaks
    ties). On a tie, round-robin via insertion order. Returns ``None`` to
    signal the caller should fall back to the LOCAL node.
    """

    def __init__(self, registry: NodeRegistry) -> None:
        self.registry = registry
        self._rr_index = 0

    async def select(self) -> ClusterNode | None:
        alive = await self.registry.alive_nodes()
        if not alive:
            return None
        alive.sort(key=lambda n: (n.load_score(), n.node_id))
        best_score = alive[0].load_score()
        tied = [n for n in alive if n.load_score() == best_score]
        if len(tied) == 1:
            return tied[0]
        self._rr_index %= len(tied)
        chosen = tied[self._rr_index]
        self._rr_index += 1
        return chosen


class FailoverRouter:
    """Wraps a per-node call with mid-request failover.

    ``call_fn(node)`` is an awaitable performing the request on the given
    node. If the chosen node raises ``NodeUnavailableError`` (or the
    monitor marks it DEAD during the call), the router retries on the
    next healthy node, up to ``max_retries``.

    Streaming requests MUST set ``stream=True`` — they are NOT retried
    (a partial stream already left the server; retrying would duplicate
    output). The original error surfaces for the caller to emit an
    OpenAI-style error chunk.
    """

    def __init__(
        self,
        registry: NodeRegistry,
        lb: ClusterLoadBalancer,
        monitor: ClusterHealthMonitor | None = None,
        max_retries: int = 2,
    ) -> None:
        self.registry = registry
        self.lb = lb
        self.monitor = monitor
        self.max_retries = max_retries

    async def route(
        self,
        call_fn: Callable[[ClusterNode], Coroutine[Any, Any, Any]],
        stream: bool = False,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            node = await self.lb.select()
            if node is None:
                # No alive peers — caller handles local fallback.
                raise NodeUnavailableError("local", "no alive cluster peers")
            try:
                result = await call_fn(node)
                return result
            except NodeUnavailableError as exc:
                last_error = exc
                logger.warning(
                    "cluster: node %s failed (attempt %d/%d): %s",
                    exc.node_id,
                    attempt + 1,
                    self.max_retries + 1,
                    exc.reason,
                )
                if self.monitor is not None:
                    await self.registry.mark_dead(exc.node_id, exc.reason)
                else:
                    # Without a monitor the router still excludes the failed
                    # node from the next selection by marking it dead locally.
                    await self.registry.mark_dead(exc.node_id, exc.reason)
                if stream:
                    logger.error(
                        "cluster: streaming request to %s failed — NOT retrying "
                        "(would duplicate output)",
                        exc.node_id,
                    )
                    raise
                continue
        if last_error is not None:
            raise last_error
        raise NodeUnavailableError("local", "exhausted retries")


_registry: NodeRegistry | None = None


def get_registry() -> NodeRegistry:
    global _registry
    if _registry is None:
        _registry = NodeRegistry()
    return _registry


def set_registry(registry: NodeRegistry | None) -> None:
    global _registry
    _registry = registry
