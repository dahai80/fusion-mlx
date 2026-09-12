# SPDX-License-Identifier: Apache-2.0
"""Weighted round-robin cluster router (L14 / PR-D11.1).

Supplements the least-loaded ``ClusterLoadBalancer`` with an explicit
nginx-style smooth weighted round-robin (SWRR) dispatcher for
single-host multi-instance ``/v1`` fan-out. Use case: N fusion-mlx
instances on different ports with unequal capacity (e.g. one on a
Max chip, one on a Pro chip) — route proportionally to declared
weights rather than only by transient load score.

Design:
- ``Backend`` — a single peer endpoint with a static ``weight`` and
  live health counters (failures, last_success_ts, alive).
- ``Snapshot`` — frozen view of all backends for ``/metrics`` and the
  ``/v1/cluster/route`` admin endpoint. No mutable refs leak out.
- ``ClusterRouter`` — SWRR ``select()`` (deterministic, no RNG) +
  ``dispatch()`` HTTP relay with bounded retries on a dead backend.
  Reuses ``forward_to_peer`` from ``peer_lb`` so the wire path is
  identical to the existing FailoverRouter relay.

This is OPT-IN and standalone: it does NOT replace
``ClusterLoadBalancer``/``FailoverRouter``. It coexists — operators
pick weighted routing by setting ``cluster_weights`` in ServerConfig.
With no weights set, the existing least-loaded path is unchanged.

SWRR algorithm (nginx): each backend carries a ``current_weight``
that starts at 0. On each selection, add ``weight`` to
``current_weight`` for every backend, pick the max, then subtract
``total_weight`` from the picked backend's ``current_weight``. This
distributes picks smoothly across backends proportional to weight
without clustering same-backend picks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Health-gating thresholds. A backend exceeding ``_MAX_FAILURES``
# consecutive failures is marked not-alive and skipped by ``select``
# until it recovers (a successful dispatch resets the failure count).
_MAX_FAILURES = 3
# Cooldown before a dead backend is retried (seconds). Prevents a
# flapping peer from being hammered every request.
_DEAD_COOLDOWN = 10.0


@dataclass
class Backend:
    name: str
    base_url: str
    weight: int = 1
    alive: bool = True
    failures: int = 0
    last_success_ts: float = 0.0
    last_failure_ts: float = 0.0
    # SWRR mutable state — NOT part of equality/hashing.
    current_weight: int = 0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(
                f"Backend {self.name!r}: weight must be > 0 (got {self.weight})"
            )

    def mark_dead(self) -> None:
        if self.alive:
            logger.warning(
                "cluster_router: backend %s marked dead (failures=%d)",
                self.name,
                self.failures,
            )
        self.alive = False
        self.last_failure_ts = time.monotonic()

    def record_success(self) -> None:
        self.failures = 0
        self.alive = True
        self.last_success_ts = time.monotonic()

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_ts = time.monotonic()
        if self.failures >= _MAX_FAILURES:
            self.mark_dead()

    def maybe_revive(self) -> bool:
        # A dead backend past its cooldown is eligible for one probe
        # request. ``select`` returns it; if the probe succeeds it
        # revives, if it fails the cooldown restarts.
        if self.alive:
            return True
        if time.monotonic() - self.last_failure_ts >= _DEAD_COOLDOWN:
            logger.info(
                "cluster_router: backend %s cooldown expired — probing",
                self.name,
            )
            return True
        return False


@dataclass(frozen=True)
class Snapshot:
    # Frozen metrics view — safe to hand to /metrics or admin endpoints
    # without leaking mutable backend state.
    name: str
    base_url: str
    weight: int
    alive: bool
    failures: int
    last_success_ts: float
    last_failure_ts: float

    @classmethod
    def from_backend(cls, b: Backend) -> Snapshot:
        return cls(
            name=b.name,
            base_url=b.base_url,
            weight=b.weight,
            alive=b.alive,
            failures=b.failures,
            last_success_ts=b.last_success_ts,
            last_failure_ts=b.last_failure_ts,
        )


class ClusterRouter:
    """Smooth weighted round-robin dispatcher over a fixed backend set.

    Backends are registered once (typically from ``cluster_peers`` +
    ``cluster_weights`` at boot). ``select()`` is deterministic SWRR
    over the alive set. ``dispatch()`` relays an HTTP request to the
    selected backend via ``forward_to_peer`` and updates health on
    success/failure.
    """

    def __init__(self, backends: list[Backend] | None = None) -> None:
        self._backends: dict[str, Backend] = {}
        self._lock = asyncio.Lock()
        self._rr_total: int = 0
        for b in backends or []:
            self.add_backend(b)
        logger.info(
            "cluster_router: initialized with %d backend(s): %s",
            len(self._backends),
            ", ".join(f"{b.name}(w={b.weight})" for b in self._backends.values())
            or "(none)",
        )

    def add_backend(self, backend: Backend) -> None:
        if backend.name in self._backends:
            logger.warning(
                "cluster_router: backend %s already registered — replacing",
                backend.name,
            )
        self._backends[backend.name] = backend
        self._rr_total = sum(b.weight for b in self._backends.values())

    def remove_backend(self, name: str) -> bool:
        removed = self._backends.pop(name, None)
        if removed:
            self._rr_total = sum(b.weight for b in self._backends.values())
            logger.info("cluster_router: removed backend %s", name)
        return removed is not None

    def get_backend(self, name: str) -> Backend | None:
        return self._backends.get(name)

    def backends(self) -> list[Backend]:
        return list(self._backends.values())

    async def select(self) -> Backend | None:
        # SWRR over the eligible (alive or cooldown-expired) backends.
        # Deterministic — no RNG. Returns None only if no backends
        # registered or all dead and within cooldown.
        async with self._lock:
            eligible = [b for b in self._backends.values() if b.maybe_revive()]
            if not eligible:
                return None
            if len(eligible) == 1:
                return eligible[0]
            best: Backend | None = None
            for b in eligible:
                b.current_weight += b.weight
                if best is None or b.current_weight > best.current_weight:
                    best = b
            assert best is not None
            best.current_weight -= self._rr_total
            return best

    async def dispatch(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        stream: bool = False,
        timeout: float = 120.0,
        api_key: str | None = None,
        max_attempts: int = 3,
    ) -> Any:
        # Relay to the SWRR-selected backend. On failure, record it and
        # retry on the next selected backend up to ``max_attempts``.
        # Streaming requests are NOT retried past the first backend
        # that accepts the connection (partial output already left) —
        # the error surfaces to the caller per OpenAI streaming semantics.
        from .peer_lb import forward_to_peer
        from .registry import NodeUnavailableError, PartialStreamError

        last_error: Exception | None = None
        attempted: set[str] = set()
        for attempt in range(max_attempts):
            backend = await self.select()
            if backend is None:
                logger.error(
                    "cluster_router: no eligible backend after %d attempt(s)",
                    attempt,
                )
                break
            if backend.name in attempted and not stream:
                # Already tried every backend once — stop, don't loop.
                logger.warning(
                    "cluster_router: exhausted backends after %d attempt(s)",
                    attempt,
                )
                break
            attempted.add(backend.name)
            logger.debug(
                "cluster_router: dispatch %s %s -> %s (attempt %d/%d)",
                method,
                path,
                backend.name,
                attempt + 1,
                max_attempts,
            )
            node = _BackendNodeAdapter(backend)
            try:
                result = await forward_to_peer(
                    node,
                    method,
                    path,
                    headers=headers,
                    body=body,
                    stream=stream,
                    timeout=timeout,
                    api_key=api_key,
                )
                backend.record_success()
                return result
            except PartialStreamError as exc:
                # Stream already delivered bytes — do not retry.
                backend.record_failure()
                raise
            except NodeUnavailableError as exc:
                backend.record_failure()
                last_error = exc
                logger.warning(
                    "cluster_router: backend %s failed (attempt %d/%d): %s",
                    backend.name,
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                continue
            except Exception as exc:
                backend.record_failure()
                last_error = exc
                logger.error(
                    "cluster_router: backend %s unexpected error: %s",
                    backend.name,
                    exc,
                    exc_info=True,
                )
                continue
        if last_error is not None:
            raise last_error
        raise NodeUnavailableError("cluster_router", "no eligible backend for dispatch")

    async def snapshot(self) -> list[Snapshot]:
        async with self._lock:
            return [Snapshot.from_backend(b) for b in self._backends.values()]


@dataclass
class _BackendNodeAdapter:
    # Adapts a ``Backend`` to the shape ``forward_to_peer`` expects
    # (node_id / host / port / base_url). Avoids widening the
    # ``ClusterNode`` schema for a standalone-router backend.
    _backend: Backend

    @property
    def node_id(self) -> str:
        return self._backend.name

    @property
    def base_url(self) -> str:
        return self._backend.base_url

    @property
    def host(self) -> str:
        return self._parse_host()

    @property
    def port(self) -> int:
        return self._parse_port()

    def _parse_host(self) -> str:
        url = self._backend.base_url
        rest = url.split("://", 1)[1] if "://" in url else url
        rest = rest.split("/", 1)[0]
        if ":" in rest:
            return rest.rsplit(":", 1)[0]
        return rest

    def _parse_port(self) -> int:
        url = self._backend.base_url
        scheme = "http"
        rest = url
        if "://" in url:
            scheme, rest = url.split("://", 1)
        rest = rest.split("/", 1)[0]
        if ":" in rest:
            try:
                return int(rest.rsplit(":", 1)[1])
            except ValueError:
                pass
        return 80 if scheme == "http" else 443


# Module-level singleton — lazily built from ServerConfig at boot by
# ``bootstrap_weighted``. ``None`` means weighted routing is not active
# (no ``cluster_weights`` configured); the existing least-loaded path
# remains in effect.
_router: ClusterRouter | None = None


def get_router() -> ClusterRouter | None:
    return _router


def set_router(router: ClusterRouter | None) -> None:
    global _router
    _router = router


def build_backends_from_config(config: Any) -> list[Backend]:
    # Parse ``cluster_peers`` + ``cluster_weights`` into Backend list.
    # Peers without an explicit weight default to 1. Peers listed only
    # in ``cluster_weights`` (not in ``cluster_peers``) are skipped —
    # a weight without a peer is a config error, logged not raised.
    from .peer_lb import _parse_peer_url

    peers = list(getattr(config, "cluster_peers", None) or [])
    weights = dict(getattr(config, "cluster_weights", None) or {})
    backends: list[Backend] = []
    for peer in peers:
        try:
            base_url, host, port = _parse_peer_url(str(peer))
        except ValueError as exc:
            logger.warning("cluster_router: skipping invalid peer %r: %s", peer, exc)
            continue
        name = f"{host}:{port}"
        weight = int(weights.pop(name, 1))
        if weight <= 0:
            logger.warning(
                "cluster_router: peer %s weight %d <= 0 — defaulting to 1",
                name,
                weight,
            )
            weight = 1
        backends.append(Backend(name=name, base_url=base_url, weight=weight))
    for orphan in weights:
        logger.warning(
            "cluster_router: cluster_weights has %r but no matching peer — ignored",
            orphan,
        )
    return backends


async def bootstrap_weighted(config: Any) -> int:
    # Build + install the module-level ClusterRouter from config. Returns
    # the number of backends registered. Called from the server lifespan
    # when ``cluster_lb_enabled`` is True AND ``cluster_weights`` is
    # non-empty. With no weights, weighted routing stays inactive and
    # the existing least-loaded LB handles selection.
    global _router
    weights = getattr(config, "cluster_weights", None) or {}
    if not weights:
        _router = None
        return 0
    backends = build_backends_from_config(config)
    if not backends:
        logger.warning(
            "cluster_router: cluster_weights set but no valid peers — "
            "weighted routing inactive"
        )
        _router = None
        return 0
    _router = ClusterRouter(backends)
    logger.info("cluster_router: activated with %d weighted backend(s)", len(backends))
    return len(backends)
