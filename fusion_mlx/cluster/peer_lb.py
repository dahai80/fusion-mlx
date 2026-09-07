# SPDX-License-Identifier: Apache-2.0
"""Config-driven multi-instance load balancing (issue #811).

Activates the dormant cluster self-heal layer (NodeRegistry +
ClusterHealthMonitor + ClusterLoadBalancer + FailoverRouter) for a
single-host, multi-port fusion-mlx deployment — the case fusion-autotest
needs (N instances behind one logical endpoint) without requiring a
separate fusion-gateway process.

Boot path:
- ``bootstrap_peers(config)`` parses ``cluster_peers`` (a list of
  ``host:port`` or full URLs) into ``ClusterNode`` entries and registers
  them in the process-local ``NodeRegistry``.
- ``start_health_monitor()`` wires ``ClusterHealthMonitor`` with an
  HTTP ``/health`` heartbeat — fixing audit CL-1 (the monitor existed
  but was never started, so peers stayed ALIVE forever).
- ``forward_to_peer(node, method, path, ...)``` is the per-node
  ``call_fn`` the ``FailoverRouter`` drives: an httpx relay to the peer.
  Non-streaming requests retry on the next healthy peer; streaming
  requests do NOT retry (PartialStreamError).

This module is OPT-IN: it does nothing unless ``cluster_lb_enabled`` is
set in ``ServerConfig``. With it off, fusion-mlx behaves exactly as
before (single instance, no peer registry, no monitor). The gateway
remains the authoritative cross-cluster router for multi-host topologies
(see registry.py module docstring); this layer serves the single-host
multi-instance case the gateway is not needed for.
"""

from __future__ import annotations

import logging
import typing
from typing import Any

logger = logging.getLogger(__name__)


def _parse_peer_url(peer: str) -> tuple[str, str, int]:
    # Accept "host:port", "http://host:port", or a full URL. Returns
    # (base_url, host, port) where base_url has no trailing slash and no
    # path component (the relay path is supplied per-request).
    peer = peer.strip()
    if not peer:
        raise ValueError("empty peer string")
    scheme = "http"
    rest = peer
    if "://" in peer:
        scheme, rest = peer.split("://", 1)
    # Strip any path/query — peers are base endpoints.
    rest = rest.split("/", 1)[0]
    if ":" in rest:
        host, port_str = rest.rsplit(":", 1)
        port = int(port_str)
    else:
        host = rest
        port = 80 if scheme == "http" else 443
    if not host:
        raise ValueError(f"peer {peer!r} has no host")
    base_url = f"{scheme}://{host}:{port}"
    logger.debug(
        "parsed peer %r -> base=%s host=%s port=%d", peer, base_url, host, port
    )
    return base_url, host, port


async def bootstrap_peers(config: Any) -> int:
    # Register config.cluster_peers into the NodeRegistry. Returns the
    # number of peers registered. Idempotent: re-bootstrap refreshes load
    # snapshots without duplicating entries. Async because NodeRegistry
    # is lock-guarded; called from the server lifespan (running loop).
    from .registry import ClusterNode, get_registry

    peers = getattr(config, "cluster_peers", None) or []
    if not peers:
        logger.info("cluster_lb: no cluster_peers configured — skipping bootstrap")
        return 0
    registry = get_registry()
    count = 0
    for peer in peers:
        try:
            base_url, host, port = _parse_peer_url(str(peer))
        except ValueError as exc:
            logger.warning("cluster_lb: skipping invalid peer %r: %s", peer, exc)
            continue
        node_id = f"{host}:{port}"
        node = ClusterNode(
            node_id=node_id,
            host=host,
            port=port,
            platform=str(getattr(config, "platform", None) or "mac"),
        )
        # Stash base_url on the node for the forwarder (ClusterNode is a
        # dataclass; use a side attribute so we don't widen its schema).
        node.base_url = base_url  # type: ignore[attr-defined]
        await registry.register(node)
        count += 1
        logger.info("cluster_lb: registered peer %s (%s)", node_id, base_url)
    return count


async def _http_health_beat(node: Any) -> bool:
    # Heartbeat for ClusterHealthMonitor: GET {base_url}/health. Returns
    # True on any 2xx, False otherwise. Uses a short timeout so a dead
    # peer is flagged within one beat interval, not after a long hang.
    import httpx

    base_url = getattr(node, "base_url", None) or f"http://{node.host}:{node.port}"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{base_url}/health")
        ok = 200 <= resp.status_code < 300
        if ok:
            # Opportunistically refresh load from the peer's /health body
            # if it exposes active_requests / available_percent.
            try:
                data = resp.json()
                ar = int(data.get("active_requests", 0))
                avail = float(data.get("available_percent", 100.0))
                from .registry import get_registry

                await get_registry().update_load(node.node_id, ar, avail)
            except (ValueError, TypeError, KeyError):
                pass
        return ok
    except Exception as exc:
        logger.debug("cluster_lb: heartbeat %s failed: %s", node.node_id, exc)
        return False


async def start_health_monitor(interval: float = 5.0, max_missed: int = 3) -> Any:
    # Build + start ClusterHealthMonitor with an HTTP /health heartbeat.
    # Returns the monitor (caller stops it in lifespan shutdown). Fixes
    # audit CL-1: peers now get heartbeat-verified, not stale-ALIVE.
    from .registry import ClusterHealthMonitor, get_registry

    registry = get_registry()
    monitor = ClusterHealthMonitor(
        registry=registry,
        beat_fn=_http_health_beat,
        interval=interval,
        max_missed=max_missed,
    )
    await monitor.start()
    logger.info(
        "cluster_lb: health monitor started (interval=%.1fs max_missed=%d)",
        interval,
        max_missed,
    )
    return monitor


async def stop_health_monitor(monitor: Any) -> None:
    if monitor is None:
        return
    try:
        await monitor.stop()
    except Exception:
        logger.debug("cluster_lb: health monitor stop failed", exc_info=True)


async def forward_to_peer(
    node: Any,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    stream: bool = False,
    timeout: float = 120.0,
    api_key: str | None = None,
) -> Any:
    # The per-node call_fn for FailoverRouter. Relays an HTTP request to
    # a peer fusion-mlx instance via httpx. Raises NodeUnavailableError
    # on connect/read failure so the router retries (non-stream) or
    # surfaces PartialStreamError (stream). Returns the httpx Response
    # (non-stream) or an async byte iterator (stream).
    import httpx

    from .registry import NodeUnavailableError, PartialStreamError

    base_url = getattr(node, "base_url", None) or f"http://{node.host}:{node.port}"
    url = f"{base_url}{path}"
    fwd_headers = dict(headers or {})
    if api_key and "Authorization" not in fwd_headers:
        fwd_headers["Authorization"] = f"Bearer {api_key}"
    logger.debug("cluster_lb: forward %s %s -> %s", method, path, url)
    try:
        client = httpx.AsyncClient(timeout=timeout)
    except Exception as exc:
        raise NodeUnavailableError(node.node_id, f"httpx init failed: {exc}")
    try:
        if stream:
            req = client.build_request(method, url, headers=fwd_headers, content=body)
            resp = await client.send(req, stream=True)
            if resp.status_code >= 500:
                await resp.aclose()
                await client.aclose()
                raise NodeUnavailableError(
                    node.node_id, f"peer returned {resp.status_code}"
                )

            async def _relay() -> typing.AsyncIterator[bytes]:
                delivered = False
                try:
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            delivered = True
                            yield chunk
                except Exception as exc:
                    if delivered:
                        raise PartialStreamError(
                            node.node_id, f"stream broke: {exc}"
                        ) from exc
                    raise NodeUnavailableError(
                        node.node_id, f"stream failed before any output: {exc}"
                    ) from exc
                finally:
                    await resp.aclose()
                    await client.aclose()

            return _relay()
        resp = await client.request(method, url, headers=fwd_headers, content=body)
        await client.aclose()
        if resp.status_code >= 500:
            raise NodeUnavailableError(
                node.node_id, f"peer returned {resp.status_code}"
            )
        return resp
    except (NodeUnavailableError, PartialStreamError):
        raise
    except Exception as exc:
        try:
            await client.aclose()
        except Exception:
            pass
        raise NodeUnavailableError(node.node_id, f"{type(exc).__name__}: {exc}")
