# SPDX-License-Identifier: Apache-2.0
"""Cluster health + admin routes (self-healing observability surface).

GET  /v1/cluster/health   — all peer node states (alive/dead/evicted).
POST /v1/cluster/evict    — manually evict a node (sticky until re-registered).
POST /v1/cluster/register — register/refresh a peer node (discovery handshake).

Auth: management access (Bearer/x-api-key), same gate as /v1/status.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..middleware.auth import verify_management_access
from .registry import ClusterNode, NodeState, get_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/cluster", tags=["cluster"])


class RegisterRequest(BaseModel):
    node_id: str
    host: str
    port: int
    platform: str = "mac"
    active_requests: int = 0
    available_percent: float = 100.0
    models_loaded: list[str] = []
    # CL-4 (#811 audit 0906): cluster-shared-secret token proving the peer
    # belongs to this cluster. Verified against compute_cluster_token().
    cluster_token: str | None = None


class EvictRequest(BaseModel):
    node_id: str
    reason: str = ""


@router.get("/health")
async def cluster_health(
    _auth: bool = Depends(verify_management_access),
) -> dict[str, Any]:
    registry = get_registry()
    nodes = await registry.all_nodes()
    alive = sum(1 for n in nodes if n.is_alive())
    dead = sum(1 for n in nodes if n.state == NodeState.DEAD)
    evicted = sum(1 for n in nodes if n.state == NodeState.EVICTED)
    logger.debug(
        "cluster health: %d nodes (%d alive, %d dead, %d evicted)",
        len(nodes),
        alive,
        dead,
        evicted,
    )
    # CL-1 (#811 audit 0906): the health monitor is not wired into the server
    # lifespan in this release, so ALIVE here is the *registered* state, not a
    # heartbeat-verified state. Fail visibly: when peers are present but the
    # monitor never ran, every node stays ALIVE forever and a dead peer is
    # never evicted — operators routing on this would hit silent dead-node
    # failures. Surface a loud warning instead of lying.
    monitor_active = registry.health_monitor_active
    health_warning = ""
    if nodes and not monitor_active:
        health_warning = (
            "cluster self-healing monitor is NOT running: node liveness is "
            "unverified (registered state, not heartbeat-verified). A dead "
            "peer will NOT be detected or evicted. Do not route production "
            "traffic on this view. See audit CL-1."
        )
        logger.warning("cluster /health: %s", health_warning)
    return {
        "total": len(nodes),
        "alive": alive,
        "dead": dead,
        "evicted": evicted,
        "health_monitor_active": monitor_active,
        "health_warning": health_warning,
        "nodes": [n.snapshot() for n in nodes],
    }


@router.post("/register")
async def cluster_register(
    req: RegisterRequest,
    _auth: bool = Depends(verify_management_access),
) -> dict[str, Any]:
    # CL-4 (#811 audit 0906): authenticate the registering peer with the
    # cluster-shared secret. A rogue host on the subnet could otherwise
    # register a fake node (via the gateway or directly) and have prompts
    # routed to it. Fail-closed: no secret configured -> reject all peers;
    # token mismatch -> 403.
    from .mdns import verify_cluster_token

    if not verify_cluster_token(req.cluster_token):
        logger.warning(
            "cluster route: REJECTED register node=%s — invalid/absent "
            "cluster_token (CL-4 #811 audit 0906)",
            req.node_id,
        )
        raise HTTPException(
            status_code=403,
            detail="invalid cluster_token — peer not authenticated to this cluster",
        )
    registry = get_registry()
    node = ClusterNode(
        node_id=req.node_id,
        host=req.host,
        port=req.port,
        platform=req.platform,
        active_requests=req.active_requests,
        available_percent=req.available_percent,
        models_loaded=list(req.models_loaded),
    )
    await registry.register(node)
    logger.info("cluster route: registered node %s", req.node_id)
    return {"status": "registered", "node_id": req.node_id}


@router.post("/evict")
async def cluster_evict(
    req: EvictRequest,
    _auth: bool = Depends(verify_management_access),
) -> dict[str, Any]:
    registry = get_registry()
    ok = await registry.evict(req.node_id, req.reason)
    if not ok:
        logger.warning("cluster route: evict node %s not found", req.node_id)
        raise HTTPException(status_code=404, detail=f"node not found: {req.node_id}")
    return {"status": "evicted", "node_id": req.node_id}


@router.get("/route")
async def cluster_route_snapshot(
    _auth: bool = Depends(verify_management_access),
) -> dict[str, Any]:
    # PR-D11.1 (L14): weighted router observability. Returns the SWRR
    # backend set + health counters. When weighted routing is inactive
    # (no cluster_weights configured), returns inactive=true so operators
    # know the least-loaded LB — not the weighted router — is selecting.
    from .router import get_router

    router = get_router()
    if router is None:
        return {
            "active": False,
            "total": 0,
            "backends": [],
            "note": "weighted routing inactive (no cluster_weights configured)",
        }
    snaps = await router.snapshot()
    return {
        "active": True,
        "total": len(snaps),
        "backends": [
            {
                "name": s.name,
                "base_url": s.base_url,
                "weight": s.weight,
                "alive": s.alive,
                "failures": s.failures,
                "last_success_ts": s.last_success_ts,
                "last_failure_ts": s.last_failure_ts,
            }
            for s in snaps
        ],
    }
