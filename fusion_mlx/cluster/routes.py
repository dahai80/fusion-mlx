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
    return {
        "total": len(nodes),
        "alive": alive,
        "dead": dead,
        "evicted": evicted,
        "nodes": [n.snapshot() for n in nodes],
    }


@router.post("/register")
async def cluster_register(
    req: RegisterRequest,
    _auth: bool = Depends(verify_management_access),
) -> dict[str, Any]:
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
