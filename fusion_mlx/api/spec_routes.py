# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding routing introspection API for fusion-mlx.

Provides /v1/spec/routes endpoint for inspecting the spec-decode routing
table, and /v1/spec/resolve for simulating a routing decision given model
signals. Phase 3 observability surface.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..middleware.auth import verify_api_key
from ..speculative.auto_router import (
    _SPEC_ROUTING_TABLE,
    METHOD_DFLASH,
    METHOD_DFLY,
    METHOD_DSPARK,
    METHOD_MTP,
    METHOD_NGRAM,
    RouteSignals,
    SpecAutoRouter,
    routing_table,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/spec", tags=["spec-decode"])


class SpecRouteEntryResponse(BaseModel):
    family: str
    methods: list[str]
    constraints: list[str]
    drafter_map: dict[str, str] | None = None


class SpecRoutesResponse(BaseModel):
    routes: list[SpecRouteEntryResponse]
    method_names: dict[str, str]


class SpecResolveRequest(BaseModel):
    model_family: str | None = None
    has_mtp: bool = False
    is_moe: bool = False
    quant_bits: int | None = None
    is_recurrent: bool = False
    prompt_token_count: int = 0


class SpecResolveResponse(BaseModel):
    method: str
    model_family: str | None
    entry_family: str | None
    matched_entry: SpecRouteEntryResponse | None = None


_METHOD_DESCRIPTIONS = {
    METHOD_NGRAM: "n-gram suffix decoding (zero GPU cost, safe default)",
    METHOD_DFLASH: "DFlash block-diffusion drafter (operator-shipped)",
    METHOD_MTP: "model-native multi-token prediction (no extra model load)",
    METHOD_DSPARK: "DSpark drafter-based speculative decoding",
    METHOD_DFLY: "DFly block-parallel drafter (Hy3-native, hidden-state correction)",
}


@router.get("/routes", response_model=SpecRoutesResponse)
async def get_spec_routes(
    _auth: bool = Depends(verify_api_key),
) -> SpecRoutesResponse:
    merged: dict[str, SpecRouteEntryResponse] = {}
    for entry in routing_table():
        if entry.family in merged:
            prev = merged[entry.family]
            prev.methods.extend(m for m in entry.methods if m not in prev.methods)
            prev.constraints.extend(c for c in entry.constraints if c not in prev.constraints)
            if entry.drafter_map:
                if prev.drafter_map is None:
                    prev.drafter_map = {}
                prev.drafter_map.update(entry.drafter_map)
        else:
            merged[entry.family] = SpecRouteEntryResponse(
                family=entry.family,
                methods=list(entry.methods),
                constraints=list(entry.constraints),
                drafter_map=dict(entry.drafter_map) if entry.drafter_map else None,
            )
    return SpecRoutesResponse(
        routes=list(merged.values()),
        method_names=_METHOD_DESCRIPTIONS,
    )


@router.post("/resolve", response_model=SpecResolveResponse)
async def resolve_spec_route(
    req: SpecResolveRequest,
    _auth: bool = Depends(verify_api_key),
) -> SpecResolveResponse:
    router_obj = SpecAutoRouter()
    signals = RouteSignals(
        prompt_token_count=req.prompt_token_count,
        has_mtp=req.has_mtp,
        model_family=req.model_family,
        is_moe=req.is_moe,
        quant_bits=req.quant_bits,
        is_recurrent=req.is_recurrent,
    )
    method = router_obj.decide(signals)

    matched_entry = None
    entry_family = None
    if req.model_family is not None:
        merged_methods: list[str] = []
        merged_constraints: list[str] = []
        merged_drafter: dict[str, str] = {}
        seen_families: set[str] = set()
        for entry in _SPEC_ROUTING_TABLE:
            if entry.family not in (req.model_family, "*"):
                continue
            if not entry.evaluate_constraints(signals):
                continue
            if entry_family is not None and entry.family != entry_family:
                break
            if entry_family is None:
                entry_family = entry.family
            for m in entry.methods:
                if m not in merged_methods:
                    merged_methods.append(m)
            for c in entry.constraints:
                if c not in merged_constraints:
                    merged_constraints.append(c)
            if entry.drafter_map:
                merged_drafter.update(entry.drafter_map)
        if entry_family is not None:
            matched_entry = SpecRouteEntryResponse(
                family=entry_family,
                methods=merged_methods,
                constraints=merged_constraints,
                drafter_map=merged_drafter if merged_drafter else None,
            )

    logger.info(
        "spec-resolve: family=%s method=%s entry=%s",
        req.model_family,
        method,
        entry_family,
    )
    return SpecResolveResponse(
        method=method,
        model_family=req.model_family,
        entry_family=entry_family,
        matched_entry=matched_entry,
    )
