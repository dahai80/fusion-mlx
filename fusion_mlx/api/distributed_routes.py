# SPDX-License-Identifier: Apache-2.0
"""Distributed pipeline-parallelism HTTP endpoints (#621).

First-version surface for fusion-multi-node Pipeline Parallelism:

  POST /distributed/load_shard      — load model, register a layer-range shard
  POST /distributed/pipeline_step   — run forward over the shard's layers,
                                       receive/send activation tensors
  POST /distributed/decode          — apply final norm + lm_head to the last
                                       shard's hidden states, return token ids
  POST /distributed/sync_weights    — hot-update shard weights
  GET  /distributed/shards          — list registered shards (ops/debug)
  DELETE /distributed/shards/{id}   — drop a shard

Activation tensors travel base64-encoded (.npy, bit-exact per mlx dtype).
Transport-level framing/compression/encryption is the scheduler's job —
these endpoints only expose the forward step. See docs/distributed-pipeline.md.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..distributed.shard import ShardError, get_manager
from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/distributed", tags=["distributed"])


class LoadShardRequest(BaseModel):
    model_id: str = Field(..., description="repo id or local path of the LM")
    shard_index: int = Field(0, ge=0)
    layer_range: list[int] = Field(..., description="[start, end) layer indices")
    dtype: str | None = Field(None, description="optional dtype hint (informational)")


class LoadShardResponse(BaseModel):
    shard_id: str
    model_id: str
    shard_index: int
    layer_range: list[int]
    num_layers: int
    dtype: str | None = None


class PipelineStepRequest(BaseModel):
    shard_id: str
    hidden_states: str | None = Field(
        None, description="base64 .npy of incoming hidden states (None for first shard)"
    )
    input_ids: list[int] | None = Field(
        None,
        description="token ids for the first shard (required when hidden_states is None)",
    )
    position_ids: list[int] | None = Field(
        None, description="optional position ids (informational, first-version)"
    )


class PipelineStepResponse(BaseModel):
    hidden_states: str = Field(..., description="base64 .npy of outgoing hidden states")
    shape: list[int]
    dtype: str


class DecodeRequest(BaseModel):
    shard_id: str
    hidden_states: str = Field(
        ..., description="base64 .npy of the last shard's hidden states"
    )
    temperature: float | None = Field(
        None, description="sampling temperature; 0/None = greedy argmax"
    )
    top_p: float | None = Field(
        None, description="nucleus sampling top_p (with temp>0)"
    )
    return_logits: bool = Field(
        False, description="include base64 .npy logits in the response (bandwidth cost)"
    )


class DecodeResponse(BaseModel):
    token_ids: list[int] = Field(..., description="sampled token id per position")
    shape: list[int]
    dtype: str
    logits: str | None = Field(
        None, description="base64 .npy logits (if return_logits)"
    )
    logits_shape: list[int] | None = None
    logits_dtype: str | None = None


class SyncWeightsRequest(BaseModel):
    shard_id: str
    weights: str | None = Field(None, description="base64 .npz of {param_path: array}")
    manifest: dict | None = Field(
        None, description="forward-compat: {path: pull_url} (not yet fetched)"
    )


class SyncWeightsResponse(BaseModel):
    shard_id: str
    params_updated: int


class ShardInfo(BaseModel):
    shard_id: str
    model_id: str
    shard_index: int
    layer_range: list[int]
    dtype: str | None = None
    num_layers: int


class ShardsListResponse(BaseModel):
    shards: list[ShardInfo]


class DropShardResponse(BaseModel):
    shard_id: str
    dropped: bool


def _shard_error_response(exc: ShardError):
    # 400 for caller errors (bad range / payload), 404 for unknown shard,
    # 502 for model-load failure. Kept here so the route handlers stay flat.
    msg = str(exc)
    if msg.startswith("unknown shard_id"):
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=msg)
    if msg.startswith("failed to load model"):
        from fastapi import HTTPException

        raise HTTPException(status_code=502, detail=msg)
    from fastapi import HTTPException

    raise HTTPException(status_code=400, detail=msg)


@router.post("/load_shard", response_model=LoadShardResponse)
async def load_shard(
    req: LoadShardRequest,
    _auth: bool = Depends(verify_api_key),
) -> LoadShardResponse:
    try:
        info = get_manager().load_shard(
            req.model_id, req.shard_index, req.layer_range, req.dtype
        )
    except ShardError as exc:
        _shard_error_response(exc)
    return LoadShardResponse(**info)


@router.post("/pipeline_step", response_model=PipelineStepResponse)
async def pipeline_step(
    req: PipelineStepRequest,
    _auth: bool = Depends(verify_api_key),
) -> PipelineStepResponse:
    try:
        out = get_manager().pipeline_step(
            req.shard_id, req.hidden_states, req.input_ids, req.position_ids
        )
    except ShardError as exc:
        _shard_error_response(exc)
    return PipelineStepResponse(**out)


@router.post("/decode", response_model=DecodeResponse)
async def decode(
    req: DecodeRequest,
    _auth: bool = Depends(verify_api_key),
) -> DecodeResponse:
    try:
        out = get_manager().decode(
            req.shard_id,
            req.hidden_states,
            req.temperature,
            req.top_p,
            req.return_logits,
        )
    except ShardError as exc:
        _shard_error_response(exc)
    return DecodeResponse(**out)


@router.post("/sync_weights", response_model=SyncWeightsResponse)
async def sync_weights(
    req: SyncWeightsRequest,
    _auth: bool = Depends(verify_api_key),
) -> SyncWeightsResponse:
    try:
        out = get_manager().sync_weights(req.shard_id, req.weights, req.manifest)
    except ShardError as exc:
        _shard_error_response(exc)
    return SyncWeightsResponse(**out)


@router.get("/shards", response_model=ShardsListResponse)
async def list_shards(
    _auth: bool = Depends(verify_api_key),
) -> ShardsListResponse:
    shards = [ShardInfo(**s) for s in get_manager().list_shards()]
    return ShardsListResponse(shards=shards)


@router.delete("/shards/{shard_id}", response_model=DropShardResponse)
async def drop_shard(
    shard_id: str,
    _auth: bool = Depends(verify_api_key),
) -> DropShardResponse:
    try:
        out = get_manager().drop_shard(shard_id)
    except ShardError as exc:
        _shard_error_response(exc)
    return DropShardResponse(**out)
