# SPDX-License-Identifier: Apache-2.0
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


class TokenizeRequest(BaseModel):
    model: str = Field(..., description="model id or alias")
    text: str = Field(..., description="text to tokenize")
    add_special_tokens: bool = Field(True, description="prepend/append special tokens")


class TokenizeResponse(BaseModel):
    model: str
    tokens: list[int]
    count: int


class DetokenizeRequest(BaseModel):
    model: str = Field(..., description="model id or alias")
    tokens: list[int] = Field(..., description="token ids to decode")
    skip_special_tokens: bool = Field(
        True, description="strip special tokens from output"
    )


class DetokenizeResponse(BaseModel):
    model: str
    text: str


def _resolve_tokenizer(model_id: str):
    from ..server import _server_state

    pool = _server_state.get("engine_pool")
    if pool is None:
        raise HTTPException(503, "engine pool not initialized")
    entry = pool.get_entry(model_id)
    if entry is None:
        aliases = pool.get_model_ids()
        raise HTTPException(404, f"model '{model_id}' not found; known: {aliases[:10]}")
    engine = entry.engine
    if engine is None:
        raise HTTPException(409, f"model '{model_id}' discovered but not loaded")
    tok = getattr(engine, "tokenizer", None)
    if tok is None:
        tok = getattr(engine, "_tokenizer", None)
    if tok is None:
        raise HTTPException(500, f"model '{model_id}' engine has no tokenizer")
    return tok


@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest, _auth: bool = Depends(verify_api_key)):
    tok = _resolve_tokenizer(req.model)
    try:
        ids = tok.encode(req.text, add_special_tokens=req.add_special_tokens)
    except TypeError:
        ids = tok.encode(req.text)
    ids = list(ids)
    logger.info(
        "tokenize model=%s in_len=%d out_tokens=%d", req.model, len(req.text), len(ids)
    )
    return TokenizeResponse(model=req.model, tokens=ids, count=len(ids))


@router.post("/detokenize", response_model=DetokenizeResponse)
async def detokenize(req: DetokenizeRequest, _auth: bool = Depends(verify_api_key)):
    tok = _resolve_tokenizer(req.model)
    try:
        text = tok.decode(req.tokens, skip_special_tokens=req.skip_special_tokens)
    except TypeError:
        text = tok.decode(req.tokens)
    logger.info(
        "detokenize model=%s in_tokens=%d out_len=%d",
        req.model,
        len(req.tokens),
        len(text),
    )
    return DetokenizeResponse(model=req.model, text=text)


@router.get("/props")
async def props(_auth: bool = Depends(verify_api_key)):
    from ..config import get_config
    from ..server import _server_state

    cfg = get_config()
    pool = _server_state.get("engine_pool")
    sched = getattr(cfg, "scheduler", None)
    mem = getattr(cfg, "memory", None)
    import os

    mtp_chain_k = int(os.environ.get("FUSION_MLX_MTP_CHAIN_K", "2"))
    loaded = pool.get_loaded_model_ids() if pool else []
    settings = {
        "version": _safe_version(),
        "host": getattr(cfg, "bind_host", "127.0.0.1"),
        "port": getattr(cfg, "bind_port", 11434),
        "loaded_models": loaded,
        "loaded_model_count": len(loaded),
        "memory_tier": (
            getattr(mem, "tier", None).value if getattr(mem, "tier", None) else None
        ),
        "scheduler_policy": (
            getattr(sched, "policy", None).value
            if getattr(sched, "policy", None)
            else None
        ),
        "max_num_seqs": getattr(sched, "max_num_seqs", None),
        "max_num_batched_tokens": getattr(sched, "max_num_batched_tokens", None),
        "prefill_batch_size": getattr(sched, "prefill_batch_size", None),
        "prefill_step_size": getattr(sched, "prefill_step_size", None),
        "chunked_prefill": getattr(sched, "chunked_prefill", None),
        "spec_decode_strategy": getattr(sched, "spec_decode", None),
        "spec_decode_enabled": getattr(sched, "spec_decode", "none") != "none",
        "enable_mtp": getattr(sched, "enable_mtp", None),
        "mtp_chain_k": mtp_chain_k,
        "mtp_sidecar": getattr(sched, "mtp_sidecar", None),
        "kv_cache_dtype": getattr(sched, "kv_cache_dtype", None),
        "kv_cache_quantization": getattr(sched, "kv_cache_quantization", None),
        "prefix_cache_enabled": getattr(sched, "enable_prefix_cache", None),
        "gpu_memory_utilization": getattr(sched, "gpu_memory_utilization", None),
        "model_dir": getattr(cfg, "model_dir", None),
    }
    logger.info(
        "props served: %d loaded, strategy=%s, chain_k=%s",
        len(loaded),
        settings["spec_decode_strategy"],
        mtp_chain_k,
    )
    return JSONResponse(settings)


@router.post("/v1/models/rescan")
async def rescan_models(_auth: bool = Depends(verify_api_key)):
    from ..config import get_config
    from ..server import _server_state

    cfg = get_config()
    pool = _server_state.get("engine_pool")
    if pool is None:
        raise HTTPException(
            503, "engine pool not initialized; multi-model mode required"
        )
    model_dirs = getattr(cfg, "model_dirs", None) or [getattr(cfg, "model_dir", None)]
    model_dirs = [d for d in model_dirs if d]
    if not model_dirs:
        raise HTTPException(409, "no model_dir configured to rescan")
    before = set(pool.get_model_ids())
    await pool.discover_models_async(model_dirs, pinned_models=None)
    after = set(pool.get_model_ids())
    added = sorted(after - before)
    logger.info(
        "rescan: dirs=%s before=%d after=%d added=%d",
        model_dirs,
        len(before),
        len(after),
        len(added),
    )
    return {
        "status": "ok",
        "models_before": len(before),
        "models_after": len(after),
        "added": added,
        "added_count": len(added),
        "all_models": sorted(after),
    }


def _safe_version() -> str:
    try:
        from .._version import __version__

        return __version__
    except Exception:
        return "unknown"
