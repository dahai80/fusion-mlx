"""Admin-plane activation readout + guarded single-tensor delta apply (#1751).

Issue #1751 (fusion-trainer#123 path 1): the trainer needs an engine-side
surface for per-layer hidden states and a guarded apply/rollback for a
single late-MLP-layer tensor delta. Policy (eval gates, anti-poisoning)
stays trainer-owned; this module only reads activations and applies or
reverts one tensor on the resident engine. 100% offline, admin-plane only.

Endpoints (all under /admin, require_admin):
- POST /api/fine-tune/activations   -> hidden states after layer L (full
  block, i.e. after its MLP) for a batch of conversations
- POST /api/fine-tune/apply-delta   -> guarded single-tensor delta apply
  with in-memory snapshot
- POST /api/fine-tune/rollback-delta -> revert one apply (or all for a model)
- GET  /api/fine-tune/delta-status  -> list applied deltas

Applied deltas are ephemeral: they live only in the resident engine's
in-memory weights and vanish on engine unload/reload. This is stated in
every response (`ephemeral: true`).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import threading
import time
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .auth import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin-fine-tune-activations"])

_pool: Any = None


def set_activation_context(pool: Any) -> None:
    global _pool
    _pool = pool


def _get_pool() -> Any:
    if _pool is None:
        raise HTTPException(status_code=503, detail="activation route not wired")
    return _pool


# Late-MLP-layer eligibility: only tensors under a transformer block's mlp
# subtree are eligible for delta apply. Late margin: layer_index must be
# >= num_layers - late_margin (trainer-configurable per request, default 4).
_MLP_PATH_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.[A-Za-z0-9_.]+$")

_MAX_DELTA_FILE_BYTES = 512 * 1024 * 1024


class ActivationRequest(BaseModel):
    model: str
    messages: list[list[dict[str, Any]]] | None = None
    input: list[str] | None = None
    layer: int = -4
    positions: str | list[int] = "all"
    add_generation_prompt: bool = False
    adapter_path: str | None = None


class ApplyDeltaRequest(BaseModel):
    model: str
    param_path: str
    delta_path: str | None = None
    delta_b64: str | None = None
    tensor_key: str | None = None
    scale: float = 1.0
    late_margin: int = 4
    comment: str = ""


class RollbackDeltaRequest(BaseModel):
    model: str
    apply_id: str | None = None
    all: bool = False


_DELTA_REGISTRY: dict[str, dict[str, Any]] = {}
_DELTA_LOCK = threading.Lock()


def _executor_for(engine: Any):
    core = getattr(engine, "_engine", None)
    inner = getattr(core, "engine", None)
    executor = getattr(inner, "_mlx_executor", None)
    if executor is None:
        raise HTTPException(
            status_code=500,
            detail="engine has no mlx executor; cannot run forward safely",
        )
    return executor


def _resolve_layer_index(layer: int, num_layers: int) -> int:
    idx = layer if layer >= 0 else num_layers + layer
    if idx < 0 or idx >= num_layers:
        raise HTTPException(
            status_code=400,
            detail=f"layer {layer} resolves to {idx}, out of range "
            f"[0, {num_layers - 1}] (num_layers={num_layers})",
        )
    return idx


def _pick_positions(request: ActivationRequest, num_tokens: int) -> list[int]:
    spec = request.positions
    if spec == "all":
        return list(range(num_tokens))
    if spec == "last":
        return [num_tokens - 1] if num_tokens > 0 else []
    if isinstance(spec, list):
        out = []
        for p in spec:
            q = p if p >= 0 else num_tokens + p
            if q < 0 or q >= num_tokens:
                raise HTTPException(
                    status_code=400,
                    detail=f"position {p} out of range [0, {num_tokens - 1}]",
                )
            out.append(q)
        return out
    raise HTTPException(
        status_code=400,
        detail="positions must be 'all', 'last', or a list of ints",
    )


def _encode_f32_b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode(
        "ascii"
    )


def _extract_lm(engine: Any) -> Any:
    if type(engine).__name__ not in ("BatchedEngine",):
        raise HTTPException(
            status_code=400,
            detail=f"unsupported engine type {type(engine).__name__}; "
            "activation readout requires a resident text LM (BatchedEngine)",
        )
    model = getattr(engine, "_model", None)
    tokenizer = getattr(engine, "tokenizer", None)
    inner = getattr(model, "model", None)
    if model is None or inner is None or not hasattr(inner, "layers"):
        raise HTTPException(
            status_code=400,
            detail="resident model does not expose a standard mlx_lm "
            "transformer (model.model.layers missing)",
        )
    return model, tokenizer


def _tokenize_batch(tokenizer: Any, request: ActivationRequest) -> list[list[int]]:
    try:
        if request.messages is not None:
            encoded = [
                tokenizer.apply_chat_template(
                    conv,
                    tokenize=True,
                    add_generation_prompt=request.add_generation_prompt,
                )
                for conv in request.messages
            ]
            return [list(e) for e in encoded]
        if request.input is not None:
            return [
                tokenizer.encode(text, add_special_tokens=True)
                for text in request.input
            ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"tokenization failed: {e}") from e
    raise HTTPException(
        status_code=400,
        detail="provide either 'messages' (batch of conversations) "
        "or 'input' (batch of raw texts)",
    )


def _forward_hidden_states(
    model: Any, tokens: list[int], layer_index: int
) -> np.ndarray:
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner = model.model
    h = inner.embed_tokens(mx.array([tokens]))
    mask = create_attention_mask(h)
    for block in inner.layers[: layer_index + 1]:
        h = block(h, mask=mask, cache=None)
    out = np.array(mx.evaluate(h)[0], dtype=np.float32)
    return out


async def _leased_engine(model_id: str, adapter_path: str | None) -> tuple[Any, Any]:
    pool = _get_pool()
    engine = await pool.get_engine(model_id, _lease=True, adapter_path=adapter_path)
    return engine, pool


@router.post("/api/fine-tune/activations")
async def readout_activations(
    request: ActivationRequest, is_admin: bool = Depends(require_admin)
):
    started = time.monotonic()
    model_id = request.model
    logger.info(
        "activations readout: model=%s layer=%s n_seqs=%s adapter=%s",
        model_id,
        request.layer,
        len(request.messages or request.input or []),
        request.adapter_path or "-",
    )
    engine, pool = await _leased_engine(model_id, request.adapter_path)
    try:
        model, tokenizer = _extract_lm(engine)
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="engine tokenizer missing")
        num_layers = len(model.model.layers)
        layer_index = _resolve_layer_index(request.layer, num_layers)
        token_batch = _tokenize_batch(tokenizer, request)
        executor = _executor_for(engine)
        loop = asyncio.get_running_loop()
        sequences = []
        for i, tokens in enumerate(token_batch):
            if not tokens:
                raise HTTPException(
                    status_code=400, detail=f"sequence {i} tokenized to zero tokens"
                )
            hidden = await loop.run_in_executor(
                executor, _forward_hidden_states, model, tokens, layer_index
            )
            positions = _pick_positions(request, hidden.shape[0])
            flat = hidden[positions]
            sequences.append(
                {
                    "batch_index": i,
                    "num_tokens": int(hidden.shape[0]),
                    "picked_positions": positions,
                    "token_ids": [int(tokens[p]) for p in positions],
                    "tokens": [tokenizer.decode([int(tokens[p])]) for p in positions],
                    "hidden_b64": _encode_f32_b64(flat),
                    "shape": [len(positions), int(hidden.shape[1])],
                }
            )
        logger.info(
            "activations readout done: model=%s layer=%d seqs=%d in=%.2fs",
            model_id,
            layer_index,
            len(sequences),
            time.monotonic() - started,
        )
        return {
            "model": model_id,
            "layer_requested": request.layer,
            "layer_index": layer_index,
            "num_layers": num_layers,
            "dtype": "float32",
            "encoding": "base64_f32_le",
            "ephemeral": False,
            "sequences": sequences,
        }
    finally:
        try:
            await pool.release_engine(model_id, request.adapter_path)
        except Exception:
            logger.exception("activations readout: lease release failed")


def _validate_param_path(param_path: str, model: Any, late_margin: int) -> Any:
    match = _MLP_PATH_RE.match(param_path)
    if not match:
        raise HTTPException(
            status_code=400,
            detail=f"param_path {param_path!r} is not an MLP-layer tensor; "
            "expected model.layers.<i>.mlp.<param> (fail-visible guard, #1751)",
        )
    layer_idx = int(match.group(1))
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.parameters()))
    if param_path not in params:
        raise HTTPException(
            status_code=400,
            detail=f"param_path {param_path!r} not found in model parameters",
        )
    num_layers = len(model.model.layers)
    if layer_idx >= num_layers:
        raise HTTPException(
            status_code=400,
            detail=f"layer index {layer_idx} >= num_layers {num_layers}",
        )
    if layer_idx < num_layers - late_margin:
        raise HTTPException(
            status_code=400,
            detail=f"layer {layer_idx} is not late (late_margin={late_margin}, "
            f"num_layers={num_layers}); only layers >= "
            f"{num_layers - late_margin} are eligible",
        )
    return params[param_path]


def _load_delta(request: ApplyDeltaRequest, target: Any) -> Any:
    import mlx.core as mx

    if request.delta_path:
        path = os.path.realpath(os.path.expanduser(request.delta_path))
        if not os.path.isfile(path):
            raise HTTPException(status_code=400, detail=f"delta file not found: {path}")
        size = os.path.getsize(path)
        if size > _MAX_DELTA_FILE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"delta file too large: {size} bytes "
                f"(cap {_MAX_DELTA_FILE_BYTES})",
            )
        try:
            data = mx.load(path)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"delta file load failed: {e}"
            ) from e
        if request.tensor_key:
            if request.tensor_key not in data:
                raise HTTPException(
                    status_code=400,
                    detail=f"tensor_key {request.tensor_key!r} not in delta file "
                    f"(keys: {sorted(data.keys())})",
                )
            delta = data[request.tensor_key]
        else:
            delta = next(iter(data.values()))
    elif request.delta_b64:
        try:
            raw = base64.b64decode(request.delta_b64, validate=True)
            expected = int(np.prod(target.shape)) * target.itemsize
            if len(raw) != expected:
                raise HTTPException(
                    status_code=400,
                    detail=f"delta_b64 decoded to {len(raw)} bytes, target tensor "
                    f"needs {expected} (shape {list(target.shape)})",
                )
            arr = np.frombuffer(raw, dtype=target.dtype)
            delta = mx.array(arr.reshape(target.shape))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"delta_b64 decode failed: {e}"
            ) from e
    else:
        raise HTTPException(status_code=400, detail="provide delta_path or delta_b64")
    if tuple(delta.shape) != tuple(target.shape):
        raise HTTPException(
            status_code=400,
            detail=f"delta shape {list(delta.shape)} != target shape "
            f"{list(target.shape)}; only same-shape single-tensor updates "
            "are accepted (#1751 guard)",
        )
    return delta.astype(target.dtype)


def _apply_on_engine(engine: Any, request: ApplyDeltaRequest) -> dict[str, Any]:
    import mlx.core as mx

    model, _ = _extract_lm(engine)
    target = _validate_param_path(request.param_path, model, request.late_margin)
    delta = _load_delta(request, target)
    if request.scale != 1.0:
        delta = delta * request.scale
    old_snapshot = mx.evaluate(target + 0)
    new_value = old_snapshot + delta
    model.load_weights([(request.param_path, new_value)])
    mx.eval(new_value)
    return {"old_snapshot": old_snapshot, "new_value": new_value, "delta": delta}


@router.post("/api/fine-tune/apply-delta")
async def apply_delta(
    request: ApplyDeltaRequest, is_admin: bool = Depends(require_admin)
):
    logger.info(
        "apply-delta: model=%s path=%s scale=%s source=%s comment=%r",
        request.model,
        request.param_path,
        request.scale,
        "file" if request.delta_path else "b64",
        request.comment,
    )
    engine, pool = await _leased_engine(request.model, None)
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            _executor_for(engine), _apply_on_engine, engine, request
        )
    finally:
        try:
            await pool.release_engine(request.model, None)
        except Exception:
            logger.exception("apply-delta: lease release failed")
    apply_id = uuid.uuid4().hex[:12]
    with _DELTA_LOCK:
        _DELTA_REGISTRY[apply_id] = {
            "apply_id": apply_id,
            "model": request.model,
            "param_path": request.param_path,
            "scale": request.scale,
            "comment": request.comment,
            "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "old_snapshot": result["old_snapshot"],
            "shape": list(result["old_snapshot"].shape),
            "ephemeral": True,
        }
    logger.info(
        "apply-delta applied: id=%s model=%s path=%s",
        apply_id,
        request.model,
        request.param_path,
    )
    return {
        "apply_id": apply_id,
        "model": request.model,
        "param_path": request.param_path,
        "scale": request.scale,
        "shape": list(result["old_snapshot"].shape),
        "ephemeral": True,
        "note": "delta lives in in-memory weights only; lost on engine "
        "unload/reload. Rollback via /api/fine-tune/rollback-delta.",
    }


def _rollback_on_engine(engine: Any, param_path: str, old_snapshot: Any) -> None:
    import mlx.core as mx

    model, _ = _extract_lm(engine)
    model.load_weights([(param_path, old_snapshot)])
    mx.eval(old_snapshot)


@router.post("/api/fine-tune/rollback-delta")
async def rollback_delta(
    request: RollbackDeltaRequest, is_admin: bool = Depends(require_admin)
):
    with _DELTA_LOCK:
        if request.all:
            targets = [
                rec for rec in _DELTA_REGISTRY.values() if rec["model"] == request.model
            ]
        elif request.apply_id:
            rec = _DELTA_REGISTRY.get(request.apply_id)
            if rec is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"apply_id {request.apply_id} not found",
                )
            if rec["model"] != request.model:
                raise HTTPException(
                    status_code=400,
                    detail=f"apply_id {request.apply_id} belongs to model "
                    f"{rec['model']!r}, not {request.model!r}",
                )
            targets = [rec]
        else:
            raise HTTPException(status_code=400, detail="provide apply_id or all=true")
        records = dict(_DELTA_REGISTRY)
    if not targets:
        raise HTTPException(
            status_code=404,
            detail=f"no applied deltas for model {request.model!r}",
        )
    engine, pool = await _leased_engine(request.model, None)
    rolled_back = []
    try:
        for rec in targets:
            await asyncio.get_running_loop().run_in_executor(
                _executor_for(engine),
                _rollback_on_engine,
                engine,
                rec["param_path"],
                rec["old_snapshot"],
            )
            rolled_back.append(rec["apply_id"])
            with _DELTA_LOCK:
                _DELTA_REGISTRY.pop(rec["apply_id"], None)
    finally:
        try:
            await pool.release_engine(request.model, None)
        except Exception:
            logger.exception("rollback-delta: lease release failed")
    logger.info("rollback-delta done: model=%s ids=%s", request.model, rolled_back)
    return {
        "model": request.model,
        "rolled_back": rolled_back,
        "remaining": [
            {"apply_id": r["apply_id"], "param_path": r["param_path"]}
            for r in records.values()
            if r["model"] == request.model and r["apply_id"] not in rolled_back
        ],
    }


@router.get("/api/fine-tune/delta-status")
async def delta_status(
    model: str | None = None, is_admin: bool = Depends(require_admin)
):
    with _DELTA_LOCK:
        records = list(_DELTA_REGISTRY.values())
    if model:
        records = [r for r in records if r["model"] == model]
    return {
        "model": model,
        "deltas": [
            {
                "apply_id": r["apply_id"],
                "model": r["model"],
                "param_path": r["param_path"],
                "scale": r["scale"],
                "comment": r["comment"],
                "applied_at": r["applied_at"],
                "shape": r["shape"],
                "ephemeral": True,
            }
            for r in records
        ],
    }
