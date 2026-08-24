# SPDX-License-Identifier: Apache-2.0
"""Shard manager + activation (de)serialization for pipeline parallelism (#621).

A ``ShardManager`` keeps loaded mlx_lm models in memory keyed by model_id and
hands out shard ids. Each shard runs a contiguous slice of the transformer
layers forward over hidden states handed from the previous shard. The first
shard embeds token ids; subsequent shards receive hidden states.

Serialization: ``mx.save`` writes the single array in numpy ``.npy`` format
to a temp path, base64-encode the bytes for HTTP transport, decode + ``mx.load``
on the other side. This preserves every mlx dtype bit-exactly (bfloat16
included — numpy cannot host bf16, so a np.array detour would lose it).
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import uuid
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

_NPY_SUFFIX = ".npy"

# Defense-in-depth ceilings on decoded payloads (the body-size middleware
# caps the raw HTTP body; these cap the decoded tensor/weights bytes so a
# path-traversal / oversized-payload caller can't exhaust disk/memory).
# Overridable via env. Activations: a 1x8192x8192 bf16 tensor ≈ 128 MiB;
# weights: a small LoRA is well under 1 GiB.
_MAX_ACTIVATION_BYTES = int(
    os.environ.get("FUSION_DIST_MAX_ACTIVATION_BYTES", 0) or 0
) or (256 * 1024 * 1024)
_MAX_WEIGHTS_BYTES = int(os.environ.get("FUSION_DIST_MAX_WEIGHTS_BYTES", 0) or 0) or (
    1024 * 1024 * 1024
)
_MAX_INPUT_IDS = int(os.environ.get("FUSION_DIST_MAX_INPUT_IDS", 0) or 0) or 8192


class ShardError(Exception):
    """Shard lifecycle / forward error surfaced as HTTP 4xx/5xx."""


def _allowed_model_roots() -> list[str]:
    # Mirrors fusion_mlx.model_aliases._allowed_model_dirs so distributed
    # load_shard uses the same confinement as the rest of the server.
    home = os.path.realpath(os.path.expanduser("~"))
    roots = [
        os.path.join(home, ".fusion-mlx", "models"),
        os.path.join(home, ".cache", "huggingface"),
    ]
    env_dir = os.environ.get("FUSION_MLX_MODEL_DIR", "").strip()
    if env_dir:
        roots.append(os.path.realpath(os.path.expanduser(env_dir)))
    cwd = os.path.realpath(os.getcwd())
    if cwd != "/":
        roots.append(cwd)
    return roots


def _resolve_model_path(model_id: str) -> str:
    """Resolve a model_id to a filesystem path confined to an allowed root.

    Bare repo ids (no '/', not absolute) resolve against the model dir.
    Absolute or relative paths are realpath'd and rejected unless they sit
    under one of the allowed roots — blocks path traversal (``..``) and
    arbitrary-file loads. Returns the confined path (str)."""
    if not model_id or not model_id.strip():
        raise ShardError("model_id is empty")
    mid = model_id.strip()
    # Bare repo id (e.g. "mlx-community/Llama-3.2-1B-Instruct-4bit" HAS a '/',
    # so only truly bare names with no separator qualify as model-dir lookup).
    if not os.path.isabs(mid) and "/" not in mid:
        env_dir = os.environ.get("FUSION_MLX_MODEL_DIR", "").strip()
        if env_dir:
            candidate = os.path.join(os.path.expanduser(env_dir), mid)
            if os.path.exists(candidate):
                resolved = os.path.realpath(candidate)
                if any(
                    Path(resolved).is_relative_to(Path(r))
                    for r in _allowed_model_roots()
                ):
                    return resolved
    # Anything else (HF repo id with '/', or a path) -> resolve + confine.
    resolved = os.path.realpath(os.path.expanduser(mid))
    roots = _allowed_model_roots()
    if not any(Path(resolved).is_relative_to(Path(r)) for r in roots):
        raise ShardError(
            f"model_id '{model_id}' resolves outside allowed model directories"
        )
    return resolved


def serialize_activation(arr: mx.array) -> str:
    """Serialize a single ``mx.array`` to a base64 string (bit-exact, dtype
    preserved via .npy). Round-trips with :func:`deserialize_activation`."""
    with tempfile.NamedTemporaryFile(suffix=_NPY_SUFFIX, delete=False) as fh:
        path = fh.name
    try:
        mx.save(path, arr)
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def deserialize_activation(b64: str) -> mx.array:
    """Inverse of :func:`serialize_activation`. Raises :class:`ShardError`
    on malformed input so the route returns a clean 400, not a 500."""
    try:
        raw = base64.b64decode(b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ShardError(f"activation base64 invalid: {exc}") from exc
    if not raw:
        raise ShardError("activation payload empty")
    if len(raw) > _MAX_ACTIVATION_BYTES:
        raise ShardError(
            f"activation payload {len(raw)} bytes exceeds cap {_MAX_ACTIVATION_BYTES}"
        )
    with tempfile.NamedTemporaryFile(suffix=_NPY_SUFFIX, delete=False) as fh:
        path = fh.name
        fh.write(raw)
    try:
        return mx.load(path)
    except Exception as exc:
        raise ShardError(f"activation .npy decode failed: {exc}") from exc
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class ShardManager:
    """Holds loaded models + shard registry. Single-process, single-node —
    the distributed scheduler spawns one fusion-mlx process per node, each
    owning its shard's model. First-version: no cross-process coordination."""

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._shards: dict[str, dict] = {}
        # (model_id, layer_range) -> shard_id, so repeated load_shard of the
        # same slice reuses the in-memory model (weights already on GPU).
        self._by_key: dict[tuple[str, tuple[int, int]], str] = {}

    def _load_model(self, model_id: str) -> object:
        if model_id in self._models:
            return self._models[model_id]
        import mlx_lm

        path = _resolve_model_path(model_id)
        logger.info("distributed: loading model %s from %s", model_id, path)
        try:
            model, _tokenizer = mlx_lm.load(path)
        except Exception as exc:
            raise ShardError(f"failed to load model {model_id}: {exc}") from exc
        self._models[model_id] = model
        return model

    def _num_layers(self, model: object) -> int:
        inner = getattr(model, "model", None)
        if inner is None or not hasattr(inner, "layers"):
            raise ShardError("model has no .model.layers — not a layer-slicable LM")
        return len(inner.layers)

    def load_shard(
        self,
        model_id: str,
        shard_index: int,
        layer_range: list[int],
        dtype: str | None = None,
    ) -> dict:
        """Load ``model_id`` and register a shard over ``layer_range``
        ``[start, end)``. Returns ``{"shard_id", "num_layers", "layer_range",
        "dtype"}``. Idempotent for the same (model_id, layer_range)."""
        if len(layer_range) != 2:
            raise ShardError("layer_range must be [start, end]")
        start, end = int(layer_range[0]), int(layer_range[1])
        if start < 0 or end <= start:
            raise ShardError(f"invalid layer_range [{start}, {end})")
        model = self._load_model(model_id)
        total = self._num_layers(model)
        if end > total:
            raise ShardError(f"layer_range end {end} > num_layers {total}")
        key = (model_id, (start, end))
        if key in self._by_key:
            sid = self._by_key[key]
            logger.info(
                "distributed: reuse shard %s for %s layers [%d,%d)",
                sid,
                model_id,
                start,
                end,
            )
            return self._shards[sid]
        shard_id = f"shard-{uuid.uuid4().hex[:12]}"
        self._shards[shard_id] = {
            "shard_id": shard_id,
            "model_id": model_id,
            "shard_index": shard_index,
            "layer_range": [start, end],
            "dtype": dtype,
            "num_layers": total,
        }
        self._by_key[key] = shard_id
        logger.info(
            "distributed: registered shard %s model=%s layers=[%d,%d) of %d",
            shard_id,
            model_id,
            start,
            end,
            total,
        )
        return self._shards[shard_id]

    def _get_shard(self, shard_id: str) -> dict:
        if shard_id not in self._shards:
            raise ShardError(f"unknown shard_id {shard_id}")
        return self._shards[shard_id]

    def pipeline_step(
        self,
        shard_id: str,
        hidden_states_b64: str | None,
        input_ids: list[int] | None,
        position_ids: list[int] | None,
    ) -> dict:
        """Run forward over the shard's ``layer_range``.

        First shard (``hidden_states_b64`` is None, ``input_ids`` given):
        embed the token ids, then run layers [start:end).
        Later shard (``hidden_states_b64`` given): deserialize the incoming
        hidden states, run layers [start:end).

        Returns ``{"hidden_states": <b64>, "shape": [...], "dtype": "..."}``
        for the next shard to consume."""
        shard = self._get_shard(shard_id)
        model = self._models[shard["model_id"]]
        inner = model.model
        start, end = shard["layer_range"]
        layers = inner.layers

        if hidden_states_b64:
            hidden = deserialize_activation(hidden_states_b64)
            logger.debug(
                "distributed: step %s received hidden shape=%s dtype=%s",
                shard_id,
                hidden.shape,
                hidden.dtype,
            )
        else:
            if not input_ids:
                raise ShardError("first shard needs input_ids when no hidden_states")
            if len(input_ids) > _MAX_INPUT_IDS:
                raise ShardError(
                    f"input_ids length {len(input_ids)} exceeds cap {_MAX_INPUT_IDS}"
                )
            ids = mx.array(input_ids, dtype=mx.int32)
            # mlx_lm embed_tokens expects shape (seq,) or (batch, seq).
            if ids.ndim == 1:
                hidden = inner.embed_tokens(ids[None, :])  # (1, seq, hidden)
            else:
                hidden = inner.embed_tokens(ids)
            logger.debug(
                "distributed: step %s embedded %d tokens -> %s",
                shard_id,
                len(input_ids),
                hidden.shape,
            )

        for i in range(start, end):
            hidden = layers[i](hidden)

        # Force eval so the returned buffer is materialized before the next
        # shard (cross-node) consumes it — lazy graphs don't serialize.
        mx.eval(hidden)
        return {
            "hidden_states": serialize_activation(hidden),
            "shape": list(hidden.shape),
            "dtype": str(hidden.dtype),
        }

    def sync_weights(
        self, shard_id: str, weights_b64: str | None, manifest: dict | None
    ) -> dict:
        """Hot-update a shard's model weights. First-version accepts a single
        serialized weight dict via ``weights_b64`` (a base64-encoded .npz of
        {param_path: array}). ``manifest`` (path → pull URL) is accepted for
        forward-compat but not yet fetched — the scheduler must inline the
        weights for now. Returns the updated param count."""
        shard = self._get_shard(shard_id)
        model = self._models[shard["model_id"]]
        if weights_b64 is None and manifest is None:
            raise ShardError("sync_weights needs weights_b64 or manifest")
        if weights_b64 is None:
            raise ShardError(
                "manifest pull not implemented in first version; inline weights_b64"
            )
        try:
            raw = base64.b64decode(weights_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ShardError(f"weights base64 invalid: {exc}") from exc
        if len(raw) > _MAX_WEIGHTS_BYTES:
            raise ShardError(
                f"weights payload {len(raw)} bytes exceeds cap {_MAX_WEIGHTS_BYTES}"
            )
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
            path = fh.name
            fh.write(raw)
        try:
            tree = mx.load(path)
        except Exception as exc:
            raise ShardError(f"weights .npz decode failed: {exc}") from exc
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if not isinstance(tree, dict) or not tree:
            raise ShardError("weights payload must be a non-empty {path: array} dict")
        model.load_weights(list(tree.items()), strict=False)
        updated = len(tree)
        logger.info(
            "distributed: synced %d weight params into shard %s", updated, shard_id
        )
        return {"shard_id": shard_id, "params_updated": updated}

    def drop_shard(self, shard_id: str) -> dict:
        """Release a shard's registration (model stays cached for other
        shards of the same model)."""
        shard = self._get_shard(shard_id)
        key = (shard["model_id"], tuple(shard["layer_range"]))
        self._by_key.pop(key, None)
        self._shards.pop(shard_id, None)
        logger.info("distributed: dropped shard %s", shard_id)
        return {"shard_id": shard_id, "dropped": True}

    def list_shards(self) -> list[dict]:
        return list(self._shards.values())


# Process-singleton — the routes module grabs this.
_manager: ShardManager | None = None


def get_manager() -> ShardManager:
    global _manager
    if _manager is None:
        _manager = ShardManager()
    return _manager
