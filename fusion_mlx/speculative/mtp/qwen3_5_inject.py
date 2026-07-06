# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_inner_text_model(model: Any) -> Any:
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "args") and hasattr(lm, "model"):
        return lm

    if hasattr(model, "model") and hasattr(model, "args"):
        return model

    return None


def _detect_base_quantization(inner: Any) -> dict | None:
    try:
        from mlx.nn import QuantizedEmbedding, QuantizedLinear
    except ImportError:
        return None

    backbone = getattr(inner, "model", None)
    if backbone is None:
        return None

    for layer in getattr(backbone, "layers", []):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
            qp = layer.self_attn.q_proj
            if isinstance(qp, QuantizedLinear):
                return {
                    "bits": int(qp.bits),
                    "group_size": int(qp.group_size),
                }

    embed = getattr(backbone, "embed_tokens", None)
    if isinstance(embed, QuantizedEmbedding):
        return {
            "bits": int(embed.bits),
            "group_size": int(embed.group_size),
        }

    return None


def _resolve_sidecar_file(mtp_sidecar: str | Path) -> Path | None:
    if mtp_sidecar is None:
        return None

    path = Path(mtp_sidecar)
    if path.is_file():
        return path
    if path.is_dir():
        return _find_mtp_weights_file(path)

    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=str(mtp_sidecar))
        return _find_mtp_weights_file(Path(local))
    except Exception as exc:
        logger.warning(
            "[mtp.inject] could not resolve sidecar %r: %s",
            mtp_sidecar,
            exc,
        )
        return None


def _find_mtp_weights_file(sidecar_dir: Path) -> Path | None:
    candidates = (
        sidecar_dir / "model-mtp.safetensors",
        sidecar_dir / "model.safetensors",
    )
    for c in candidates:
        if c.exists():
            return c
    return None


def inject_mtp_support(
    model: Any,
    mtp_sidecar: str | Path | None = None,
    *,
    allow_random_init: bool = False,
) -> bool:
    import mlx.core as mx
    import mlx.nn as nn

    inner = _resolve_inner_text_model(model)
    if inner is None:
        logger.warning(
            "[mtp.inject] model %s has neither model.language_model nor "
            "(model + args); skipping MTP injection.",
            type(model).__name__,
        )
        return False

    args = inner.args

    num_mtp_layers = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
    if num_mtp_layers < 1:
        outer_args = getattr(model, "args", None)
        text_config = getattr(outer_args, "text_config", None) or {}
        if isinstance(text_config, dict):
            num_mtp_layers = int(text_config.get("mtp_num_hidden_layers", 0) or 0)
        if num_mtp_layers >= 1:
            try:
                object.__setattr__(args, "mtp_num_hidden_layers", num_mtp_layers)
            except (TypeError, AttributeError):
                pass

    if num_mtp_layers < 1:
        logger.info(
            "[mtp.inject] config has no mtp_num_hidden_layers; skipping MTP injection."
        )
        return False

    # --- Step 1: Build the MTP module ---
    from .head import build_mtp_module

    mtp = build_mtp_module(args, num_mtp_layers)
    logger.info(
        "[mtp.inject] Built MTP module (%d layer(s), hidden_size=%d).",
        num_mtp_layers,
        getattr(args, "hidden_size", -1),
    )

    # --- Step 2: Quantize MTP to match the base model's quantization ---
    quant_info = _detect_base_quantization(inner)
    if quant_info is not None:
        nn.quantize(
            mtp,
            group_size=quant_info["group_size"],
            bits=quant_info["bits"],
        )
        logger.info(
            "[mtp.inject] Quantized MTP: %d-bit, group_size=%d",
            quant_info["bits"],
            quant_info["group_size"],
        )

    # --- Step 3: Load MTP weights from sidecar safetensors ---
    if mtp_sidecar is not None:
        weights_file = _resolve_sidecar_file(mtp_sidecar)
        if weights_file is None:
            logger.warning(
                "[mtp.inject] sidecar %r could not be resolved to a "
                "safetensors file; skipping MTP injection. "
                "Pass either a repo id (mlx-community/Qwen3.5-9B-MTP-4bit), "
                "a directory containing model.safetensors / "
                "model-mtp.safetensors, or the file path directly.",
                mtp_sidecar,
            )
            return False
        raw = mx.load(str(weights_file))
        mtp_weights = {
            (k.removeprefix("mtp.") if k.startswith("mtp.") else k): v
            for k, v in raw.items()
        }
        from mlx.utils import tree_flatten

        expected_keys = {k for k, _ in tree_flatten(mtp.parameters())}
        loaded_keys = set(mtp_weights.keys())
        missing = expected_keys - loaded_keys
        if missing:
            logger.warning(
                "[mtp.inject] sidecar %s is missing %d required MTP "
                "tensor(s); refusing to ship a partially-random-init head. "
                "Missing keys (first 8): %s. "
                "Either grab a correctly-converted sidecar (e.g. "
                "mlx-community/Qwen3.5-9B-MTP-4bit) or regenerate via "
                "the add_mtp_weights.py converter.",
                weights_file.name,
                len(missing),
                sorted(missing)[:8],
            )
            return False
        mtp.load_weights(list(mtp_weights.items()), strict=False)
        mx.eval(mtp.parameters())
        extra = loaded_keys - expected_keys
        logger.info(
            "[mtp.inject] Loaded %d/%d expected MTP weight tensors from %s%s",
            len(expected_keys),
            len(expected_keys),
            weights_file.name,
            f" (+{len(extra)} extra sidecar key(s) ignored)" if extra else "",
        )
    else:
        if not allow_random_init:
            logger.warning(
                "[mtp.inject] inject_mtp_support called without "
                "mtp_sidecar and allow_random_init=False; refusing to "
                "ship a random-init MTP head. Pass "
                "mtp_sidecar='mlx-community/Qwen3.5-9B-MTP-4bit' (or "
                "equivalent) for production use, or set "
                "allow_random_init=True for unit-test wiring probes."
            )
            return False
        mx.eval(mtp.parameters())
        logger.warning(
            "[mtp.inject] inject_mtp_support called with "
            "allow_random_init=True — MTP head retains RANDOM init "
            "weights (accept rate ~0%%). This is the test-only path; "
            "do not use in production."
        )

    # --- Step 4: Install global ArraysCache + GatedDeltaNet patches ---
    from .cache_patch import (
        patch_arrays_cache_rollback_state,
        patch_gated_delta_net_for_mtp,
    )

    patch_arrays_cache_rollback_state()
    patch_gated_delta_net_for_mtp()

    # --- Step 5: Attach + monkey-patch TextModel class ---
    inner.mtp = mtp
    original_class = type(inner)

    class _Qwen3_5WithMTP(original_class):  # type: ignore[valid-type, misc]

        def __call__(  # type: ignore[override]
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            return_hidden: bool = False,
            n_confirmed: int = 0,
        ):
            from mlx_lm.models.base import create_attention_mask, create_ssm_mask

            inner_m = self.model
            if input_embeddings is not None:
                hidden_states = input_embeddings
            else:
                hidden_states = inner_m.embed_tokens(inputs)
            if cache is None:
                cache = [None] * len(inner_m.layers)

            if n_confirmed > 0:
                for c in cache:
                    if c is not None and hasattr(c, "rollback_state"):
                        c.n_confirmed_for_mtp = n_confirmed

            try:
                fa_mask = create_attention_mask(hidden_states, cache[inner_m.fa_idx])
                ssm_mask = create_ssm_mask(hidden_states, cache[inner_m.ssm_idx])
                for layer, c in zip(inner_m.layers, cache):
                    mask = ssm_mask if layer.is_linear else fa_mask
                    hidden_states = layer(hidden_states, mask=mask, cache=c)
            finally:
                if n_confirmed > 0:
                    for c in cache:
                        if c is not None and hasattr(c, "n_confirmed_for_mtp"):
                            c.n_confirmed_for_mtp = 0

            normed = inner_m.norm(hidden_states)
            if self.args.tie_word_embeddings:
                out = inner_m.embed_tokens.as_linear(normed)
            else:
                out = self.lm_head(normed)

            if return_hidden:
                return out, hidden_states
            return out

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache,
        ):
            mtp_out = self.mtp(
                hidden_states,
                next_token_ids,
                self.model.embed_tokens,
                mtp_cache,
            )
            if self.args.tie_word_embeddings:
                return self.model.embed_tokens.as_linear(mtp_out)
            return self.lm_head(mtp_out)

        def make_mtp_cache(self):
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in self.mtp.layers]

    inner.__class__ = _Qwen3_5WithMTP
    logger.info(
        "[mtp.inject] Patched %s with MTP surfaces "
        "(return_hidden, n_confirmed, mtp_forward, make_mtp_cache).",
        original_class.__name__,
    )
    return True


def validate_mtp_support(model: Any) -> bool:
    import inspect

    inner = _resolve_inner_text_model(model)
    if inner is None:
        return False

    if getattr(inner, "mtp", None) is None:
        logger.warning("[mtp.validate] model.mtp is missing.")
        return False
    if not callable(getattr(inner, "mtp_forward", None)):
        logger.warning("[mtp.validate] model.mtp_forward is missing.")
        return False
    if not callable(getattr(inner, "make_mtp_cache", None)):
        logger.warning("[mtp.validate] model.make_mtp_cache is missing.")
        return False
    sig = inspect.signature(type(inner).__call__)
    if "return_hidden" not in sig.parameters:
        logger.warning("[mtp.validate] model.__call__ does not accept return_hidden.")
        return False
    if "n_confirmed" not in sig.parameters:
        logger.warning("[mtp.validate] model.__call__ does not accept n_confirmed.")
        return False
    return True
