# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_ASSISTANT_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "gemma4_assistant",
        "gemma4_unified_assistant",
    }
)


def _resolve_inner_text_model(model: Any) -> Any:
    lm = getattr(model, "language_model", None)
    if lm is not None:
        if hasattr(lm, "args") and hasattr(lm, "model"):
            return lm
        if hasattr(lm, "config") and hasattr(lm, "model") and not hasattr(lm, "args"):
            try:
                lm.args = lm.config
            except AttributeError:
                from types import SimpleNamespace

                ns = SimpleNamespace()
                for k in dir(lm.config):
                    if k.startswith("_"):
                        continue
                    try:
                        setattr(ns, k, getattr(lm.config, k))
                    except AttributeError:
                        pass
                object.__setattr__(lm, "args", ns)
            return lm
    if hasattr(model, "model") and hasattr(model, "args"):
        return model
    return None


def _resolve_sidecar_dir(mtp_sidecar: str | Path) -> Path | None:
    if mtp_sidecar is None:
        return None

    path = Path(mtp_sidecar)
    if path.is_dir():
        return path
    if path.is_file():
        parent = path.parent
        if (parent / "config.json").exists():
            return parent
        logger.warning(
            "[mtp.inject.gemma4] sidecar file %s has no sibling config.json; "
            "cannot resolve as an assistant checkpoint.",
            path,
        )
        return None

    ref = str(mtp_sidecar)
    is_hf_shape = (
        "/" in ref
        and not ref.startswith("/")
        and not ref.startswith("./")
        and not ref.startswith("../")
        and ref.count("/") == 1
        and not ref.endswith("/")
    )
    if not is_hf_shape:
        logger.warning(
            "[mtp.inject.gemma4] sidecar %r is neither an existing local path "
            "nor an HF-repo-id-shape (``owner/name``). Refusing to attempt "
            "snapshot_download.",
            ref,
        )
        return None

    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=str(mtp_sidecar))
        return Path(local)
    except Exception as exc:
        logger.warning(
            "[mtp.inject.gemma4] could not resolve sidecar repo %r: %s",
            mtp_sidecar,
            exc,
        )
        return None


def _load_assistant_config(sidecar_dir: Path) -> dict | None:
    cfg_path = sidecar_dir / "config.json"
    if not cfg_path.exists():
        logger.warning(
            "[mtp.inject.gemma4] assistant dir %s has no config.json.",
            sidecar_dir,
        )
        return None
    try:
        return json.loads(cfg_path.read_text())
    except Exception as exc:
        logger.warning(
            "[mtp.inject.gemma4] could not parse config.json in %s: %s",
            sidecar_dir,
            exc,
        )
        return None


def _find_safetensors(sidecar_dir: Path) -> Path | None:
    matches = sorted(sidecar_dir.glob("*.safetensors"))
    if len(matches) > 1:
        logger.warning(
            "[mtp.inject.gemma4] %s contains %d .safetensors files: %s. "
            "Refusing to guess — sharded / multi-file assistant loading "
            "is not supported.",
            sidecar_dir,
            len(matches),
            [p.name for p in matches[:5]],
        )
        return None
    if len(matches) == 1:
        return matches[0]
    return None


def _build_assistant_model_args(
    assistant_cfg: dict,
    target_backbone_hidden: int,
) -> Any:
    from mlx_lm.models.gemma4_text import ModelArgs

    tc = assistant_cfg.get("text_config", {}) or {}

    n_layers = int(tc.get("num_hidden_layers", 4))
    layer_types = list(tc.get("layer_types") or [])
    if len(layer_types) != n_layers:
        logger.warning(
            "[mtp.inject.gemma4] layer_types has %d entries but num_hidden_layers=%d; "
            "refusing to build assistant args (schema mismatch).",
            len(layer_types),
            n_layers,
        )
        return None

    args = ModelArgs(
        model_type=str(tc.get("model_type", "gemma4_unified_text")),
        hidden_size=int(tc.get("hidden_size", 1024)),
        num_hidden_layers=n_layers,
        intermediate_size=int(tc.get("intermediate_size", 8192)),
        num_attention_heads=int(tc.get("num_attention_heads", 16)),
        head_dim=int(tc.get("head_dim", 256)),
        global_head_dim=int(tc.get("global_head_dim", 512)),
        rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
        vocab_size=int(tc.get("vocab_size", 262144)),
        vocab_size_per_layer_input=int(tc.get("vocab_size_per_layer_input", 0)),
        num_key_value_heads=int(tc.get("num_key_value_heads", 8)),
        num_global_key_value_heads=tc.get("num_global_key_value_heads"),
        num_kv_shared_layers=int(tc.get("num_kv_shared_layers", n_layers)),
        pad_token_id=int(tc.get("pad_token_id", 0)),
        hidden_size_per_layer_input=int(tc.get("hidden_size_per_layer_input", 0)),
        rope_parameters=tc.get("rope_parameters"),
        sliding_window=int(tc.get("sliding_window", 1024)),
        sliding_window_pattern=int(tc.get("sliding_window_pattern", 6)),
        max_position_embeddings=int(tc.get("max_position_embeddings", 262144)),
        attention_k_eq_v=bool(tc.get("attention_k_eq_v", False)),
        final_logit_softcapping=tc.get("final_logit_softcapping"),
        use_double_wide_mlp=bool(tc.get("use_double_wide_mlp", False)),
        enable_moe_block=bool(tc.get("enable_moe_block", False)),
        tie_word_embeddings=bool(tc.get("tie_word_embeddings", True)),
        layer_types=layer_types if layer_types else None,
    )
    bb = int(
        assistant_cfg.get("backbone_hidden_size")
        or tc.get("backbone_hidden_size")
        or target_backbone_hidden
    )
    if bb != target_backbone_hidden:
        logger.warning(
            "[mtp.inject.gemma4] assistant backbone_hidden_size=%d does not match "
            "target hidden_size=%d; drafter cross-projection shapes will not fit. "
            "Refusing to inject.",
            bb,
            target_backbone_hidden,
        )
        return None
    try:
        object.__setattr__(args, "backbone_hidden_size", bb)
    except (TypeError, AttributeError):
        pass
    return args


def _build_assistant_model(args: Any, backbone_hidden_size: int):
    import mlx.nn as nn
    from mlx_lm.models.gemma4_text import DecoderLayer

    class _AssistantBackbone(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
            self.layers = [
                DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    class _AssistantModel(nn.Module):
        def __init__(self, args, backbone_hidden_size):
            super().__init__()
            self.args = args
            self.backbone_hidden_size = backbone_hidden_size
            self.model = _AssistantBackbone(args)
            self.pre_projection = nn.Linear(
                2 * backbone_hidden_size,
                args.hidden_size,
                bias=False,
            )
            self.post_projection = nn.Linear(
                args.hidden_size,
                backbone_hidden_size,
                bias=False,
            )

    return _AssistantModel(args, backbone_hidden_size)


def inject_mtp_support(
    model: Any,
    mtp_sidecar: str | Path | None = None,
    *,
    allow_random_init: bool = False,
) -> bool:
    import mlx.core as mx

    inner = _resolve_inner_text_model(model)
    if inner is None:
        logger.warning(
            "[mtp.inject.gemma4] model %s has no .language_model or (.model + .args); "
            "skipping.",
            type(model).__name__,
        )
        return False

    if mtp_sidecar is None and not allow_random_init:
        logger.warning(
            "[mtp.inject.gemma4] inject_mtp_support called without mtp_sidecar and "
            "allow_random_init=False; refusing to ship a random-init drafter. "
            "Pass mtp_sidecar='google/gemma-4-12B-it-assistant' (or equivalent) "
            "for production use."
        )
        return False

    target_hidden = int(getattr(inner.args, "hidden_size", 0) or 0)
    if target_hidden <= 0:
        logger.warning(
            "[mtp.inject.gemma4] target hidden_size=%d unresolved; skipping.",
            target_hidden,
        )
        return False

    assistant_cfg: dict | None = None
    weights_file: Path | None = None
    if mtp_sidecar is not None:
        sidecar_dir = _resolve_sidecar_dir(mtp_sidecar)
        if sidecar_dir is None:
            return False

        assistant_cfg = _load_assistant_config(sidecar_dir)
        if assistant_cfg is None:
            return False

        cfg_type = str(assistant_cfg.get("model_type", ""))
        if cfg_type not in _ASSISTANT_MODEL_TYPES:
            logger.warning(
                "[mtp.inject.gemma4] sidecar %s has model_type=%r; expected one of %s. "
                "Refusing to inject.",
                sidecar_dir,
                cfg_type,
                sorted(_ASSISTANT_MODEL_TYPES),
            )
            return False

        weights_file = _find_safetensors(sidecar_dir)
        if weights_file is None:
            logger.warning(
                "[mtp.inject.gemma4] no safetensors found under %s.", sidecar_dir
            )
            return False

    if assistant_cfg is not None:
        args = _build_assistant_model_args(assistant_cfg, target_hidden)
        if args is None:
            return False
        backbone_hidden = int(getattr(args, "backbone_hidden_size", target_hidden))

        target_vocab = int(getattr(inner.args, "vocab_size", 0) or 0)
        assistant_vocab = int(getattr(args, "vocab_size", 0) or 0)
        if target_vocab <= 0 or assistant_vocab <= 0:
            logger.warning(
                "[mtp.inject.gemma4] vocab_size unresolved: target=%d assistant=%d. "
                "Both must be positive integers; refusing to inject a drafter "
                "with an invalid vocabulary size.",
                target_vocab,
                assistant_vocab,
            )
            return False
        if target_vocab != assistant_vocab:
            logger.warning(
                "[mtp.inject.gemma4] vocab_size mismatch: target=%d assistant=%d. "
                "The assistant tokenizer differs from the target's — refusing to "
                "inject; drafter logits would be indexed into the wrong vocab.",
                target_vocab,
                assistant_vocab,
            )
            return False

        assistant_layer_types = list(getattr(args, "layer_types", []) or [])
        target_layer_types = list(getattr(inner.args, "layer_types", []) or [])
        n_assistant = len(assistant_layer_types)
        if assistant_layer_types and target_layer_types:
            if len(target_layer_types) < n_assistant:
                logger.warning(
                    "[mtp.inject.gemma4] target published only %d layer_types "
                    "but the assistant has %d drafter layers; the tail mapping "
                    "cannot be resolved. Refusing to inject.",
                    len(target_layer_types),
                    n_assistant,
                )
                return False
            target_tail = target_layer_types[-n_assistant:]
            if target_tail != assistant_layer_types:
                logger.warning(
                    "[mtp.inject.gemma4] target tail layer_types %r do not match "
                    "assistant layer_types %r; refusing to inject cross-KV under "
                    "mismatched attention semantics.",
                    target_tail,
                    assistant_layer_types,
                )
                return False
        elif not target_layer_types and assistant_layer_types:
            logger.warning(
                "[mtp.inject.gemma4] target has no layer_types published on "
                "``inner.args``; cannot verify tail layer_types match the "
                "assistant's %r. Refusing to inject rather than silently ship "
                "a semantics-mismatched drafter.",
                assistant_layer_types,
            )
            return False
    else:
        from mlx_lm.models.gemma4_text import ModelArgs

        random_vocab = int(getattr(inner.args, "vocab_size", 0) or 0)
        if random_vocab <= 0:
            logger.warning(
                "[mtp.inject.gemma4] allow_random_init=True but target has no "
                "resolvable vocab_size; refusing to build random drafter.",
            )
            return False

        _target_head_dim = int(getattr(inner.args, "head_dim", 16) or 16)
        _target_global_head_dim = int(
            getattr(inner.args, "global_head_dim", _target_head_dim) or _target_head_dim
        )
        _target_n_kv = int(getattr(inner.args, "num_key_value_heads", 1) or 1)
        _target_n_global_kv = int(
            getattr(inner.args, "num_global_key_value_heads", _target_n_kv)
            or _target_n_kv
        )

        args = ModelArgs(
            model_type="gemma4_unified_text",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            head_dim=_target_head_dim,
            global_head_dim=_target_global_head_dim,
            num_key_value_heads=_target_n_kv,
            num_global_key_value_heads=_target_n_global_kv,
            rms_norm_eps=1e-6,
            vocab_size=random_vocab,
            vocab_size_per_layer_input=0,
            num_kv_shared_layers=2,
            hidden_size_per_layer_input=0,
            sliding_window=64,
            sliding_window_pattern=2,
            max_position_embeddings=128,
            final_logit_softcapping=None,
            enable_moe_block=False,
            use_double_wide_mlp=False,
            tie_word_embeddings=True,
            layer_types=["sliding_attention", "full_attention"],
            attention_k_eq_v=False,
        )
        backbone_hidden = target_hidden

    try:
        assistant = _build_assistant_model(args, backbone_hidden)
    except Exception as exc:
        logger.warning(
            "[mtp.inject.gemma4] failed to instantiate AssistantModel: %s", exc
        )
        return False

    if weights_file is not None:
        try:
            raw = mx.load(str(weights_file))
        except Exception as exc:
            logger.warning(
                "[mtp.inject.gemma4] mx.load(%s) failed: %s. Refusing to inject.",
                weights_file,
                exc,
            )
            return False
        from mlx.utils import tree_flatten

        expected_pairs = list(tree_flatten(assistant.parameters()))
        expected_shapes = {k: tuple(v.shape) for k, v in expected_pairs}
        expected_keys = set(expected_shapes)
        loaded_keys = set(raw.keys())
        missing = expected_keys - loaded_keys
        if missing:
            logger.warning(
                "[mtp.inject.gemma4] assistant weights missing %d tensor(s) — "
                "first 8: %s. Refusing to inject with a partially-random head.",
                len(missing),
                sorted(missing)[:8],
            )
            return False
        shape_mismatches: list[tuple[str, tuple, tuple]] = []
        for k in expected_keys:
            got = tuple(raw[k].shape)
            want = expected_shapes[k]
            if got != want:
                shape_mismatches.append((k, got, want))
        if shape_mismatches:
            logger.warning(
                "[mtp.inject.gemma4] %d tensor(s) in %s have shapes "
                "incompatible with the AssistantModel. First 4: %s. "
                "Refusing to inject.",
                len(shape_mismatches),
                weights_file,
                [(k, g, w) for (k, g, w) in shape_mismatches[:4]],
            )
            return False
        try:
            assistant.load_weights(list(raw.items()), strict=False)
            mx.eval(assistant.parameters())
        except Exception as exc:
            logger.warning(
                "[mtp.inject.gemma4] load_weights / eval failed on %s: %s. "
                "Refusing to inject with an inconsistent drafter.",
                weights_file,
                exc,
            )
            return False
        extra = loaded_keys - expected_keys
        if extra:
            logger.warning(
                "[mtp.inject.gemma4] sidecar %s carries %d tensor(s) the MVP "
                "consumer does not load (first 8: %s). Loading continues; "
                "review the follow-up TODOs (centroid embedder / draft chain) "
                "before treating these as unused.",
                weights_file.name,
                len(extra),
                sorted(extra)[:8],
            )
        logger.info(
            "[mtp.inject.gemma4] Loaded %d/%d assistant tensors from %s",
            len(expected_keys),
            len(expected_keys),
            weights_file.name,
        )
    else:
        mx.eval(assistant.parameters())
        logger.warning(
            "[mtp.inject.gemma4] allow_random_init=True — assistant drafter has "
            "RANDOM init weights (accept rate ~0%%). Test-only wiring probe."
        )

    from .cache_patch import patch_arrays_cache_rollback_state

    patch_arrays_cache_rollback_state()

    original_class = type(inner)
    _target_num_layers = len(getattr(inner.model, "layers", []) or [])
    _n_assistant_layers = len(assistant.model.layers)

    _assistant_layer_types = list(getattr(assistant.args, "layer_types", []) or [])
    _target_layer_types = list(getattr(inner.args, "layer_types", []) or [])
    _last_target_idx_by_type: dict[str, int] = {}
    for _i, _lt in enumerate(_target_layer_types):
        _last_target_idx_by_type[_lt] = _i
    if _assistant_layer_types and all(
        lt in _last_target_idx_by_type for lt in _assistant_layer_types
    ):
        _shared_kv_target_indices: list[int] = [
            _last_target_idx_by_type[lt] for lt in _assistant_layer_types
        ]
    else:
        _shared_kv_target_indices = list(
            range(_target_num_layers - _n_assistant_layers, _target_num_layers)
        )
    _target_embed_scale = float(
        getattr(getattr(inner, "model", None), "embed_scale", 1.0) or 1.0
    )

    class _Gemma4WithMTP(original_class):
        mtp_max_batch_size = 1

        def __call__(  # type: ignore[override]
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            per_layer_inputs=None,
            *args,
            return_hidden: bool = False,
            n_confirmed: int = 0,
            **kwargs,
        ):
            self._mtp_target_cache = cache
            hidden = self.model(
                inputs,
                *args,
                cache=cache,
                input_embeddings=input_embeddings,
                per_layer_inputs=per_layer_inputs,
                **kwargs,
            )
            if self.args.tie_word_embeddings:
                out = self.model.embed_tokens.as_linear(hidden)
            else:
                out = self.lm_head(hidden)
            if self.args.final_logit_softcapping is not None:
                from mlx_lm.models.gemma4_text import logit_softcap

                out = logit_softcap(self.args.final_logit_softcapping, out)
            _ = n_confirmed
            if return_hidden:
                return out, hidden
            return out

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache,
            *,
            return_hidden: bool = False,
        ):
            import mlx.core as _mx

            target_cache = getattr(self, "_mtp_target_cache", None)
            if target_cache is None:
                raise RuntimeError(
                    "[mtp.inject.gemma4] mtp_forward invoked before a target "
                    "backbone forward — target KV cache is not populated. This "
                    "should not happen with the vendored mtp_generate_step."
                )
            n_take = _n_assistant_layers
            if len(target_cache) < n_take:
                raise RuntimeError(
                    f"[mtp.inject.gemma4] target has {len(target_cache)} cache slots "
                    f"but the assistant requires {n_take}."
                )
            shared_kv_slots = [target_cache[idx] for idx in _shared_kv_target_indices]

            if hidden_states.ndim == 2:
                hidden_states = hidden_states[None]
            if next_token_ids.ndim == 1:
                next_token_ids = next_token_ids[None]

            if hidden_states.shape[0] != 1:
                raise ValueError(
                    "[mtp.inject.gemma4] mtp_forward only supports batch=1 "
                    f"today; got hidden_states.shape[0]={hidden_states.shape[0]}."
                )
            if next_token_ids.shape[0] != 1:
                raise ValueError(
                    "[mtp.inject.gemma4] mtp_forward only supports batch=1 "
                    f"today; got next_token_ids.shape[0]={next_token_ids.shape[0]}."
                )

            if mtp_cache is not None:
                for _slot in mtp_cache:
                    if int(getattr(_slot, "offset", 0)) != 0:
                        raise ValueError(
                            "[mtp.inject.gemma4] mtp_cache slots must be "
                            "empty on entry — Gemma 4 assistant drafter "
                            "reads target's shared K/V and does not "
                            "maintain its own per-layer cache; a "
                            "populated ``mtp_cache`` indicates the caller "
                            "expects chained drafts, which are not "
                            "supported. See ``make_mtp_cache`` docstring."
                        )

            n_positions = int(hidden_states.shape[1])

            layer_states: list[tuple] = []
            layer_offsets: list[int] = []
            for tgt_cache in shared_kv_slots:
                try:
                    state = tgt_cache.state
                except AttributeError:
                    state = None
                if state is None or (
                    isinstance(state, tuple) and state[0] is None
                ):
                    raise RuntimeError(
                        "[mtp.inject.gemma4] target cache slot has empty state; "
                        "cannot compute drafter attention without any K/V."
                    )
                if isinstance(state, tuple):
                    keys, values = state[0], state[1]
                else:
                    keys = state
                    values = state
                layer_states.append((keys, values))
                layer_offsets.append(int(tgt_cache.offset))

            target_embed = self.model.embed_tokens
            next_embed_all = (
                target_embed(next_token_ids) * _target_embed_scale
            )

            per_position_h: list = []
            for pos in range(n_positions):
                h_pos = hidden_states[:, pos : pos + 1, :]
                next_pos = next_embed_all[:, pos : pos + 1, :]
                fused_pos = _mx.concatenate(
                    [next_pos, h_pos], axis=-1
                )
                h_pos_proj = self.mtp.pre_projection(fused_pos)

                h_layer = h_pos_proj
                for layer, (keys, values), tgt_offset in zip(
                    self.mtp.model.layers, layer_states, layer_offsets
                ):
                    row_offset_int = tgt_offset - n_positions + 1 + pos
                    if row_offset_int < 0:
                        raise ValueError(
                            "[mtp.inject.gemma4] target cache offset "
                            f"({tgt_offset}) is smaller than "
                            f"n_positions-1 ({n_positions - 1}) at drafter "
                            f"row {pos}; caller must prefill target before "
                            "drafting."
                        )
                    row_offset = _mx.array(row_offset_int)
                    h_layer, _shared, _off = layer(
                        h_layer,
                        mask=None,
                        cache=None,
                        per_layer_input=None,
                        shared_kv=(keys, values),
                        offset=row_offset,
                    )
                per_position_h.append(h_layer)

            h = _mx.concatenate(per_position_h, axis=1)
            h = self.mtp.model.norm(h)
            logits = self.mtp.model.embed_tokens.as_linear(h)
            _ = mtp_cache
            if return_hidden:
                drafter_backbone_hidden = self.mtp.post_projection(h)
                return logits, drafter_backbone_hidden
            return logits

        def make_mtp_cache(self):
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in self.mtp.model.layers]

    import types as _types

    try:
        _delegate_forward = None
        _delegate_cache = None
        if model is not inner:

            def _delegate_forward(
                _self,
                hidden_states,
                next_token_ids,
                mtp_cache,
                *,
                return_hidden: bool = False,
            ):
                return inner.mtp_forward(
                    hidden_states,
                    next_token_ids,
                    mtp_cache,
                    return_hidden=return_hidden,
                )

            def _delegate_cache(_self):
                return inner.make_mtp_cache()

        inner.__class__ = _Gemma4WithMTP
        inner.mtp = assistant
        if model is not inner:
            model.mtp = inner.mtp
            model.mtp_forward = _types.MethodType(_delegate_forward, model)
            model.make_mtp_cache = _types.MethodType(_delegate_cache, model)
            model.mtp_max_batch_size = _Gemma4WithMTP.mtp_max_batch_size
    except Exception as exc:
        try:
            if type(inner) is _Gemma4WithMTP:
                inner.__class__ = original_class
            try:
                inner.__dict__.pop("mtp", None)
            except Exception:
                pass
            if model is not inner:
                for attr in (
                    "mtp",
                    "mtp_forward",
                    "make_mtp_cache",
                    "mtp_max_batch_size",
                ):
                    try:
                        model.__dict__.pop(attr, None)
                    except Exception:
                        pass
                    try:
                        delattr(model, attr)
                    except AttributeError:
                        pass
                    except Exception:
                        pass
        except Exception:
            pass
        logger.warning(
            "[mtp.inject.gemma4] failed during inject commit; rolled back. Error: %s",
            exc,
        )
        return False

    logger.info(
        "[mtp.inject.gemma4] Injected Google assistant drafter "
        "(layers=%d, hidden=%d, backbone_hidden=%d) onto %s.",
        _n_assistant_layers,
        assistant.args.hidden_size,
        backbone_hidden,
        original_class.__name__,
    )
    _ = _target_num_layers
    return True


def validate_mtp_support(model: Any) -> bool:
    import inspect

    inner = _resolve_inner_text_model(model)
    if inner is None:
        return False

    if getattr(inner, "mtp", None) is None:
        logger.warning("[mtp.validate.gemma4] model.mtp is missing.")
        return False
    if not callable(getattr(inner, "mtp_forward", None)):
        logger.warning("[mtp.validate.gemma4] model.mtp_forward is missing.")
        return False
    if not callable(getattr(inner, "make_mtp_cache", None)):
        logger.warning("[mtp.validate.gemma4] model.make_mtp_cache is missing.")
        return False
    sig = inspect.signature(type(inner).__call__)
    if "return_hidden" not in sig.parameters:
        logger.warning(
            "[mtp.validate.gemma4] model.__call__ does not accept return_hidden."
        )
        return False
    if "n_confirmed" not in sig.parameters:
        logger.warning(
            "[mtp.validate.gemma4] model.__call__ does not accept n_confirmed."
        )
        return False

    inner_max_batch = getattr(inner, "mtp_max_batch_size", None)
    if inner_max_batch != 1:
        logger.warning(
            "[mtp.validate.gemma4] inner model.mtp_max_batch_size=%r does not "
            "equal 1. Schedulers that inspect this gate would misroute B>1.",
            inner_max_batch,
        )
        return False

    if model is not inner:
        for attr in (
            "mtp",
            "mtp_forward",
            "make_mtp_cache",
            "mtp_max_batch_size",
        ):
            if not hasattr(model, attr):
                logger.warning(
                    "[mtp.validate.gemma4] outer wrapper missing delegated "
                    "surface: %s. Inner-only patch would cause the generator "
                    "to AttributeError on the outer.",
                    attr,
                )
                return False
        if getattr(model, "mtp", None) is None:
            logger.warning(
                "[mtp.validate.gemma4] outer wrapper's ``mtp`` attribute is "
                "None; drafter is not attached."
            )
            return False
        outer_max_batch = getattr(model, "mtp_max_batch_size", None)
        if outer_max_batch != 1:
            logger.warning(
                "[mtp.validate.gemma4] outer wrapper's mtp_max_batch_size=%r "
                "does not equal 1 (the value the inject path sets). This "
                "would let schedulers dispatch B>1 requests into the "
                "batch=1-only mtp_forward path.",
                outer_max_batch,
            )
            return False
        if not callable(getattr(model, "mtp_forward", None)):
            logger.warning(
                "[mtp.validate.gemma4] outer wrapper's mtp_forward is not callable."
            )
            return False
        if not callable(getattr(model, "make_mtp_cache", None)):
            logger.warning(
                "[mtp.validate.gemma4] outer wrapper's make_mtp_cache is not callable."
            )
            return False
    return True
