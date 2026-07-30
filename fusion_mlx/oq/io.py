# SPDX-License-Identifier: Apache-2.0
"""oQ: FusionMLX Universal Dynamic Quantization.

Mixed-precision quantization combining GGUF K-quant layer position strategy,
unsloth Dynamic 2.0 selective non-quantization, and BnB MSE-optimal clipping.

Supported levels: oQ2, oQ2.5, oQ2.7, oQ3, oQ3.5, oQ4, oQ5, oQ6, oQ8 (base
bits differ, same predicate). Fractional levels keep the lower level's base
bits and add a mandatory boost for routed expert down_proj (Super Weights
protection; see _LEVEL_EXPERT_DOWN_BOOST) plus a higher bpw budget.
"""

import json
import logging
import re
import shutil
import tempfile
import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.base import create_attention_mask

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from fusion_mlx.pool.model_discovery import _has_vision_subconfig

logger = logging.getLogger(__name__)

OQ_LEVELS = {2, 2.5, 2.7, 3, 3.5, 4, 5, 6, 8}

OQ_DTYPES: tuple[str, ...] = ("bfloat16", "float16")

_OQ_DEFAULT_GROUP_SIZE = 64

_MAX_MODEL_RAM_FRACTION = 0.8

# Auto-built proxy for sensitivity measurement when the source model
# exceeds available RAM. Uniform 4-bit affine quant — same shape as a
# user-supplied --sensitivity-model, but built on demand.
_PROXY_QUANT_BITS = 4
_PROXY_QUANT_GROUP_SIZE = 64

_LEVEL_BITS: dict[float, int] = {
    2: 2,
    2.5: 2,
    2.7: 2,
    3: 3,
    3.5: 3,
    4: 4,
    5: 5,
    6: 6,
    8: 8,
}

_LEVEL_PROTECTION: dict[float, str] = {
    2: "full",
    2.5: "full",
    2.7: "full",
    3: "full",
    3.5: "full",
    4: "full",
    5: "full",
    6: "full",
    8: "full",
}

# Fractional levels: mandatory protection for routed expert down_proj
# (Super Weights), expressed as bits above the level's base bits.
# 2.5 -> 3-bit, 2.7 -> 4-bit, 3.5 -> 4-bit.
_LEVEL_EXPERT_DOWN_BOOST: dict[float, int] = {2.5: 1, 2.7: 2, 3.5: 1}

_OQ_BPW_TARGETS: dict[float, tuple[float, float]] = {
    2: (2.8, 3.0),
    2.5: (3.1, 3.3),
    2.7: (3.35, 3.45),
    3: (3.5, 3.7),
    3.5: (3.8, 4.0),
    4: (4.6, 4.7),
    5: (5.5, 5.7),
    6: (6.5, 6.7),
}


from ._core import _TrackedTensor, _DiscoveredPlan, _discover_sanitize_plan, _block_dequant_fp8
def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


_MAX_SHARD_BYTES = 5_000_000_000

_SKIP_QUANT_PATTERNS = (
    "layernorm",
    "rmsnorm",
    "norm.weight",
    "norm.bias",
    "ln_",
    "layer_norm",
)


def _should_skip_tensor(name: str, preserve_mtp: bool = False) -> bool:
    """Check if a tensor should be completely excluded from output.

    By default mtp.* tensors are stripped because mlx-lm's stock sanitize()
    removes them when the model has no MTP head. When ``preserve_mtp`` is
    True the caller has stashed mtp.* tensors around the sanitize call and
    re-merged them, so we must keep them in the output shards.
    """
    if ".mtp." in name or name.startswith("mtp."):
        return not preserve_mtp
    return False


def _is_mtp_tensor(name: str) -> bool:
    """Return True iff the tensor key belongs to an MTP head."""
    return name.startswith("mtp.") or ".mtp." in name


def _normalize_mtp_in_config(config: dict) -> None:
    """Zero out MTP layer counts in the output config (in place).

    Used when preserve_mtp is False so the resulting quantized model
    presents itself as MTP-free. Without this, the source config's
    mtp_num_hidden_layers / num_nextn_predict_layers values would survive
    while the actual mtp.* tensors are stripped, producing the
    "Missing N parameters" load error we hit on Qwen3.5-27B.
    """
    for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
        if key in config and config[key]:
            config[key] = 0
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
            if key in text_cfg and text_cfg[key]:
                text_cfg[key] = 0


def _should_quantize_tensor(name: str, shape: tuple) -> bool:
    """Check if a tensor should be quantized based on name and shape."""
    if not name.endswith(".weight"):
        # Only module weights are quantizable. 2D plain parameters (e.g.
        # DeepSeek V4 hyper-connection fn/base tables, compressor.ape)
        # must pass through untouched — emitting weight/scales pairs for
        # them would corrupt the checkpoint.
        return False
    if len(shape) < 2:
        return False
    name_lower = name.lower()
    if any(p in name_lower for p in _SKIP_QUANT_PATTERNS):
        return False
    if name.endswith(".bias"):
        return False
    return True


def _cast_passthrough_tensor(tensor_name: str, w_mx, target_dtype):
    """Cast an unquantized output tensor to its storage dtype."""
    if not mx.issubdtype(w_mx.dtype, mx.floating):
        return w_mx

    if target_dtype == mx.float16 and (
        _is_vision_tensor(tensor_name) or _is_audio_tensor(tensor_name)
    ):
        if w_mx.dtype != mx.float32:
            return w_mx.astype(mx.float32)
        return w_mx

    if w_mx.dtype != target_dtype:
        return w_mx.astype(target_dtype)
    return w_mx


def _build_model_sanitizer(config: dict, text_only: bool = False):
    """Build a sanitize function from the model class.

    For VLM models, uses mlx-vlm's model class (preserves vision weights).
    For LLM models, uses mlx-lm's model class.
    When text_only is True, always uses the LLM path even for VLM
    architectures so that mlx_lm_mtp patches (which handle MTP sanitize
    for both dense and MoE) are used instead of the VLM path whose
    _Proxy-based sanitize drops the MTP head.

    Returns:
        A function that takes a dict of weights and returns sanitized weights,
        or None if the model class can't be loaded.
    """
    architectures = config.get("architectures", [])
    is_vlm = (
        any("ForConditionalGeneration" in a for a in architectures) and not text_only
    )

    if is_vlm:
        try:
            try:
                model_type = config.get("model_type")
                text_config = config.get("text_config")
                text_model_type = (
                    text_config.get("model_type")
                    if isinstance(text_config, dict)
                    else None
                )
                if model_type in ("minimax_m3", "minimax_m3_vl") or (
                    text_model_type in ("minimax_m3", "minimax_m3_vl")
                ):
                    from fusion_mlx.patches.mlx_vlm_minimax_m3_compat import (
                        apply_mlx_vlm_minimax_m3_compat_patch,
                    )

                    apply_mlx_vlm_minimax_m3_compat_patch()
            except Exception as patch_err:
                logger.debug(f"MiniMax M3 mlx-vlm patch not applied: {patch_err}")

            from mlx_vlm.utils import get_model_and_args, sanitize_weights

            # Apply mlx-vlm MTP sanitize patch so qwen3_5/qwen3_5_moe Model
            # classes keep ``mtp.*`` weights and shift the MTP-specific
            # RMSNorm tensors by +1 (matching mlx_lm_mtp/qwen35_model.py).
            # Without this, oQ output ships raw MTP norm weights, the
            # mlx-lm patched sanitize on load doesn't re-shift (it guards on
            # the unsanitized conv1d marker, which is False after oQ), and
            # the MTP head produces garbage logits — 0% accept rate.
            try:
                from fusion_mlx.patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_patch

                apply_mlx_vlm_mtp_patch()
            except Exception as patch_err:
                logger.debug(f"mlx-vlm MTP patch not applied: {patch_err}")

            # Remap language_model.model.visual.* -> vision_tower.* for
            # Qwen3.6-35B-A3B's nested ViT layout. Wraps whichever
            # Model.sanitize is current; no-op when already installed or
            # when upstream mlx-vlm grows the rule itself.
            try:
                from fusion_mlx.patches.qwen3_6_nested_visual import (
                    apply_qwen3_6_nested_visual_patch,
                )

                apply_qwen3_6_nested_visual_patch()
            except Exception as patch_err:
                logger.debug(f"qwen3_6 nested-visual patch not applied: {patch_err}")

            model_module, _ = get_model_and_args(config)
            model_config_cls = model_module.ModelConfig
            model_config = model_config_cls.from_dict(config)

            vision_config = model_config.vision_config
            if isinstance(vision_config, dict):
                vision_config = model_module.VisionConfig.from_dict(vision_config)
            text_config = model_config.text_config
            if isinstance(text_config, dict):
                text_config = model_module.TextConfig.from_dict(text_config)

            model_config.vision_config = vision_config
            model_config.text_config = text_config

            # Some VLM Model.sanitize implementations (e.g. Gemma 4) drop
            # `audio_tower.*` / `embed_audio.*` weights when `self.audio_tower`
            # is None. Set a truthy sentinel iff the source config carries an
            # `audio_config` so the audio modality survives sanitize and stays
            # in the quantization pipeline.
            has_audio = config.get("audio_config") is not None
            _AUDIO_SENTINEL = object() if has_audio else None

            def _vlm_sanitize(weights):
                class _Proxy:
                    # The audio-presence guard differs by arch: gemma4 checks
                    # ``self.audio_tower``; gemma4_unified checks
                    # ``self.embed_audio``. Expose BOTH (sentinel iff the source
                    # config carries audio) so sanitize keeps the audio modality
                    # for either. Missing ``embed_audio`` made gemma4_unified's
                    # sanitize raise AttributeError, silently dropping the whole
                    # sanitize pass → oQ shipped raw ``model.``-prefixed keys
                    # that fusion_mlx could not load.
                    audio_tower = _AUDIO_SENTINEL
                    embed_audio = _AUDIO_SENTINEL

                proxy = _Proxy()
                proxy.config = model_config
                # Nested-VLM sanitizes (e.g. MiniMax-M3 minimax_m3_vl) read
                # self.language_model.args.{num_hidden_layers,num_local_experts}
                # for MoE expert stacking; expose text_config so proxy-based
                # discovery works without instantiating the full model.
                _lm_proxy = type("_LMProxy", (), {})()
                _lm_proxy.args = text_config
                proxy.language_model = _lm_proxy
                w = model_module.Model.sanitize(proxy, weights)

                w = sanitize_weights(model_module.VisionModel, w, vision_config)
                w = sanitize_weights(model_module.LanguageModel, w, text_config)
                return w

            logger.info(
                f"Using mlx-vlm full sanitize chain for "
                f"{model_module.Model.__name__} "
                f"(preserves vision{', audio' if has_audio else ''} weights)"
            )
            return _vlm_sanitize
        except Exception as e:
            logger.debug(f"mlx-vlm sanitizer not available: {e}")

    try:
        from mlx_lm.utils import _get_classes

        if config.get("model_type") == "glm_moe_dsa":
            try:
                from fusion_mlx.patches.glm_moe_dsa import apply_glm_moe_dsa_patch

                apply_glm_moe_dsa_patch()
            except Exception as patch_err:
                logger.debug(f"glm_moe_dsa patch not applied: {patch_err}")

        # DeepSeek-V4 isn't in stock mlx-lm — its model class is injected
        # into ``sys.modules`` by FusionMLX's base patch. Trigger that here so
        # ``_get_classes(config)`` for deepseek_v4* model types succeeds.
        # No-op for other model types.
        if str(config.get("model_type", "")).startswith("deepseek_v4"):
            try:
                from fusion_mlx.patches.deepseek_v4 import apply_deepseek_v4_patch

                apply_deepseek_v4_patch()
            except Exception as patch_err:
                logger.debug(f"deepseek_v4 base patch not applied: {patch_err}")

        # Apply mlx-lm MTP patch so the patched __init__/sanitize handle
        # mtp.* tensors correctly. Idempotent — apply() is a no-op once
        # patched.
        try:
            from fusion_mlx.patches.mlx_lm_mtp import (
                apply_mlx_lm_mtp_patch,
                is_mtp_active,
                set_mtp_active,
            )

            apply_mlx_lm_mtp_patch()
            _have_mtp_patch = True
        except Exception as patch_err:
            logger.debug(f"mlx-lm MTP patch not applied: {patch_err}")
            _have_mtp_patch = False

        model_class, model_args_class = _get_classes(config)
        args = model_args_class.from_dict(config)

        # Force MTP active during model instantiation so the patched
        # ``__init__`` attaches ``self.mtp``. With ``self.mtp`` attached,
        # the patched ``Model.sanitize`` keeps ``mtp.*`` weights and applies
        # the +1 RMSNorm shift to MTP norms (matching backbone). Without
        # this, mtp.* would be stripped and MTP norms would never receive
        # the shift, producing 0% accept rate after quantization.
        if _have_mtp_patch:
            prev_active = is_mtp_active()
            try:
                set_mtp_active(True)
                model = model_class(args)
            finally:
                set_mtp_active(prev_active)
        else:
            model = model_class(args)

        if hasattr(model, "sanitize"):
            logger.info(
                f"Using mlx-lm {model_class.__name__}.sanitize() "
                f"for weight transformation"
            )
            bound_sanitize = model.sanitize

            def _sanitize(weights):
                return bound_sanitize(weights)

            # Expose the model's cast predicate (key -> bool, False = keep
            # the source dtype) so the streaming loop can skip the target
            # dtype cast for tensors the model declares non-castable
            # (e.g. DeepSeek V4 attn_sink / hyper-connection tables).
            _sanitize._fmlx_cast_predicate = getattr(model, "cast_predicate", None)
            return _sanitize
    except Exception as e:
        logger.warning(f"Could not build model sanitizer: {e}")

    return None


def _copy_model_sidecars(source: Path, output: Path) -> None:
    """Copy tokenizer/processor sidecar files needed to load the output."""
    for pattern in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "generation_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
    ):
        for src_file in source.glob(pattern):
            shutil.copy2(src_file, output / src_file.name)

    for py_file in source.glob("*.py"):
        shutil.copy2(py_file, output / py_file.name)


def _build_non_quantizable_set(config: dict) -> set:
    """Find module paths with 2D weights that lack to_quantized() support.

    Loads the model class (without real weights) and walks the module tree.
    Modules like ScaledLinear in Gemma 4 have a weight attribute but no
    to_quantized(), so they cannot be loaded as QuantizedLinear after
    quantization. Returns empty set if the model class cannot be loaded.
    """
    try:
        from mlx_lm.utils import _get_classes

        model_class, model_args_class = _get_classes(config)
        args = model_args_class.from_dict(config)
        model = model_class(args)

        result = set()
        for path, module in tree_flatten(
            model.leaf_modules(), is_leaf=nn.Module.is_module
        ):
            if hasattr(module, "weight") and not hasattr(module, "to_quantized"):
                if getattr(module.weight, "ndim", 0) >= 2:
                    result.add(_normalize_quant_path(path))

        if result:
            logger.info(
                "Non-quantizable modules (no to_quantized): "
                + ", ".join(sorted(result))
            )
        return result
    except Exception as e:
        logger.debug(f"Could not build non-quantizable set: {e}")
        return set()


def _is_mtp_protected_tensor(name: str) -> bool:
    """Tensors inside the MTP head that must stay in full precision.

    Aggressive quantization of the MTP head's fusion projection or final
    hyper-head collapses draft acceptance to ~0% (oQ4 of an MTP-preserved
    Qwen3.5-27B accepted 0/157 cycles). PR 990 protects ``mtp.fc`` for
    Qwen3.5/3.6; PR 15's DeepSeek-V4 ``MTPBlock`` exposes the same
    semantics under different names (``e_proj`` + ``h_proj`` for the
    embedding/hidden fusion; ``hc_head.*`` for the final projection).
    All of these stay in full precision; the MTP block's internal
    DeepseekV4Block (attn/ffn) gets the same quantization as the
    backbone's other layers.
    """
    if not (name.startswith("mtp.") or ".mtp." in name):
        return False
    # Qwen3.5/3.6 fusion projection
    if name.endswith("mtp.fc.weight") or ".mtp.fc.weight" in name:
        return True
    # DeepSeek-V4 MTPBlock fusion projections
    if name.endswith(".e_proj.weight") or name.endswith(".h_proj.weight"):
        return True
    # DeepSeek-V4 HyperHead final projection (sanitized form has the dot;
    # the raw-HF form arrives as ``hc_head_<param>`` and we cover both).
    if ".hc_head." in name:
        return True
    if (
        name.endswith(".hc_head_fn")
        or name.endswith(".hc_head_base")
        or name.endswith(".hc_head_scale")
    ):
        return True
    return False


def _get_predicate_bits(
    tensor_name: str, config: dict, oq_level: int, group_size: int
) -> tuple:
    """Get quantization bits, group_size, and mode for a tensor.

    Returns:
        (bits, group_size, mode) or (None, None, None) if not quantized.
    """
    # See _is_mtp_protected_tensor for why these tensors stay full precision.
    if _is_mtp_protected_tensor(tensor_name):
        return None, None, None

    base_bits = _base_bits_for_level(oq_level)

    result = universal_quant_predicate(tensor_name, None, config, oq_level)
    if result is False:
        return None, None, None
    if isinstance(result, dict):
        bits = result.get("bits", base_bits)
        gs = result.get("group_size", group_size)
        mode = result.get("mode", _mode_for_bits(bits))
        return bits, gs, mode
    return base_bits, _gs_for_mode(base_bits, group_size), _mode_for_bits(base_bits)


def _mode_for_bits(bits: int) -> str:
    """Select quantization mode. Always affine to minimize kernel combos."""
    return "affine"


def _gs_for_mode(bits: int, default_gs: int) -> int:
    """Get group_size. Always default to minimize kernel combos."""
    return default_gs


# --- chunked-quantize helpers (added for Qwen3.5-397B) ---------------------
import struct as _struct

import numpy as _np


def _metal_max_buffer_bytes() -> int:
    try:
        info = mx.device_info()
    except AttributeError:
        try:
            info = mx.metal.device_info()
        except Exception:
            return 1 << 30
    except Exception:
        return 1 << 30
    return int(info.get("max_buffer_length", 1 << 30))


_METAL_MAX_BUFFER = _metal_max_buffer_bytes()
_QUANTIZE_CHUNK_BYTES = max(1 << 20, _METAL_MAX_BUFFER // 4)
_LOAD_CHUNK_BYTES = max(1 << 20, _METAL_MAX_BUFFER // 2)


class _LazyTensorIndex:
    _DTYPE_BYTES = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "F64": 8,
        "I8": 1,
        "U8": 1,
        "I16": 2,
        "U16": 2,
        "I32": 4,
        "U32": 4,
        "I64": 8,
        "U64": 8,
        "BOOL": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "F8_E8M0": 1,
    }

    def __init__(self, weight_files):
        self._index = {}
        for sf_path in weight_files:
            with open(sf_path, "rb") as f:
                hlen = _struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(hlen))
                data_offset = 8 + hlen
                for k, meta in header.items():
                    if k == "__metadata__":
                        continue
                    self._index[k] = (
                        sf_path,
                        data_offset,
                        meta["data_offsets"][0],
                        meta["data_offsets"][1],
                        tuple(meta["shape"]),
                        meta["dtype"],
                    )
        self._fp8_pairs = {}
        self._fp8_scale_keys = set()
        self._src_quant = {}
        self._discover_fp8_pairs()

    def _discover_fp8_pairs(self):
        seen = set()
        for k in list(self._index):
            if k.endswith("_scale_inv"):
                wk = k[: -len("_scale_inv")]
                if (
                    wk in self._index
                    and wk not in seen
                    and self._index[wk][5] in _FP8_WEIGHT_DTYPES
                ):
                    self._fp8_pairs[wk] = k
                    seen.add(wk)
            elif k.endswith(".scale"):
                wk = k[: -len(".scale")] + ".weight"
                if (
                    wk in self._index
                    and wk not in seen
                    and self._index[wk][5] in _FP8_WEIGHT_DTYPES
                ):
                    self._fp8_pairs[wk] = k
                    seen.add(wk)
        self._fp8_scale_keys = set(self._fp8_pairs.values())
        for wk, sk in self._fp8_pairs.items():
            info = self._classify_pair(wk, sk)
            if info is not None:
                self._src_quant[wk] = info
        if self._fp8_pairs:
            logger.info(
                f"FP8 on-the-fly dequant: {len(self._fp8_pairs)} weight+scale pairs detected"
            )

    def _classify_pair(self, wk, sk):
        """Classify a weight+scale pair into an mlx-native quantized format.

        Returns a dict {kind, bits, group_size, mode} when the pair can be
        passed through to the output in mlx packed form (mxfp4/mxfp8), or
        None for layouts we only support via dequantization (E5M2, float
        scales, plain int8 block quant, _scale_inv pairs, ...).
        """
        w_shape, w_dtype = self._index[wk][4], self._index[wk][5]
        s_shape, s_dtype = self._index[sk][4], self._index[sk][5]
        if len(w_shape) != 2 or len(s_shape) != 2 or not sk.endswith(".scale"):
            return None
        rows, cols = w_shape
        # FP4-packed experts (DeepSeek V4): int8 bytes carry 2 fp4 values
        # each, e8m0 scale per 32 logical values -> 16 bytes per group.
        if (
            w_dtype == "I8"
            and s_dtype == "F8_E8M0"
            and cols % 16 == 0
            and tuple(s_shape) == (rows, cols // 16)
        ):
            return {"kind": "mxfp4", "bits": 4, "group_size": 32, "mode": "mxfp4"}
        # FP8 block quant (e4m3 weight, e8m0 block scale): representable as
        # mxfp8 g32 after expanding the block scale per 32-column group.
        if (
            w_dtype == "F8_E4M3"
            and s_dtype == "F8_E8M0"
            and s_shape[0] > 0
            and s_shape[1] > 0
            and rows % s_shape[0] == 0
            and cols % s_shape[1] == 0
            and (cols // s_shape[1]) % 32 == 0
            and cols % 4 == 0
        ):
            return {"kind": "fp8_block", "bits": 8, "group_size": 32, "mode": "mxfp8"}
        return None

    def _dequant_one(self, wk):
        sk = self._fp8_pairs[wk]
        w_meta = self._index[wk]
        s_meta = self._index[sk]
        w_lt = _LazyTensor(
            w_meta[0], w_meta[1], w_meta[2], w_meta[3], w_meta[4], w_meta[5]
        )
        s_lt = _LazyTensor(
            s_meta[0], s_meta[1], s_meta[2], s_meta[3], s_meta[4], s_meta[5]
        )
        weight_raw = w_lt[:]
        scale_raw = s_lt[:]
        mx.eval(weight_raw, scale_raw)
        info = self._src_quant.get(wk)
        if info is not None and info["kind"] == "mxfp4":
            # FP4-packed: reinterpret bytes as the mlx mxfp4 packed layout
            # and let mx.dequantize unpack (e8m0 uint8 scales, group 32).
            weight = mx.dequantize(
                weight_raw.view(mx.uint32),
                scale_raw,
                None,
                group_size=32,
                bits=4,
                mode="mxfp4",
            ).astype(mx.bfloat16)
        else:
            weight = _block_dequant_fp8(weight_raw, scale_raw, w_meta[5], s_meta[5])
        del weight_raw, scale_raw
        mx.clear_cache()
        return weight

    def _load_packed(self, wk):
        """Load a passthrough-capable pair in mlx packed quantized form.

        Returns (weight, scales): uint32-packed weight plus uint8 e8m0
        scales matching the pair's {bits, group_size, mode} from
        source_quant_info. fp8_block scales are expanded from per-block to
        per-32-column-group, mirroring the model sanitize's repeat expansion.
        """
        sk = self._fp8_pairs[wk]
        info = self._src_quant[wk]
        weight_raw = self._load_raw(wk)
        scale_raw = self._load_raw(sk)
        packed = weight_raw.view(mx.uint32)
        if info["kind"] == "fp8_block":
            rows, cols = self._index[wk][4]
            sm, sn = self._index[sk][4]
            row_rep = rows // sm
            col_rep = (cols // 32) // sn
            if col_rep > 1:
                scale_raw = mx.repeat(scale_raw, col_rep, -1)
            if row_rep > 1:
                scale_raw = mx.repeat(scale_raw, row_rep, 0)
        mx.eval(packed, scale_raw)
        return packed, scale_raw

    def source_quant_info(self, key):
        """Pre-quantized source metadata for a weight key, or None."""
        return self._src_quant.get(key)

    def _is_visible(self, k):
        return k not in self._fp8_scale_keys

    def logical_metadata(self):
        """Metadata for plan discovery: FP8 weights report as BF16, scale keys hidden."""
        result = {}
        for k, meta in self._index.items():
            if k in self._fp8_scale_keys:
                continue
            shape, dtype = meta[4], meta[5]
            if k in self._fp8_pairs:
                dtype = "BF16"
                info = self._src_quant.get(k)
                if info is not None and info["kind"] == "mxfp4":
                    # FP4-packed bytes: logical width is 2 values per byte.
                    shape = (shape[0], shape[1] * 2)
            result[k] = (shape, dtype)
        return result

    def keys(self):
        base = [k for k in self._index if self._is_visible(k)]
        if hasattr(self, "_overrides"):
            base.extend(self._overrides.keys())
        return base

    def __len__(self):
        n = sum(1 for k in self._index if self._is_visible(k))
        if hasattr(self, "_overrides"):
            n += len(self._overrides)
        return n

    def __contains__(self, k):
        if k in self._index and self._is_visible(k):
            return True
        return hasattr(self, "_overrides") and k in self._overrides

    def __iter__(self):
        for k in self._index:
            if self._is_visible(k):
                yield k
        if hasattr(self, "_overrides"):
            for k in self._overrides:
                if k not in self._index:
                    yield k

    def nbytes(self):
        return sum(
            e - s
            for k, (_, _, s, e, _, _) in self._index.items()
            if self._is_visible(k)
        )

    def _load_raw(self, key):
        sf_path, data_offset, start, end, shape, dtype = self._index[key]
        lt = _LazyTensor(sf_path, data_offset, start, end, shape, dtype)
        return lt[:]

    def __getitem__(self, key):
        if hasattr(self, "_overrides") and key in self._overrides:
            return self._overrides[key]
        if key not in self._index:
            raise KeyError(key)
        if key in self._fp8_pairs:
            return self._dequant_one(key)
        return self._load_raw(key)

    def items(self):
        for k in list(self._index.keys()):
            if not self._is_visible(k):
                continue
            yield k, self[k]
            mx.clear_cache()
        if hasattr(self, "_overrides"):
            for k, v in self._overrides.items():
                yield k, v

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __setitem__(self, key, value):
        if not hasattr(self, "_overrides"):
            self._overrides = {}
        self._overrides[key] = value
        self._index.pop(key, None)
        self._fp8_pairs.pop(key, None)
        self._src_quant.pop(key, None)

    def __delitem__(self, key):
        if key in self._fp8_pairs:
            sk = self._fp8_pairs.pop(key)
            self._fp8_scale_keys.discard(sk)
            self._index.pop(sk, None)
        self._index.pop(key, None)
        self._src_quant.pop(key, None)
        if hasattr(self, "_overrides"):
            self._overrides.pop(key, None)

    def update(self, other):
        if hasattr(other, "items"):
            for k, v in other.items():
                self[k] = v
        else:
            for k, v in other:
                self[k] = v

    def pop(self, key, *default):
        if hasattr(self, "_overrides") and key in self._overrides:
            return self._overrides.pop(key)
        if key not in self._index:
            if default:
                return default[0]
            raise KeyError(key)
        if key in self._fp8_pairs:
            result = self._dequant_one(key)
            sk = self._fp8_pairs.pop(key)
            self._fp8_scale_keys.discard(sk)
            self._src_quant.pop(key, None)
            self._index.pop(key, None)
            self._index.pop(sk, None)
            return result
        sf_path, data_offset, start, end, shape, dtype = self._index.pop(key)
        lt = _LazyTensor(sf_path, data_offset, start, end, shape, dtype)
        arr = lt[:]
        mx.eval(arr)
        return arr


class _LazyTensor:
    def __init__(self, sf_path, data_offset, start, end, shape, dtype):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self._sf_path = sf_path
        self._data_offset = data_offset
        self._start = start
        self._end = end
        self._dtype = dtype
        self._bpe = _LazyTensorIndex._DTYPE_BYTES.get(dtype, 2)
        self._epr = 1
        for d in self.shape[1:]:
            self._epr *= d
        self._bpr = self._epr * self._bpe

    @property
    def size(self):
        s = 1
        for d in self.shape:
            s *= d
        return s

    @property
    def nbytes(self):
        return self._end - self._start

    _SF_TO_MLX = {
        "BF16": mx.bfloat16,
        "F16": mx.float16,
        "F32": mx.float32,
        "I8": mx.int8,
        "U8": mx.uint8,
        "I16": mx.int16,
        "U16": mx.uint16,
        "I32": mx.int32,
        "U32": mx.uint32,
        "I64": mx.int64,
        "U64": mx.uint64,
        "F8_E4M3": mx.uint8,
        "F8_E5M2": mx.uint8,
        "F8_E8M0": mx.uint8,
        "BOOL": mx.bool_,
    }

    _SF_TO_NP = {
        "BF16": _np.uint16,
        "F16": _np.float16,
        "F32": _np.float32,
        "F64": _np.float64,
        "I8": _np.int8,
        "U8": _np.uint8,
        "I16": _np.int16,
        "U16": _np.uint16,
        "I32": _np.int32,
        "U32": _np.uint32,
        "I64": _np.int64,
        "U64": _np.uint64,
        "F8_E4M3": _np.uint8,
        "F8_E5M2": _np.uint8,
        "F8_E8M0": _np.uint8,
        "BOOL": _np.bool_,
    }

    def _mlx_dtype(self):
        return self._SF_TO_MLX.get(self._dtype, mx.bfloat16)

    def _np_view_dtype(self):
        return self._SF_TO_NP.get(self._dtype, _np.uint16)

    def _load_rows(self, r0, r1):
        n = r1 - r0
        if n <= 0:
            return mx.zeros((0, *self.shape[1:]), dtype=self._mlx_dtype())
        b0 = self._start + r0 * self._bpr
        b1 = self._start + r1 * self._bpr
        with open(self._sf_path, "rb") as f:
            f.seek(self._data_offset + b0)
            raw = f.read(b1 - b0)
        arr = _np.frombuffer(raw, dtype=self._np_view_dtype())
        chunk_shape = (n, *self.shape[1:])
        # Two ceilings: device buffer bytes, and MLX's int32 element count.
        _MLX_MAX_ELEMS = 1 << 30
        max_rows_bytes = max(1, _LOAD_CHUNK_BYTES // max(self._bpr, 1))
        max_rows_elems = max(1, _MLX_MAX_ELEMS // max(self._epr, 1))
        max_rows = min(max_rows_bytes, max_rows_elems)
        dt = self._mlx_dtype()
        if n <= max_rows:
            t = mx.array(arr).view(dt).reshape(chunk_shape)
            mx.eval(t)
            return t
        parts = []
        epc = max_rows * self._epr
        for s in range(0, arr.size, epc):
            sub = arr[s : s + epc]
            sr = sub.size // self._epr
            t = mx.array(sub).view(dt).reshape((sr, *self.shape[1:]))
            mx.eval(t)
            parts.append(t)
            mx.clear_cache()
        result = mx.concatenate(parts, axis=0)
        mx.eval(result)
        return result

    def __getitem__(self, idx):
        if len(self.shape) == 0:
            raise IndexError(
                "0-dim _LazyTensor cannot be indexed; caller should use "
                "_materialize_source scalar path"
            )
        if isinstance(idx, tuple):
            return self._load_rows(0, self.shape[0])[idx]
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = self.shape[0] if idx.stop is None else idx.stop
            return self._load_rows(start, stop)
        return self._load_rows(idx, idx + 1)


