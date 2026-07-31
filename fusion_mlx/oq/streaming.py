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
import shutil
import tempfile
import time as _time
from collections.abc import Callable
from pathlib import Path

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


from ._core import _discover_sanitize_plan, _DiscoveredPlan
from .io import (
    _MAX_SHARD_BYTES,
    _QUANTIZE_CHUNK_BYTES,
    _build_model_sanitizer,
    _build_non_quantizable_set,
    _cast_passthrough_tensor,
    _copy_model_sidecars,
    _get_predicate_bits,
    _gs_for_mode,
    _is_mtp_tensor,
    _LazyTensor,
    _LazyTensorIndex,
    _mode_for_bits,
    _normalize_mtp_in_config,
    _should_quantize_tensor,
)
from .levels import _sensitivity_lm_config_override
from .plan import (
    _base_bits_for_level,
    _bpw_targets_for_level,
    _build_quant_plan,
    _collect_named_weight_shapes_from_weights,
    _is_audio_tensor,
    _is_vision_tensor,
    _normalize_quant_path,
    _validate_oq_dtype_for_model,
    universal_quant_predicate,
)


def _tensor_shape_nbytes(shape, bytes_per_element: int) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n * bytes_per_element


def _progress_total_bytes(all_weights, source: Path) -> int:
    """Conservative denominator for streaming quantization progress.

    Packed or transformed checkpoints can expose a logical tensor view that
    is larger than the physical source shards. Using only source file sizes
    lets progress exceed 100% and makes ETA negative.
    """
    candidates = [
        sum(sf.stat().st_size for sf in source.glob("*.safetensors")),
    ]

    if hasattr(all_weights, "nbytes"):
        try:
            candidates.append(int(all_weights.nbytes()))
        except Exception:
            pass

    if hasattr(all_weights, "logical_metadata"):
        try:
            logical_total = 0
            for shape, dtype in all_weights.logical_metadata().values():
                logical_total += _tensor_shape_nbytes(
                    shape, _LazyTensorIndex._DTYPE_BYTES.get(dtype, 2)
                )
            candidates.append(logical_total)
        except Exception:
            pass

    plan = getattr(all_weights, "_plan", None)
    if isinstance(plan, dict):
        try:
            # _DiscoveredPlan entries no longer carry dtype. Two bytes per
            # element matches the BF16 logical view used for packed sources,
            # while source-file bytes remain a lower bound for wider tensors.
            candidates.append(
                sum(_tensor_shape_nbytes(info["shape"], 2) for info in plan.values())
            )
        except Exception:
            pass

    return max(1, *candidates)


def _row_chunks(t, max_elems):
    rows = t.shape[0]
    if rows == 0:
        return
    epr = max(1, t.size // rows)
    rpc = max(1, max_elems // epr)
    for r0 in range(0, rows, rpc):
        r1 = min(rows, r0 + rpc)
        if isinstance(t, _LazyTensor):
            chunk = t._load_rows(r0, r1)
        else:
            chunk = t[r0:r1]
            mx.eval(chunk)
        yield chunk


def _quantize_chunked(w, group_size, bits, mode):
    _MLX_MAX_ELEMS = 1 << 30
    max_elems = max(group_size, min(_QUANTIZE_CHUNK_BYTES // 2, _MLX_MAX_ELEMS))
    if not isinstance(w, _LazyTensor) and w.size <= max_elems:
        qw, scales, *rest = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
        return qw, scales, (rest[0] if rest else None)
    orig = tuple(w.shape)
    qws, scs, bis = [], [], []
    for chunk in _row_chunks(w, max_elems):
        flat = chunk.reshape(-1, chunk.shape[-1])
        mx.eval(flat)
        cqw, csc, *crest = mx.quantize(
            flat, group_size=group_size, bits=bits, mode=mode
        )
        mx.eval(cqw, csc)
        qws.append(cqw)
        scs.append(csc)
        if crest:
            bis.append(crest[0])
        mx.synchronize()
        mx.clear_cache()
    qw = mx.concatenate(qws, axis=0)
    scales = mx.concatenate(scs, axis=0)
    biases = mx.concatenate(bis, axis=0) if bis else None
    mx.eval(qw, scales)
    flat_rows = 1
    for d in orig[:-1]:
        flat_rows *= d
    if qw.shape[0] == flat_rows and len(orig) > 2:
        qw = qw.reshape(*orig[:-1], -1)
        scales = scales.reshape(*orig[:-1], -1)
        if biases is not None:
            biases = biases.reshape(*orig[:-1], -1)
    return qw, scales, biases


# --- end chunked-quantize helpers ---


def quantize_oq_streaming(
    model_path: str,
    output_path: str,
    oq_level: int,
    group_size: int = 64,
    progress_callback: Callable[[str, float], None] | None = None,
    text_only: bool = False,
    target_bpw: float | None = None,
    hard_cap_bpw: float | None = None,
    sensitivity_model_path: str = "",
    dtype: str = "bfloat16",
    preserve_mtp: bool = False,
    auto_proxy_sensitivity: bool = True,
    trust_remote_code: bool = False,
) -> None:
    """Tensor-by-tensor quantization. Memory: ~3-4GB regardless of model size.

    Reads tensors one at a time from safetensors, quantizes with the universal
    predicate, and writes output shards. Never loads the full model.

    Args:
        model_path: Path to source model directory.
        output_path: Path for output (must not exist).
        oq_level: Quantization level (2, 3, 4, 6, or 8).
        group_size: Default quantization group size.
        progress_callback: Optional fn(phase_name, progress_pct) for updates.
        text_only: Skip vision encoder weights for VLM models.
        dtype: Target fp dtype for non-quantized weights and quant scales/biases.
            Must be "bfloat16" (default) or "float16". float16 yields ~20%
            faster prefill on M1/M2 Apple Silicon (native fp16 support), but
            is unsupported for DeepSeek V4.
        preserve_mtp: Keep mtp.* tensors and config fields in the output so
            the Native MTP toggle works after quantization. Stashes mtp.*
            keys around the model.sanitize() call (which would otherwise
            strip them) and re-merges. When False (default), mtp.* tensors
            are stripped *and* the output config's mtp_num_hidden_layers /
            num_nextn_predict_layers are normalized to 0 to keep the
            quantized model self-consistent.
        auto_proxy_sensitivity: When True (default) and the source model
            exceeds available RAM, automatically build a temporary uniform
            4-bit proxy on disk and run sensitivity measurement on it,
            preserving oQ's data-driven mixed-precision allocation. When
            False, the quantization aborts on RAM-exceeding models with a
            RuntimeError so callers always get a real sensitivity-driven
            output. Ignored if sensitivity_model_path is set explicitly.
        trust_remote_code: Forwarded to mlx-lm/mlx-vlm model loads when a
            checkpoint requires custom model code.
    """
    if oq_level not in OQ_LEVELS:
        raise ValueError(
            f"Invalid oQ level {oq_level}. Must be one of {sorted(OQ_LEVELS)}"
        )
    if dtype not in OQ_DTYPES:
        raise ValueError(f"Invalid dtype {dtype!r}. Must be one of {OQ_DTYPES}")
    target_dtype = mx.bfloat16 if dtype == "bfloat16" else mx.float16

    source = Path(model_path)
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"Output directory already exists: {output_path}")

    cb = progress_callback or (lambda phase, pct: None)

    config_path = source / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    _validate_oq_dtype_for_model(config, dtype)
    config["_oq_use_budget_plan"] = oq_level in _OQ_BPW_TARGETS

    output.mkdir(parents=True, exist_ok=True)

    cb("loading", 5.0)

    weight_files = sorted(source.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"No .safetensors files found in {model_path}")

    cb("loading", 8.0)

    all_weights = _LazyTensorIndex(weight_files)
    if preserve_mtp and not any(_is_mtp_tensor(k) for k in all_weights):
        logger.warning(
            "Preserve MTP requested for %s, but no mtp.* tensors were found "
            "in the checkpoint; disabling MTP preservation",
            source.name,
        )
        preserve_mtp = False

    logger.info(
        f"oQ{oq_level:g} streaming: {len(all_weights)} tensors in "
        f"{len(weight_files)} shards"
    )

    sensitivity_map_path = Path(model_path, "oq_sensitivity_map.json")
    from fusion_mlx.pool.settings import get_system_memory as _get_system_memory

    _model_bytes = all_weights.nbytes()
    _system_ram = _get_system_memory()
    _model_exceeds_ram = _model_bytes > int(_system_ram * _MAX_MODEL_RAM_FRACTION)
    if _model_exceeds_ram:
        logger.info(
            f"oQ{oq_level:g}: model size ({_model_bytes / 1e9:.1f} GB) exceeds "
            f"80% of system RAM ({_system_ram / 1e9:.1f} GB), "
            "OOM-prone paths will be skipped"
        )

    cb("loading", 12.0)

    if sensitivity_map_path.exists():
        sensitivity_map = json.loads(sensitivity_map_path.read_text(encoding="utf-8"))
        logger.info(f"{sensitivity_map_path} found, skipping measuring.")
    else:
        # --- Sensitivity measurement (before sanitize-plan discovery) ---------
        # Must run before _build_model_sanitizer + _discover_sanitize_plan,
        # because the discovery pass feeds _TrackedTensor proxies through
        # Model.sanitize which corrupts mutable state in the MTP sanitize
        # patch (weights.pop on tracked objects). Running sensitivity first
        # ensures vlm_load_model sees a pristine patch chain.
        if sensitivity_model_path:
            logger.info(f"oQ{oq_level:g}: measuring sensitivity via proxy model")
            sensitivity_map = _measure_sensitivity_from_quantized_model(
                sensitivity_model_path,
                config,
                oq_level,
                num_samples=128,
                seq_length=256,
                trust_remote_code=trust_remote_code,
            )
        elif (
            not _model_exceeds_ram
            and str(config.get("model_type", "")).startswith("deepseek_v4")
            and isinstance(config.get("quantization_config"), dict)
            and config["quantization_config"].get("quant_method") == "fp8"
        ):
            # Native-fp8 source (e.g. DeepSeek-V4-Flash): the checkpoint
            # loads as a quantized model (mxfp4 experts / mxfp8 attention),
            # so the raw qdq measurement would only perturb the few float
            # Linears. Measure on the source itself with the re-quantization
            # perturbation instead.
            logger.info(
                f"oQ{oq_level:g}: pre-quantized fp8 source, measuring "
                "sensitivity on source"
            )
            sensitivity_map = _measure_sensitivity_from_quantized_model(
                model_path,
                config,
                oq_level,
                num_samples=128,
                seq_length=256,
                trust_remote_code=trust_remote_code,
            )
        elif _model_exceeds_ram and auto_proxy_sensitivity:
            logger.warning(
                f"oQ{oq_level:g}: model size ({_model_bytes / 1e9:.1f} GB) exceeds "
                f"{int(_MAX_MODEL_RAM_FRACTION * 100)}% of system RAM "
                f"({_system_ram / 1e9:.1f} GB). Auto-building a uniform "
                f"{_PROXY_QUANT_BITS}-bit proxy on disk so sensitivity "
                "measurement stays data-driven."
            )
            _proxy_dir: Path | None = None
            try:
                _proxy_dir = _build_proxy_for_sensitivity(
                    model_path,
                    config=config,
                    dtype=dtype,
                    working_dir=str(output.parent),
                    trust_remote_code=trust_remote_code,
                )
                logger.info(
                    f"oQ{oq_level:g}: proxy ready at {_proxy_dir}, measuring sensitivity"
                )
                sensitivity_map = _measure_sensitivity_from_quantized_model(
                    str(_proxy_dir),
                    config,
                    oq_level,
                    num_samples=128,
                    seq_length=256,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as e:
                raise RuntimeError(
                    f"oQ{oq_level:g}: auto-proxy sensitivity failed ({e}). "
                    "Pass sensitivity_model_path with a pre-quantized version "
                    "of this model, or run on a machine with enough RAM for "
                    "full-fp16 sensitivity measurement."
                ) from e
            finally:
                if _proxy_dir is not None and _proxy_dir.exists():
                    shutil.rmtree(_proxy_dir, ignore_errors=True)
                    logger.info(f"oQ{oq_level:g}: cleaned up proxy at {_proxy_dir}")
        elif _model_exceeds_ram:
            raise RuntimeError(
                f"oQ{oq_level:g}: model exceeds {int(_MAX_MODEL_RAM_FRACTION * 100)}% "
                "of system RAM and auto_proxy_sensitivity is disabled. "
                "Enable auto_proxy_sensitivity, pass sensitivity_model_path "
                "with a pre-quantized version of this model, or run on a "
                "machine with enough RAM."
            )
        else:
            logger.info(
                f"oQ{oq_level:g}: measuring layer sensitivity for streaming path"
            )
            sensitivity_map = _measure_sensitivity(
                model_path,
                config,
                oq_level,
                num_samples=128,
                seq_length=256,
                trust_remote_code=trust_remote_code,
            )

    # Single enforcement point. Inner measurement helpers may return {} on
    # load / calibration / layer-discovery failure; treat that as a hard
    # error here so the rest of quantize_oq_streaming never runs without a
    # data-driven sensitivity map.
    if not sensitivity_map:
        raise RuntimeError(
            f"oQ{oq_level:g}: sensitivity measurement produced no scores. "
            "Check the preceding log lines for the root cause (model load, "
            "calibration data, or layer discovery), and either fix it or "
            "pass an explicit sensitivity_model_path."
        )

    cb("loading", 15.0)

    # --- Sanitize-plan discovery ------------------------------------------
    sanitize_fn = _build_model_sanitizer(config, text_only=text_only)
    cast_predicate = getattr(sanitize_fn, "_fmlx_cast_predicate", None)
    # When preserve_mtp is True, the patched sanitize functions
    # (mlx_lm_mtp/qwen35_model.py and mlx_vlm_mtp/qwen35_vlm_model.py)
    # keep mtp.* in the output and apply the +1 RMSNorm shift to MTP
    # norms. No stash/merge wrapper needed — the patch covers both paths.
    if sanitize_fn is not None:
        try:
            plan = _discover_sanitize_plan(sanitize_fn, all_weights)
            all_weights = _DiscoveredPlan(plan, all_weights)
            logger.info(
                f"oQ{oq_level:g}: discovered streaming sanitize plan, "
                f"{len(all_weights)} output tensors"
            )
        except Exception as e:
            if _model_exceeds_ram:
                raise RuntimeError(
                    f"oQ{oq_level:g}: streaming sanitize-plan discovery "
                    f"failed ({e}) and the eager fallback is unsafe with "
                    f"model size {_model_bytes / 1e9:.1f} GB exceeding "
                    f"{int(_MAX_MODEL_RAM_FRACTION * 100)}% of system RAM "
                    f"({_system_ram / 1e9:.1f} GB). Run on a machine with "
                    "enough RAM, or extend _TrackedTensor to cover the "
                    "indexing pattern the sanitize uses."
                ) from e
            logger.warning(
                f"Streaming discovery failed ({e}), falling back to eager sanitize"
            )
            try:
                all_weights = sanitize_fn(all_weights)
                logger.info(
                    f"oQ{oq_level:g}: eager sanitize applied, {len(all_weights)} tensors"
                )
            except Exception as e2:
                logger.warning(f"Sanitize failed ({e2}), using original names")

    config["_oq_non_quantizable"] = _build_non_quantizable_set(config)
    config["_oq_sensitivity_map"] = {str(k): v for k, v in sensitivity_map.items()}
    logger.info(f"oQ{oq_level:g}: sensitivity applied ({len(sensitivity_map)} layers)")

    named_shapes = _collect_named_weight_shapes_from_weights(all_weights)
    if text_only:
        named_shapes = {
            k: v
            for k, v in named_shapes.items()
            if not _is_vision_tensor(k) and not _is_audio_tensor(k)
        }
    if not preserve_mtp:
        # Match the eager path (_should_skip_tensor): when MTP heads are
        # not being preserved, drop ``mtp.*`` tensors from the plan so the
        # quantizer doesn't reserve bits for them and the output shards
        # don't include them. Otherwise the output would carry the source
        # mtp.* weights while the config's mtp_num_hidden_layers gets
        # zeroed by _normalize_mtp_in_config — a config/weights mismatch
        # that breaks VLM load with "Received N parameters not in model".
        named_shapes = {k: v for k, v in named_shapes.items() if not _is_mtp_tensor(k)}
    # Pre-quantized source tensors whose pre-boost target bits already cover
    # the source precision are passed through in their packed form. Price
    # them at their true cost and keep them out of the boost competition.
    # Boosts only ever raise bits, so the passthrough decision is monotone.
    fixed_overrides = {}
    if hasattr(all_weights, "pop_packed"):
        _pre_boost_config = {**config, "_oq_boost_map": {}}
        for _path in named_shapes:
            _info = all_weights.source_quant_info(f"{_path}.weight")
            if _info is None:
                continue
            _floor_bits, _, _ = _get_predicate_bits(
                f"{_path}.weight", _pre_boost_config, oq_level, group_size
            )
            if _floor_bits is not None and _floor_bits >= _info["bits"]:
                fixed_overrides[_path] = {
                    "bits": _info["bits"],
                    "group_size": _info["group_size"],
                    "mode": _info["mode"],
                }
        if fixed_overrides:
            logger.info(
                f"oQ{oq_level:g}: {len(fixed_overrides)} pre-quantized tensors "
                "will pass through in source precision"
            )
    _level_targets = _bpw_targets_for_level(oq_level)
    if _level_targets is not None:
        _t = target_bpw if target_bpw is not None else _level_targets[0]
        _c = hard_cap_bpw if hard_cap_bpw is not None else _level_targets[1]
        plan = _build_quant_plan(
            named_shapes,
            config,
            oq_level,
            target_bpw=_t,
            hard_cap_bpw=_c,
            fixed_overrides=fixed_overrides,
        )
        config["_oq_boost_map"] = plan.boost_map
        logger.info(
            f"oQ{oq_level:g}: quant plan -> {plan.effective_bpw:.2f} bpw "
            f"with {len(plan.boost_map)} boosts"
        )
    else:
        config["_oq_boost_map"] = {}

    cb("loading", 20.0)

    tensor_names = list(all_weights.keys())
    out_shard_data = {}
    out_shard_idx = 0
    weight_map = {}
    base_bits = _base_bits_for_level(oq_level)
    base_mode = _mode_for_bits(base_bits)
    base_gs = _gs_for_mode(base_bits, group_size)
    quantization_config = {"group_size": base_gs, "bits": base_bits, "mode": base_mode}
    per_layer_config = {}
    start_time = _time.monotonic()

    total_bytes = _progress_total_bytes(all_weights, source)
    processed_bytes = 0

    for tensor_name in tensor_names:
        # Pre-quantized source tensor at or below the target precision:
        # emit the packed mxfp4/mxfp8 form unchanged (no dequant-requant).
        handled_packed = False
        if (
            hasattr(all_weights, "pop_packed")
            and not (
                text_only
                and (_is_vision_tensor(tensor_name) or _is_audio_tensor(tensor_name))
            )
            and not (not preserve_mtp and _is_mtp_tensor(tensor_name))
        ):
            src_info = all_weights.source_quant_info(tensor_name)
            if src_info is not None and _should_quantize_tensor(
                tensor_name, all_weights.plan_shape(tensor_name)
            ):
                bits, gs, qmode = _get_predicate_bits(
                    tensor_name, config, oq_level, group_size
                )
                if bits is not None and bits >= src_info["bits"]:
                    qw, scales = all_weights.pop_packed(tensor_name)
                    tensor_bytes = qw.nbytes + scales.nbytes
                    base = tensor_name[: -len(".weight")]
                    out_shard_data[f"{base}.weight"] = qw
                    out_shard_data[f"{base}.scales"] = scales
                    per_layer_config[base] = {
                        "bits": src_info["bits"],
                        "group_size": src_info["group_size"],
                        "mode": src_info["mode"],
                    }
                    del qw, scales
                    handled_packed = True

        if not handled_packed:
            w_mx = all_weights.pop(tensor_name)
            if isinstance(w_mx, _LazyTensor):
                w_mx = w_mx[:]
            tensor_bytes = w_mx.nbytes
            shape = w_mx.shape

            if text_only and (
                _is_vision_tensor(tensor_name) or _is_audio_tensor(tensor_name)
            ):
                del w_mx
                processed_bytes += tensor_bytes
                continue

            if not preserve_mtp and _is_mtp_tensor(tensor_name):
                # Strip MTP tensors when the caller asked not to preserve them.
                # _normalize_mtp_in_config will zero mtp_num_hidden_layers in
                # the output config so the result stays self-consistent.
                del w_mx
                processed_bytes += tensor_bytes
                continue

            if _should_quantize_tensor(tensor_name, shape):
                bits, gs, qmode = _get_predicate_bits(
                    tensor_name, config, oq_level, group_size
                )

                if bits is not None and len(shape) >= 2 and shape[-1] % gs == 0:
                    # Cast to target dtype before quantize: scales/biases inherit
                    # the input dtype, which drives inference speed on Apple
                    # Silicon (M1/M2 prefer float16, M3/M4 handle both).
                    if (
                        mx.issubdtype(w_mx.dtype, mx.floating)
                        and w_mx.dtype != target_dtype
                    ):
                        w_mx = w_mx.astype(target_dtype)
                    qw, scales, biases = _quantize_chunked(w_mx, gs, bits, qmode)

                    base = tensor_name
                    if base.endswith(".weight"):
                        base = base[:-7]

                    out_shard_data[f"{base}.weight"] = qw
                    out_shard_data[f"{base}.scales"] = scales
                    if biases is not None:
                        out_shard_data[f"{base}.biases"] = biases

                    base_qmode = _mode_for_bits(base_bits)
                    base_gs_check = _gs_for_mode(base_bits, group_size)
                    if bits != base_bits or gs != base_gs_check or qmode != base_qmode:
                        layer_cfg = {"bits": bits, "group_size": gs}
                        layer_cfg["mode"] = qmode
                        per_layer_config[base] = layer_cfg
                else:
                    if cast_predicate is None or cast_predicate(tensor_name):
                        w_mx = _cast_passthrough_tensor(tensor_name, w_mx, target_dtype)
                    out_shard_data[tensor_name] = w_mx
            else:
                if cast_predicate is None or cast_predicate(tensor_name):
                    w_mx = _cast_passthrough_tensor(tensor_name, w_mx, target_dtype)
                out_shard_data[tensor_name] = w_mx

            del w_mx

        current_bytes = sum(v.nbytes for v in out_shard_data.values())
        if current_bytes >= _MAX_SHARD_BYTES:
            shard_name = f"model-{out_shard_idx + 1:05d}-of-PLACEHOLDER.safetensors"
            shard_path = output / shard_name
            mx.save_safetensors(
                str(shard_path), out_shard_data, metadata={"format": "mlx"}
            )
            for k in out_shard_data:
                weight_map[k] = shard_name
            out_shard_idx += 1
            out_shard_data = {}
            mx.synchronize()
            mx.clear_cache()
            logger.info(f"oQ{oq_level:g}: wrote output shard {out_shard_idx}")

        processed_bytes += tensor_bytes
        elapsed = _time.monotonic() - start_time
        frac = min(max(processed_bytes / max(total_bytes, 1), 0.0), 1.0)
        pct = 15.0 + frac * 75.0
        display_pct = min(100, max(0, int(frac * 100)))
        if elapsed > 1.0 and frac > 0.01:
            eta_secs = max(0.0, elapsed / frac * (1.0 - frac))
            mins = int(eta_secs // 60)
            secs = int(eta_secs % 60)
            cb(
                f"quantizing_eta|{display_pct}|100|{mins}:{secs:02d}",
                pct,
            )
        else:
            cb(f"quantizing_eta|{display_pct}|100|", pct)

    del all_weights
    mx.synchronize()
    mx.clear_cache()

    if out_shard_data:
        total_shards = out_shard_idx + 1
        if total_shards == 1:
            shard_name = "model.safetensors"
        else:
            shard_name = f"model-{out_shard_idx + 1:05d}-of-PLACEHOLDER.safetensors"
        shard_path = output / shard_name
        mx.save_safetensors(str(shard_path), out_shard_data, metadata={"format": "mlx"})
        for k in out_shard_data:
            weight_map[k] = shard_name
        out_shard_idx += 1
        del out_shard_data

    total_shards = out_shard_idx
    if total_shards > 1:
        for i in range(total_shards):
            old_name = f"model-{i + 1:05d}-of-PLACEHOLDER.safetensors"
            new_name = f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
            old_path = output / old_name
            new_path = output / new_name
            if old_path.exists():
                old_path.rename(new_path)
                for k, v in weight_map.items():
                    if v == old_name:
                        weight_map[k] = new_name

    cb("saving", 92.0)

    if total_shards > 1:
        total_size = sum(f.stat().st_size for f in output.glob("*.safetensors"))
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        with open(output / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    output_config = dict(config)
    for temp_key in (
        "_oq_sensitivity_map",
        "_oq_boost_map",
        "_oq_use_budget_plan",
        "_oq_non_quantizable",
    ):
        output_config.pop(temp_key, None)
    if text_only:
        for key in (
            "vision_config",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "audio_config",
            "audio_token_id",
            "boa_token_id",
            "eoa_token_id",
            "eoa_token_index",
        ):
            output_config.pop(key, None)
    if not preserve_mtp:
        # Default path: zero out MTP layer counts so the quantized model
        # doesn't claim to have an MTP head while its weights have been
        # stripped. This keeps the output self-consistent — mtp_enabled
        # toggle's compatibility check (_has_mtp_heads) reads these
        # fields and will correctly report "no MTP heads" instead of
        # crashing during model.load_weights() with the cryptic
        # "Missing N parameters" error.
        _normalize_mtp_in_config(output_config)
    # Ensure eos_token_id is present (mlx-lm adds it from tokenizer)
    if "eos_token_id" not in output_config:
        try:
            from transformers import AutoTokenizer

            _tok = AutoTokenizer.from_pretrained(str(source))
            if hasattr(_tok, "eos_token_id") and _tok.eos_token_id is not None:
                # Some models have multiple EOS tokens
                eos_ids = getattr(_tok, "additional_special_tokens_ids", [])
                if _tok.eos_token_id not in eos_ids:
                    eos_ids = [_tok.eos_token_id] + eos_ids
                # Check generation_config for eos_token_id list
                gen_config_path = source / "generation_config.json"
                if gen_config_path.exists():
                    with open(gen_config_path) as f:
                        gen_cfg = json.load(f)
                    if "eos_token_id" in gen_cfg:
                        output_config["eos_token_id"] = gen_cfg["eos_token_id"]
                        logger.info(
                            f"Added eos_token_id from generation_config: {gen_cfg['eos_token_id']}"
                        )
                elif eos_ids:
                    output_config["eos_token_id"] = (
                        eos_ids if len(eos_ids) > 1 else eos_ids[0]
                    )
        except Exception as e:
            logger.debug(f"Could not resolve eos_token_id: {e}")
    quant_info = dict(quantization_config)
    for key, val in per_layer_config.items():
        quant_info[key] = val
    output_config["quantization"] = quant_info
    output_config["quantization_config"] = quant_info
    with open(output / "config.json", "w") as f:
        json.dump(output_config, f, indent=2, ensure_ascii=False)

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

    cb("saving", 100.0)
    logger.info(
        f"oQ{oq_level:g} streaming: completed -> {output_path} ({total_shards} shards)"
    )


_SENS_NUM_SAMPLES = 128
_SENS_SEQ_LENGTH = 256


CALIB_DATASETS = {
    "default": "Built-in (General)",
    "wikitext": "WikiText-2",
    "c4": "C4 (Web Crawl)",
    "code": "Code (StarCoder)",
    "multilingual": "Multilingual (CulturaX)",
    "code_multilingual": "Code + Multilingual + Reasoning",
}


def _load_calibration_data(
    tokenizer,
    dataset: str = "code_multilingual",
    num_samples: int = _SENS_NUM_SAMPLES,
    seq_length: int = _SENS_SEQ_LENGTH,
):
    """Load calibration data for sensitivity measurement.

    Uses built-in calibration data by default (no download needed).
    Built-in data includes English, code, Korean, Chinese, Japanese.

    Args:
        tokenizer: Model tokenizer.
        dataset: "code_multilingual" (built-in default), "code", "multilingual",
                 "default" (mlx-lm generic), or HuggingFace dataset names.
        num_samples: Number of calibration samples.
        seq_length: Sequence length per sample.

    Returns:
        MLX array of shape (num_samples, seq_length) or None on failure.
    """
    if dataset in ("code_multilingual", "code", "multilingual"):
        try:
            return _load_builtin_calibration(
                tokenizer, dataset, num_samples, seq_length
            )
        except Exception as e:
            logger.warning(
                f"Built-in calibration failed: {e}, falling back to mlx-lm default"
            )

    if dataset == "default":
        try:
            from mlx_lm.quant.utils import load_data

            return load_data(
                tokenizer, num_samples=num_samples, sequence_length=seq_length
            )
        except ImportError:
            logger.warning("mlx_lm.quant.utils.load_data not available")
            return None

    try:
        return _load_hf_calibration(tokenizer, dataset, num_samples, seq_length)
    except Exception as e:
        logger.warning(f"Failed to load {dataset}: {e}, falling back to built-in")

    try:
        return _load_builtin_calibration(
            tokenizer, "code_multilingual", num_samples, seq_length
        )
    except Exception:
        return None


def _load_builtin_calibration(
    tokenizer, dataset: str, num_samples: int, seq_length: int
):
    """Load from built-in oq_calibration_data.json (shipped with package)."""
    import mlx.core as mx

    data_path = Path(__file__).parent / "oq_calibration_data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Built-in calibration data not found: {data_path}")

    with open(data_path, encoding="utf-8") as f:
        all_data = json.load(f)

    if dataset == "code_multilingual":
        texts = []
        for key in ("code", "en", "ko", "zh", "ja", "tool_calling", "reasoning"):
            texts.extend(all_data.get(key, []))
    elif dataset == "code":
        texts = all_data.get("code", []) + all_data.get("en", [])
    elif dataset == "multilingual":
        texts = []
        for key in ("en", "ko", "zh", "ja"):
            texts.extend(all_data.get(key, []))
    else:
        texts = []
        for v in all_data.values():
            texts.extend(v)

    if not texts:
        raise ValueError("No calibration text available")

    total_kb = sum(len(t) for t in texts) // 1024
    logger.info(f"Built-in calibration: {len(texts)} texts, {total_kb} KB ({dataset})")

    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if isinstance(ids, list):
            all_ids.extend(ids)
        else:
            all_ids.extend(ids.tolist() if hasattr(ids, "tolist") else list(ids))
    tokens = mx.array(all_ids)

    usable = (tokens.size // seq_length) * seq_length
    if usable == 0:
        raise ValueError(f"Not enough tokens ({tokens.size} < {seq_length})")
    tokens = tokens[:usable].reshape(-1, seq_length)

    if num_samples > 0 and tokens.shape[0] > num_samples:
        indices = mx.random.permutation(tokens.shape[0])[:num_samples]
        tokens = tokens[indices]

    logger.info(f"Calibration: {tokens.shape[0]} samples x {seq_length} tokens")
    return tokens


def _load_hf_calibration(tokenizer, dataset: str, num_samples: int, seq_length: int):
    """Load calibration data from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required for non-default calibration. "
            "Install with: pip install datasets"
        )

    logger.info(f"Loading calibration dataset: {dataset}")

    if dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = "\n".join(t for t in ds["text"] if t.strip())
    elif dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = "\n".join(
            item["text"] for i, item in enumerate(ds) if i < num_samples * 2
        )
    elif dataset == "code":
        ds = load_dataset(
            "bigcode/starcoderdata", "python", split="train", streaming=True
        )
        texts = "\n".join(
            item["content"] for i, item in enumerate(ds) if i < num_samples * 2
        )
    elif dataset == "multilingual":
        langs = ["en", "ko", "zh", "ja", "de", "fr", "es"]
        per_lang = max(1, num_samples // len(langs))
        all_texts = []
        for lang in langs:
            try:
                ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
                lang_texts = [
                    item["text"] for i, item in enumerate(ds) if i < per_lang * 2
                ]
                all_texts.extend(lang_texts)
            except Exception:
                logger.warning(f"Failed to load CulturaX/{lang}, skipping")
        texts = "\n".join(all_texts)
    elif dataset == "code_multilingual":
        half = max(1, num_samples // 2)
        code_texts = []
        try:
            ds = load_dataset(
                "bigcode/starcoderdata", "python", split="train", streaming=True
            )
            code_texts = [item["content"] for i, item in enumerate(ds) if i < half * 2]
        except Exception:
            logger.warning("Failed to load code dataset")

        ml_texts = []
        for lang in ["en", "ko", "zh", "ja"]:
            try:
                ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
                ml_texts.extend(
                    item["text"] for i, item in enumerate(ds) if i < half // 2
                )
            except Exception:
                pass
        texts = "\n".join(code_texts + ml_texts)
    else:
        raise ValueError(f"Unknown calibration dataset: {dataset}")

    if not texts:
        raise ValueError(f"No text loaded from {dataset}")

    tokens = tokenizer.encode(texts)
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    if isinstance(tokens, list):
        tokens = mx.array(tokens)
    elif not isinstance(tokens, mx.array):
        import numpy as np

        tokens = mx.array(np.array(tokens))

    if tokens.ndim > 1:
        tokens = tokens.reshape(-1)

    n_tokens = tokens.size
    usable = (n_tokens // seq_length) * seq_length
    if usable == 0:
        raise ValueError(f"Not enough tokens from {dataset} (got {n_tokens})")
    tokens = tokens[:usable].reshape(-1, seq_length)

    n_available = tokens.shape[0]
    if num_samples > 0 and n_available > num_samples:
        indices = mx.random.permutation(n_available)[:num_samples]
        tokens = tokens[indices]

    logger.info(
        f"Calibration: {tokens.shape[0]} samples × {seq_length} tokens from {dataset}"
    )
    return tokens


def _find_model_layers(model):
    """Find embedding function and transformer layers in the model.

    Searches common model structures: standard, VLM, and direct.
    Returns (embed_fn, layers) or (None, None).
    """
    embed_fn = None
    layers = None

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_fn = model.model.embed_tokens
        layers = model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        lm = model.language_model.model
        if hasattr(lm, "embed_tokens"):
            embed_fn = lm.embed_tokens
            layers = lm.layers
    elif hasattr(model, "embed_tokens"):
        embed_fn = model.embed_tokens
        layers = model.layers
    elif hasattr(model, "backbone") and hasattr(model.backbone, "embeddings"):
        embed_fn = model.backbone.embeddings
        layers = model.layers

    return embed_fn, layers


def _forward_layer_result(block, inputs, mask, position_ids):
    """Forward pass through a transformer layer, returning output and aux."""
    if isinstance(position_ids, dict) and position_ids.get("kind") == "glm_moe_dsa":
        try:
            result = block(
                inputs,
                mask,
                None,
                position_ids.get("prev_topk_indices"),
            )
            if isinstance(result, tuple):
                return result[0], result[1] if len(result) > 1 else None
            return result, None
        except (TypeError, ValueError, RuntimeError, AttributeError) as e:
            logger.debug(
                f"_forward_layer: GLM MoE DSA signature failed for "
                f"{type(block).__name__}: {e}"
            )
            return None, None

    last_exc = None
    for call_args in [
        (inputs, mask, None, position_ids),
        (inputs, mask, None),
        (inputs, mask),
        (inputs, None, mask, None),
        (inputs,),
    ]:
        try:
            result = block(*call_args)
            if isinstance(result, tuple):
                return result[0], result[1] if len(result) > 1 else None
            return result, None
        except (TypeError, ValueError, RuntimeError, AttributeError) as e:
            last_exc = e
            continue
    if last_exc is not None:
        logger.debug(
            f"_forward_layer: all signatures failed for "
            f"{type(block).__name__}: {last_exc}"
        )
    return None, None


def _forward_layer(block, inputs, mask, position_ids):
    """Forward pass through a transformer layer with flexible signature."""
    return _forward_layer_result(block, inputs, mask, position_ids)[0]


def _layer_masks_for_model(model, layers, inputs):
    """Build the per-layer mask schedule used by the original model."""
    if hasattr(model, "make_cache") and any(
        hasattr(layer, "is_linear") for layer in layers
    ):
        try:
            from mlx_lm.models.base import create_attention_mask, create_ssm_mask

            cache = model.make_cache()
            fa_idx = getattr(getattr(model, "model", model), "fa_idx", 0)
            ssm_idx = getattr(getattr(model, "model", model), "ssm_idx", 0)
            fa_cache = cache[fa_idx] if fa_idx < len(cache) else None
            ssm_cache = cache[ssm_idx] if ssm_idx < len(cache) else None
            try:
                fa_mask = create_attention_mask(inputs, fa_cache)
            except TypeError:
                # mlx-lm API changed — cache.make_mask signature differs
                fa_mask = None
            try:
                ssm_mask = create_ssm_mask(inputs, ssm_cache)
            except TypeError:
                ssm_mask = None
            if fa_mask is not None or ssm_mask is not None:
                if fa_mask is None:
                    fa_mask = nn.MultiHeadAttention.create_additive_causal_mask(
                        inputs.shape[1]
                    ).astype(inputs.dtype if hasattr(inputs, "dtype") else mx.float16)
                # SSM layers (GatedDeltaNet) expect (B, S) boolean mask, not
                # (S, S) causal mask.  During calibration there is no padding,
                # so None is the correct mask for SSM layers.
                return [
                    ssm_mask if getattr(layer, "is_linear", False) else fa_mask
                    for layer in layers
                ]
        except (ImportError, AttributeError):
            pass

    seq_len = inputs.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    dtype = inputs.dtype if hasattr(inputs, "dtype") else mx.float16
    return [mask.astype(dtype)] * len(layers)


def _qdq_weight_only(weight, bits: int, group_size: int, mode: str):
    qw, scales, *rest = mx.quantize(weight, group_size=group_size, bits=bits, mode=mode)
    return mx.dequantize(
        qw,
        scales,
        rest[0] if rest else None,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


def _temporary_quantize_block(block, config, oq_level, group_size: int):
    """Quantize-dequantize a block using the active predicate configuration."""
    saved = {}
    for path, module in tree_flatten(block.leaf_modules(), is_leaf=nn.Module.is_module):
        if not hasattr(module, "weight") or not hasattr(module, "to_quantized"):
            continue
        if getattr(module.weight, "ndim", 0) < 2:
            continue
        norm_path = _normalize_quant_path(path)
        bits, gs, mode = _get_predicate_bits(norm_path, config, oq_level, group_size)
        if bits is None or module.weight.shape[-1] % gs != 0:
            continue
        saved[path] = module.weight
        module.weight = _qdq_weight_only(module.weight, bits, gs, mode)
    return saved


def _restore_saved_weights(block, saved):
    """Restore temporarily quantized block weights."""
    modules_by_path = dict(
        tree_flatten(block.leaf_modules(), is_leaf=nn.Module.is_module)
    )
    for path, weight in saved.items():
        if path in modules_by_path:
            modules_by_path[path].weight = weight


def _prepare_layer_inputs(model, layers, calib_data, inputs):
    """Model-specific (inputs, per-layer masks, 4th forward arg) for
    block-level sensitivity forwards.

    DeepSeek V4 blocks run on a 4D hidden (B, S, hc_mult, hidden), take a
    window-limited array mask, and need the real token ids as their 4th
    argument (hash expert routing indexes tid2eid with them). Everything
    else keeps the generic 3D inputs + causal masks + position ids.
    """
    model_type = str(
        getattr(model, "model_type", "")
        or getattr(getattr(model, "args", None), "model_type", "")
        or getattr(
            getattr(getattr(model, "model", None), "args", None),
            "model_type",
            "",
        )
    )
    if model_type.startswith("deepseek_v4"):
        args = model.args
        h = mx.broadcast_to(
            inputs[:, :, None, :],
            (inputs.shape[0], inputs.shape[1], args.hc_mult, inputs.shape[2]),
        )
        h = mx.contiguous(h)
        mask = create_attention_mask(
            h[:, :, 0, :],
            None,
            window_size=args.sliding_window,
            return_array=True,
        )
        return h, [mask] * len(layers), calib_data
    if model_type == "glm_moe_dsa":
        mask = create_attention_mask(inputs, None, return_array=True)
        state = {"kind": "glm_moe_dsa", "prev_topk_indices": None}
        return inputs, [mask] * len(layers), state
    masks = _layer_masks_for_model(model, layers, inputs)
    position_ids = mx.arange(calib_data.shape[1])[None, :]
    return inputs, masks, position_ids


def _measure_sensitivity_from_model(
    model,
    tokenizer,
    config,
    oq_level,
    calib_dataset="code_multilingual",
    num_samples=32,
    seq_length=256,
):
    """Measure per-layer quantization sensitivity on an already-loaded model.

    Does NOT modify weights — uses temporary quantize→dequantize per layer.
    Used by both streaming (after temporary load) and enhanced (before AWQ).

    Returns:
        Dict of {layer_idx: relative_mse_score}.
    """
    calib_data = _load_calibration_data(
        tokenizer,
        dataset=calib_dataset,
        num_samples=num_samples,
        seq_length=seq_length,
    )
    if calib_data is None:
        return {}

    embed_fn, layers = _find_model_layers(model)
    if embed_fn is None or layers is None:
        return {}

    inputs = embed_fn(calib_data)
    inputs, layer_masks, position_ids = _prepare_layer_inputs(
        model, layers, calib_data, inputs
    )
    sensitivity = {}

    for layer_idx, block in enumerate(layers):
        layer_mask = layer_masks[layer_idx] if layer_idx < len(layer_masks) else None
        prev_aux = (
            position_ids.get("prev_topk_indices")
            if isinstance(position_ids, dict)
            and position_ids.get("kind") == "glm_moe_dsa"
            else None
        )
        out_float, baseline_aux = _forward_layer_result(
            block, inputs, layer_mask, position_ids
        )
        if out_float is None:
            continue

        saved = _temporary_quantize_block(
            block, config, oq_level, _OQ_DEFAULT_GROUP_SIZE
        )
        if isinstance(position_ids, dict) and position_ids.get("kind") == "glm_moe_dsa":
            position_ids["prev_topk_indices"] = prev_aux
        out_quant, _ = _forward_layer_result(block, inputs, layer_mask, position_ids)
        if out_quant is not None:
            raw_mse = ((out_float - out_quant) ** 2).mean()
            out_magnitude = (out_float**2).mean()
            mse_val = raw_mse / mx.maximum(out_magnitude, 1e-10)
            mx.eval(mse_val)
            sensitivity[layer_idx] = mse_val.item()

        _restore_saved_weights(block, saved)

        if isinstance(position_ids, dict) and position_ids.get("kind") == "glm_moe_dsa":
            position_ids["prev_topk_indices"] = baseline_aux
        inputs = out_float
        mx.synchronize()
        mx.clear_cache()

    if sensitivity:
        ranked = sorted(sensitivity.items(), key=lambda x: -x[1])
        logger.info(
            f"oQ{oq_level:g}: layer sensitivity (descending): "
            + ", ".join(f"L{i}={s:.4f}" for i, s in ranked)
        )

    return sensitivity


def _measure_sensitivity(
    model_path: str,
    config: dict,
    oq_level,
    calib_dataset="code_multilingual",
    num_samples=32,
    seq_length=256,
    trust_remote_code: bool = False,
):
    """Measure sensitivity by loading model temporarily. Used by streaming path."""
    from fusion_mlx.utils.model_loading import (
        _checkpoint_has_mtp_weights,
        _has_mtp_heads,
        maybe_apply_pre_load_patches,
    )

    # Treat any model with a vision sub-config (vision_config / vit_config /
    # mm_vision_tower) as a VLM for the MTP attach decision. The classifier
    # in model_discovery._has_vision_subconfig owns the canonical predicate.
    is_vlm = _has_vision_subconfig(config)
    has_mtp_weights = _checkpoint_has_mtp_weights(model_path)

    # Reuse the centralised pre-load dispatch so every current and future
    # patch (MTP sanitize, DeepSeek V4, nested-visual, load_config, …) is
    # applied exactly as in the production load path.
    maybe_apply_pre_load_patches(model_path, for_vlm=is_vlm)

    # maybe_apply_pre_load_patches leaves mtp_active False, which is correct
    # for the text path: the patched qwen35_model.sanitize self-consistently
    # strips mtp.* when no head is attached. The VLM path needs both patches.
    # apply_mlx_vlm_mtp_patch fixes Model.sanitize so language_model.mtp.*
    # weights survive the load with the correct keys (stock mlx-vlm sanitize
    # strips them, which is what made the strict load fail with "Missing N
    # parameters" and the measurement silently return {}). The runtime patch
    # then attaches the MTP head so the checkpoint matches the model. Both
    # are idempotent. Sensitivity only reads backbone decoder layers, so this
    # is load-only.
    restore_mtp_active = None
    if is_vlm and _has_mtp_heads(config) and has_mtp_weights:
        try:
            from fusion_mlx.patches.mlx_lm_mtp import is_mtp_active, set_mtp_active
            from fusion_mlx.patches.mlx_vlm_mtp import (
                apply_mlx_vlm_mtp_patch,
                apply_mlx_vlm_mtp_runtime_patch,
            )

            apply_mlx_vlm_mtp_patch()
            apply_mlx_vlm_mtp_runtime_patch()
            prev_active = is_mtp_active()
            set_mtp_active(True)
            restore_mtp_active = lambda: set_mtp_active(prev_active)  # noqa: E731
        except Exception as e:
            logger.debug(f"mlx-vlm MTP runtime patch skipped for sensitivity: {e}")

    try:
        if is_vlm:
            import mlx.nn as _nn
            from mlx_vlm.utils import load_model as vlm_load_model

            # mlx_vlm.load_model calls model.load_weights(weights) without strict=False.
            # Shared-KV models (e.g. Gemma 4 2B/4B) omit k/v weights for shared layers,
            # so strict=True raises ValueError. Relax temporarily — sensitivity only needs
            # approximate weights; shared layers receive pre-computed KV at inference time.
            _orig_lw = _nn.Module.load_weights

            def _lenient_load_weights(self, file_or_weights, *args, **kwargs):
                kwargs.pop("strict", None)
                return _orig_lw(self, file_or_weights, *args, strict=False, **kwargs)

            _nn.Module.load_weights = _lenient_load_weights
            try:
                # No QAT config override needed here: mlx_vlm.utils.load_model
                # uses quantization_config.get("quant_method") rather than direct
                # key access, so a missing quant_method falls through silently.
                model = vlm_load_model(
                    Path(model_path),
                    lazy=True,
                    trust_remote_code=trust_remote_code,
                )
            finally:
                _nn.Module.load_weights = _orig_lw
            from mlx_lm.tokenizer_utils import load as load_tokenizer

            tokenizer = load_tokenizer(Path(model_path))
        else:
            from mlx_lm import load as lm_load

            model, tokenizer = lm_load(
                model_path,
                lazy=True,
                trust_remote_code=trust_remote_code,
                model_config=_sensitivity_lm_config_override(config),
            )
    except Exception as e:
        logger.error(f"Sensitivity measurement: model load failed ({e})")
        return {}
    finally:
        if restore_mtp_active is not None:
            restore_mtp_active()

    sensitivity = _measure_sensitivity_from_model(
        model,
        tokenizer,
        config,
        oq_level,
        calib_dataset,
        num_samples,
        seq_length,
    )

    del model, tokenizer
    mx.synchronize()
    mx.clear_cache()

    return sensitivity


_REQUANT_VALID_BITS = {2, 3, 4, 5, 6, 8}


def _perturb_bits_for(bits: int):
    """Closest valid re-quantization width below ``bits``, or None."""
    lower = [b for b in _REQUANT_VALID_BITS if b < bits]
    return max(lower) if lower else None


def _build_proxy_for_sensitivity(
    model_path: str,
    *,
    config: dict | None = None,
    dtype: str,
    working_dir: str | None = None,
    trust_remote_code: bool = False,
) -> Path:
    """Build a temporary uniform 4-bit proxy for sensitivity measurement.

    Used when the source model exceeds available RAM and full-fp16
    sensitivity measurement is not feasible. The proxy keeps oQ data-driven;
    without it, quantize_oq_streaming aborts the run with a RuntimeError.

    ``working_dir`` controls where the proxy is written. Defaults to the
    system temp dir when None, but callers should pass the parent of the
    output directory so the proxy lands on the same volume the user has
    already provisioned for the quantized output. This avoids the trap of
    Linux ``/tmp`` being tmpfs (RAM-backed), which would defeat the whole
    point of the OOM-driven proxy.

    The caller is responsible for deleting the returned directory.
    """
    # Reserve a unique temp name and let the streaming writer create it.
    proxy_dir = Path(tempfile.mkdtemp(prefix="fmlx_oq_proxy_", dir=working_dir))
    shutil.rmtree(proxy_dir)
    _build_streaming_proxy_for_sensitivity(
        model_path,
        proxy_dir,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    return proxy_dir


def _build_streaming_proxy_for_sensitivity(
    model_path: str,
    output_path: Path,
    *,
    dtype: str,
    trust_remote_code: bool = False,
) -> None:
    """Build a loadable 4-bit sensitivity proxy without loading the source.

    This is the RAM-safe counterpart to ``mlx_lm.convert(..., quantize=True)``.
    It uses the same header-only tensor index, streaming sanitize discovery,
    FP8 dequantization, and chunked quantization path as oQ itself, but skips
    sensitivity measurement and dynamic boost planning. The proxy is only used
    to rank layer sensitivity, so a compact uniform-ish 4-bit model is enough.
    """
    del trust_remote_code  # Kept for API symmetry; model code comes from config.

    source = Path(model_path)
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"Proxy output directory already exists: {output}")

    with open(source / "config.json") as f:
        config = json.load(f)
    _validate_oq_dtype_for_model(config, dtype)
    target_dtype = mx.bfloat16 if dtype == "bfloat16" else mx.float16

    weight_files = sorted(source.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"No .safetensors files found in {model_path}")

    all_weights = _LazyTensorIndex(weight_files)
    sanitize_fn = _build_model_sanitizer(config, text_only=False)
    cast_predicate = getattr(sanitize_fn, "_fmlx_cast_predicate", None)
    if sanitize_fn is not None:
        try:
            plan = _discover_sanitize_plan(sanitize_fn, all_weights)
            all_weights = _DiscoveredPlan(plan, all_weights)
            logger.info(
                "oQ proxy: discovered streaming sanitize plan, "
                f"{len(all_weights)} output tensors"
            )
        except Exception as e:
            raise RuntimeError(
                "oQ proxy: streaming sanitize-plan discovery failed "
                f"({e}). Extend _TrackedTensor for this sanitize pattern "
                "or provide sensitivity_model_path explicitly."
            ) from e

    config["_oq_non_quantizable"] = _build_non_quantizable_set(config)
    config["_oq_use_budget_plan"] = False
    config["_oq_boost_map"] = {}

    output.mkdir(parents=True, exist_ok=False)

    out_shard_data = {}
    out_shard_idx = 0
    weight_map = {}
    per_layer_config = {}
    tensor_names = list(all_weights.keys())
    base_bits = _PROXY_QUANT_BITS
    base_gs = _PROXY_QUANT_GROUP_SIZE
    base_mode = "affine"
    quantization_config = {
        "group_size": base_gs,
        "bits": base_bits,
        "mode": base_mode,
    }

    def _flush_shard() -> None:
        nonlocal out_shard_data, out_shard_idx
        if not out_shard_data:
            return
        shard_name = f"model-{out_shard_idx + 1:05d}-of-PLACEHOLDER.safetensors"
        shard_path = output / shard_name
        mx.save_safetensors(str(shard_path), out_shard_data, metadata={"format": "mlx"})
        for key in out_shard_data:
            weight_map[key] = shard_name
        out_shard_idx += 1
        out_shard_data = {}
        mx.synchronize()
        mx.clear_cache()

    for tensor_name in tensor_names:
        handled_packed = False
        if hasattr(all_weights, "pop_packed") and not _is_mtp_tensor(tensor_name):
            src_info = all_weights.source_quant_info(tensor_name)
            if src_info is not None and _should_quantize_tensor(
                tensor_name, all_weights.plan_shape(tensor_name)
            ):
                pred = universal_quant_predicate(
                    tensor_name, None, config, _PROXY_QUANT_BITS
                )
                if pred is not False and base_bits >= src_info["bits"]:
                    qw, scales = all_weights.pop_packed(tensor_name)
                    base = tensor_name[: -len(".weight")]
                    out_shard_data[f"{base}.weight"] = qw
                    out_shard_data[f"{base}.scales"] = scales
                    per_layer_config[base] = {
                        "bits": src_info["bits"],
                        "group_size": src_info["group_size"],
                        "mode": src_info["mode"],
                    }
                    del qw, scales
                    handled_packed = True

        if not handled_packed:
            w_mx = all_weights.pop(tensor_name)
            if isinstance(w_mx, _LazyTensor):
                w_mx = w_mx[:]
            shape = w_mx.shape

            if _is_mtp_tensor(tensor_name):
                del w_mx
                continue

            if _should_quantize_tensor(tensor_name, shape):
                pred = universal_quant_predicate(
                    tensor_name, None, config, _PROXY_QUANT_BITS
                )
                if pred is not False and len(shape) >= 2 and shape[-1] % base_gs == 0:
                    if (
                        mx.issubdtype(w_mx.dtype, mx.floating)
                        and w_mx.dtype != target_dtype
                    ):
                        w_mx = w_mx.astype(target_dtype)
                    qw, scales, biases = _quantize_chunked(
                        w_mx, base_gs, base_bits, base_mode
                    )
                    base = (
                        tensor_name[:-7]
                        if tensor_name.endswith(".weight")
                        else tensor_name
                    )
                    out_shard_data[f"{base}.weight"] = qw
                    out_shard_data[f"{base}.scales"] = scales
                    if biases is not None:
                        out_shard_data[f"{base}.biases"] = biases
                    del qw, scales, biases
                else:
                    if cast_predicate is None or cast_predicate(tensor_name):
                        w_mx = _cast_passthrough_tensor(tensor_name, w_mx, target_dtype)
                    out_shard_data[tensor_name] = w_mx
            else:
                if cast_predicate is None or cast_predicate(tensor_name):
                    w_mx = _cast_passthrough_tensor(tensor_name, w_mx, target_dtype)
                out_shard_data[tensor_name] = w_mx

            del w_mx

        if sum(v.nbytes for v in out_shard_data.values()) >= _MAX_SHARD_BYTES:
            _flush_shard()

    del all_weights
    mx.synchronize()
    mx.clear_cache()
    _flush_shard()

    total_shards = out_shard_idx
    if total_shards == 1:
        only = output / "model-00001-of-PLACEHOLDER.safetensors"
        final = output / "model.safetensors"
        only.rename(final)
        for key in list(weight_map):
            weight_map[key] = "model.safetensors"
    elif total_shards > 1:
        for i in range(total_shards):
            old_name = f"model-{i + 1:05d}-of-PLACEHOLDER.safetensors"
            new_name = f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
            old_path = output / old_name
            new_path = output / new_name
            if old_path.exists():
                old_path.rename(new_path)
                for key, value in list(weight_map.items()):
                    if value == old_name:
                        weight_map[key] = new_name

        total_size = sum(f.stat().st_size for f in output.glob("*.safetensors"))
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        with open(output / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    output_config = dict(config)
    for temp_key in (
        "_oq_sensitivity_map",
        "_oq_boost_map",
        "_oq_use_budget_plan",
        "_oq_non_quantizable",
    ):
        output_config.pop(temp_key, None)
    _normalize_mtp_in_config(output_config)
    quant_info = dict(quantization_config)
    for key, val in per_layer_config.items():
        quant_info[key] = val
    output_config["quantization"] = quant_info
    output_config["quantization_config"] = quant_info
    with open(output / "config.json", "w") as f:
        json.dump(output_config, f, indent=2, ensure_ascii=False)

    _copy_model_sidecars(source, output)


def _measure_sensitivity_from_quantized_model(
    model_path: str,
    config: dict,
    oq_level,
    calib_dataset="code_multilingual",
    num_samples=32,
    seq_length=256,
    trust_remote_code: bool = False,
):
    """Measure sensitivity via re-quantization on a quantized model.

    Loads a quantized model (~4x less memory than fp16) and perturbs each
    layer by re-quantizing one valid bit-width below the module's own
    bits. The relative MSE ranking matches fp16 qdq-MSE with ~90% top-10
    overlap.
    """
    from fusion_mlx.utils.model_loading import (
        _checkpoint_has_mtp_weights,
        _has_mtp_heads,
        maybe_apply_pre_load_patches,
    )

    # Reuse the centralised pre-load dispatch (DeepSeek V4 base patch,
    # load_model replacement for F8_E8M0 checkpoints, MTP sanitize, ...)
    # so the quantized source/proxy loads exactly as in production.
    # Idempotent; harmless for plain mlx-lm proxies.
    is_vlm = _has_vision_subconfig(config)
    has_mtp_weights = _checkpoint_has_mtp_weights(model_path)
    maybe_apply_pre_load_patches(model_path, for_vlm=is_vlm)

    restore_mtp_active = None
    try:
        if is_vlm:
            if _has_mtp_heads(config) and has_mtp_weights:
                try:
                    from fusion_mlx.patches.mlx_lm_mtp import (
                        is_mtp_active,
                        set_mtp_active,
                    )
                    from fusion_mlx.patches.mlx_vlm_mtp import (
                        apply_mlx_vlm_mtp_patch,
                        apply_mlx_vlm_mtp_runtime_patch,
                    )

                    apply_mlx_vlm_mtp_patch()
                    apply_mlx_vlm_mtp_runtime_patch()
                    prev_active = is_mtp_active()
                    set_mtp_active(True)
                    restore_mtp_active = lambda: set_mtp_active(
                        prev_active
                    )  # noqa: E731
                except Exception as e:
                    logger.debug(
                        "mlx-vlm MTP runtime patch skipped for proxy sensitivity: "
                        f"{e}"
                    )

            from mlx_lm.tokenizer_utils import load as load_tokenizer
            from mlx_vlm.utils import load_model as vlm_load_model

            model = vlm_load_model(
                Path(model_path),
                lazy=True,
                trust_remote_code=trust_remote_code,
            )
            tokenizer = load_tokenizer(Path(model_path))
        else:
            from mlx_lm import load as lm_load

            # Mirror the main quantize path's MTP patch sequence so an
            # MTP-bearing quantized proxy (e.g. a Qwen3.5 LLM oQ output with
            # preserve_mtp=True) loads cleanly. Without set_mtp_active(True) the
            # mlx-lm __init__ skips ``self.mtp`` and the load rejects the
            # ``mtp.*`` weights present in the proxy.
            try:
                from fusion_mlx.patches.mlx_lm_mtp import (
                    apply_mlx_lm_mtp_patch,
                    is_mtp_active,
                    set_mtp_active,
                )

                have_lm_patch = apply_mlx_lm_mtp_patch()
            except Exception:
                have_lm_patch = False
                is_mtp_active = None
                set_mtp_active = None

            if have_lm_patch:
                prev_active = is_mtp_active()
                set_mtp_active(True)
                restore_mtp_active = lambda: set_mtp_active(prev_active)  # noqa: E731

            model, tokenizer = lm_load(
                model_path,
                lazy=True,
                trust_remote_code=trust_remote_code,
            )
    except Exception as e:
        logger.error(f"Sensitivity proxy load failed ({e})")
        return {}
    finally:
        if restore_mtp_active is not None:
            restore_mtp_active()

    if config.get("model_type") == "glm_moe_dsa":
        capped_samples = min(num_samples, 16)
        capped_seq = min(seq_length, 128)
        if capped_samples != num_samples or capped_seq != seq_length:
            logger.info(
                "GLM MoE DSA proxy sensitivity: capping calibration to "
                f"{capped_samples} samples x {capped_seq} tokens"
            )
        num_samples = capped_samples
        seq_length = capped_seq

    calib_data = _load_calibration_data(
        tokenizer,
        dataset=calib_dataset,
        num_samples=num_samples,
        seq_length=seq_length,
    )
    if calib_data is None:
        del model, tokenizer
        mx.synchronize()
        mx.clear_cache()
        return {}

    embed_fn, layers = _find_model_layers(model)
    if embed_fn is None or layers is None:
        del model, tokenizer
        mx.synchronize()
        mx.clear_cache()
        return {}

    inputs = embed_fn(calib_data)
    inputs, layer_masks, position_ids = _prepare_layer_inputs(
        model, layers, calib_data, inputs
    )
    sensitivity = {}

    for layer_idx, block in enumerate(layers):
        layer_mask = layer_masks[layer_idx] if layer_idx < len(layer_masks) else None
        prev_aux = (
            position_ids.get("prev_topk_indices")
            if isinstance(position_ids, dict)
            and position_ids.get("kind") == "glm_moe_dsa"
            else None
        )
        out_baseline, baseline_aux = _forward_layer_result(
            block, inputs, layer_mask, position_ids
        )
        if out_baseline is None:
            continue
        # Materialize the baseline before mutating module weights below.
        # Without this, the lazy graph would resolve baseline against the
        # already-perturbed weights and the MSE would always be ~0.
        mx.eval(out_baseline)

        saved = {}
        for p, m in tree_flatten(block.leaf_modules(), is_leaf=nn.Module.is_module):
            if not hasattr(m, "scales") or not hasattr(m, "weight"):
                continue
            bits = getattr(m, "bits", 4)
            gs = getattr(m, "group_size", 64)
            mode = getattr(m, "mode", "affine")
            # Perturb at the closest valid bit-width below the module's own
            # bits (8→6, 4→3, ...). bits-1 alone silently skipped every
            # 8-bit module (7 is not a valid width), which made the whole
            # measurement a no-op on 8-bit-dominated checkpoints.
            perturb_bits = _perturb_bits_for(bits)
            if perturb_bits is None:
                continue
            w_float = mx.dequantize(
                m.weight,
                m.scales,
                getattr(m, "biases", None),
                group_size=gs,
                bits=bits,
                mode=mode,
            )
            saved[p] = (m.weight, m.scales, getattr(m, "biases", None), bits, mode)
            qw, sc, *rest = mx.quantize(
                w_float, group_size=gs, bits=perturb_bits, mode="affine"
            )
            m.weight = qw
            m.scales = sc
            m.biases = rest[0] if rest else None
            m.bits = perturb_bits
            m.mode = "affine"
            # Force re-quant materialization so the next forward sees the
            # perturbed weights instead of the lazy reference to the originals.
            if m.biases is not None:
                mx.eval(m.weight, m.scales, m.biases)
            else:
                mx.eval(m.weight, m.scales)

        if isinstance(position_ids, dict) and position_ids.get("kind") == "glm_moe_dsa":
            position_ids["prev_topk_indices"] = prev_aux
        out_perturbed, _ = _forward_layer_result(
            block, inputs, layer_mask, position_ids
        )

        modules_by_path = dict(
            tree_flatten(block.leaf_modules(), is_leaf=nn.Module.is_module)
        )
        for p, (w, s, b, orig_bits, orig_mode) in saved.items():
            if p in modules_by_path:
                mod = modules_by_path[p]
                mod.weight = w
                mod.scales = s
                if b is not None:
                    mod.biases = b
                elif hasattr(mod, "biases"):
                    del mod.biases
                mod.bits = orig_bits
                mod.mode = orig_mode

        if out_perturbed is not None:
            # Cast to float32 first: float16 squared differences overflow
            # easily on long sequences, producing NaN sensitivity scores.
            ob32 = out_baseline.astype(mx.float32)
            op32 = out_perturbed.astype(mx.float32)
            raw_mse = ((ob32 - op32) ** 2).mean()
            out_mag = (ob32**2).mean()
            mse_val = raw_mse / mx.maximum(out_mag, 1e-10)
            mx.eval(mse_val)
            sensitivity[layer_idx] = mse_val.item()

        if isinstance(position_ids, dict) and position_ids.get("kind") == "glm_moe_dsa":
            position_ids["prev_topk_indices"] = baseline_aux
        inputs = out_baseline
        mx.eval(inputs)
        mx.synchronize()
        mx.clear_cache()

    del model, tokenizer
    mx.synchronize()
    mx.clear_cache()

    if sensitivity:
        ranked = sorted(sensitivity.items(), key=lambda x: -x[1])
        logger.info(
            f"oQ{oq_level:g}: proxy sensitivity (descending): "
            + ", ".join(f"L{i}={s:.4f}" for i, s in ranked)
        )

    return sensitivity
