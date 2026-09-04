# SPDX-License-Identifier: Apache-2.0
"""Model loading utilities."""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _w4a8_requested(model_settings: Any | None) -> bool:
    # Env FUSION_MLX_W4A8=1 forces W4A8 on even when settings are absent;
    # explicit settings field wins over env when present.
    if model_settings is not None and getattr(model_settings, "w4a8_enabled", False):
        return True
    return os.environ.get("FUSION_MLX_W4A8", "") == "1"


def _nvfp4_dequant_requested(model_settings: Any | None) -> bool:
    if model_settings is not None and getattr(
        model_settings, "nvfp4_dequant_enabled", False
    ):
        return True
    return os.environ.get("FUSION_MLX_NVFP4_DEQUANT", "") == "1"


def _fused_gdn_requested(model_settings: Any | None) -> bool:
    if model_settings is not None and getattr(
        model_settings, "fused_gdn_enabled", False
    ):
        return True
    return os.environ.get("FUSION_MLX_FUSED_GDN", "") == "1"


def _apply_nvfp4_dequant(model: Any) -> Any:
    # Post-load NVFP4 parameter-tree rewrite. Walks model.parameters() (a
    # nested {key: array|dict}) into a flat dotted dict, runs the NVFP4
    # reader, and if any tensors changed, writes them back via
    # model.update(). No-op (debug log) for non-NVFP4 checkpoints.
    try:
        from ..custom_kernels.nvfp4 import dequant_nvfp4_weights
    except Exception as e:
        logger.warning("NVFP4 dequant import failed: %s", e)
        return model
    params = model.parameters() if hasattr(model, "parameters") else None
    if params is None:
        logger.debug("NVFP4 dequant: model has no parameters(), skip")
        return model

    def _flatten(tree, prefix=""):
        out = {}
        for k, v in tree.items():
            key = f"{prefix}{k}"
            if hasattr(v, "items"):
                out.update(_flatten(v, f"{key}."))
            else:
                out[key] = v
        return out

    flat = _flatten(params)
    before_ids = {k: id(v) for k, v in flat.items()}
    flat = dequant_nvfp4_weights(flat)
    changed = [k for k, v in flat.items() if id(v) != before_ids.get(k)]
    if not changed:
        logger.debug(
            "NVFP4 dequant: no NVFP4 (uint8 + block-scale) tensors found "
            "(expected — non-NVFP4 checkpoint)"
        )
        return model
    logger.info("NVFP4 dequant: rewrote %d tensors", len(changed))

    def _inject(tree, prefix, flat):
        for k, v in tree.items():
            key = f"{prefix}{k}"
            if hasattr(v, "items"):
                _inject(v, f"{key}.", flat)
            elif key in flat:
                tree[k] = flat[key]

    _inject(params, "", flat)
    try:
        model.update(params)
    except Exception as e:
        logger.warning("NVFP4 dequant: model.update() failed: %s", e)
    return model


def materialize_lazy_state(state: dict[str, Any]) -> dict[str, Any]:
    """Force-evaluate any lazy tensors in the model state dict."""
    result = {}
    for key, value in state.items():
        if hasattr(value, "materialize"):
            value.materialize()
        result[key] = value
    return result


def _has_mtp_heads(config: dict) -> bool:
    """True iff the model config declares any MTP head layers."""
    if int(config.get("mtp_num_hidden_layers", 0) or 0) > 0:
        return True
    if int(config.get("num_nextn_predict_layers", 0) or 0) > 0:
        return True
    text_cfg = config.get("text_config") or {}
    if int(text_cfg.get("mtp_num_hidden_layers", 0) or 0) > 0:
        return True
    if int(text_cfg.get("num_nextn_predict_layers", 0) or 0) > 0:
        return True
    return False


_MTP_WEIGHT_PREFIXES = (
    "mtp.",
    "language_model.mtp.",
    "model.mtp.",
    "model.language_model.mtp.",
)


def _checkpoint_has_mtp_weights(model_path: str | Path) -> bool:
    """True iff the checkpoint at model_path ships any mtp.* weight tensor."""
    p = Path(model_path)
    if not p.is_dir():
        return False

    index_path = p / "model.safetensors.index.json"
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text())
            weight_map = data.get("weight_map") or {}
            return any(k.startswith(_MTP_WEIGHT_PREFIXES) for k in weight_map)
        except Exception as e:
            logger.debug("Failed to read %s for mtp weight scan: %s", index_path, e)

    shards = sorted(p.glob("*.safetensors"))
    if not shards:
        return False
    try:
        import safetensors
    except Exception as e:
        logger.debug("safetensors import failed for mtp weight scan: %s", e)
        return False

    for shard in shards:
        try:
            with safetensors.safe_open(str(shard), framework="numpy") as f:
                for k in f.keys():  # noqa: SIM118
                    if k.startswith(_MTP_WEIGHT_PREFIXES):
                        return True
        except Exception as e:
            logger.debug("Failed to read %s header for mtp weight scan: %s", shard, e)
    return False


def apply_post_load_transforms(model: Any, model_settings: Any | None = None) -> Any:
    # Centralized post-load transform entry point. Dispatches per-setting
    # transforms that mutate the loaded model in-place. Returns the model
    # unchanged (same object) so callers can treat this as a pass-through
    # pipeline. Settings that are None/absent are no-ops.
    #
    # IndexCache: index_cache_freq is parsed from admin UI / engine_pool
    # settings (model_settings.index_cache_freq) but, without this entry
    # point, apply_index_cache was never called from any load path — the
    # feature was half-wired (settings stored, transform dead). freq<2 is
    # a no-op (apply_index_cache itself rejects <2); None/absent no-op.
    if model_settings is None:
        return model
    # Fusion takeover: settings-driven lower-layer takeover (quant tagging,
    # paged kv). Runs before index_cache dispatch; independent of
    # index_cache_freq. Default OFF (fusion_takeover_enabled absent/False)
    # is a no-op passthrough.
    try:
        from ..fusion_takeover import apply_fusion_takeover

        model = apply_fusion_takeover(model, model_settings)
    except Exception as e:
        logger.warning("fusion takeover dispatch failed: %s", e)
    # Phase C #4: W4A8 activation-int8 linear conversion. Replaces nn.Linear
    # and nn.QuantizedLinear with W4A8Linear (int8 activation path). Runs
    # after fusion_takeover so takeover-tagged layers are skipped by the
    # W4A8 walker (they are not bare nn.Linear). Default OFF.
    if _w4a8_requested(model_settings):
        try:
            from ..custom_kernels.phase_c import convert_to_w4a8

            group_size = (
                getattr(model_settings, "w4a8_group_size", 64)
                if model_settings is not None
                else 64
            )
            model, n = convert_to_w4a8(model, group_size=group_size)
            logger.info("post-load W4A8 conversion: %d layers", n)
        except Exception as e:
            logger.warning("post-load W4A8 conversion failed: %s", e)
    # Phase C #4: fused GDN megakernel. No in-repo model consumes standalone
    # GDN today; the converter is a no-op scan unless a module declares
    # _is_gdn. Logged at debug so the "no GDN found" message is visible
    # only when an operator explicitly opts in.
    if _fused_gdn_requested(model_settings):
        try:
            from ..custom_kernels.phase_c import apply_fused_gdn

            model = apply_fused_gdn(model)
        except Exception as e:
            logger.warning("post-load fused GDN conversion failed: %s", e)
    # Phase C #4: NVFP4 load-time dequant. Flattens the model parameter tree
    # to a {dotted.key: array} dict and runs dequant_nvfp4_weights, which only
    # fires on uint8 weights with a sibling block-scale tensor (1 scale per 16
    # elements). A normal fp16/bf16/W4 LLM checkpoint has no such pairs → the
    # pass is a no-op logged at debug. This is a format-compatibility bridge
    # (4-bit storage win is NOT retained at inference); blocked on upstream
    # mlx#2962 for a native speed path. See custom_kernels/nvfp4.py.
    if _nvfp4_dequant_requested(model_settings):
        model = _apply_nvfp4_dequant(model)
    freq = getattr(model_settings, "index_cache_freq", None)
    if freq is None:
        return model
    try:
        freq_int = int(freq)
    except (TypeError, ValueError):
        logger.debug("index_cache_freq not int (%r), skipping transforms", freq)
        return model
    if freq_int < 2:
        logger.debug("index_cache_freq=%d < 2, skipping IndexCache", freq_int)
        return model
    try:
        from ..patches.index_cache import apply_index_cache

        applied = apply_index_cache(model, freq_int)
        if applied:
            logger.info("post-load IndexCache applied (freq=%d)", freq_int)
        else:
            logger.debug(
                "post-load IndexCache not applied (model unsupported, freq=%d)",
                freq_int,
            )
    except Exception as e:
        logger.warning("post-load IndexCache transform failed: %s", e)
    return model


def maybe_apply_pre_load_patches(
    model_name: str,
    model_settings: Any | None = None,
    for_vlm: bool = False,
) -> None:
    """Apply patches that need to run before mlx_lm.load() runs.

    Safe to call repeatedly; the patches are idempotent.
    """
    from ..patches.mlx_lm_mtp import set_mtp_active

    set_mtp_active(False)

    config_path = Path(model_name) / "config.json"
    if not config_path.exists():
        return
    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        logger.debug(
            "Could not read %s for pre-load patch dispatch: %s", config_path, e
        )
        return

    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.startswith("deepseek_v4"):
        from ..patches.deepseek_v4 import apply_deepseek_v4_patch

        if apply_deepseek_v4_patch():
            logger.info("DeepSeek V4 pre-load patch applied for %s", model_name)

    if model_type == "glm_moe_dsa":
        from ..patches.glm_moe_dsa import apply_glm_moe_dsa_patch

        if apply_glm_moe_dsa_patch():
            logger.info("GLM MoE DSA pre-load patch applied for %s", model_name)

    if _has_mtp_heads(config) and model_type:
        mtp_enabled = bool(
            model_settings is not None and getattr(model_settings, "mtp_enabled", False)
        )
        from ..patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            set_mtp_active,
        )

        if apply_mlx_lm_mtp_patch():
            set_mtp_active(mtp_enabled)

        if for_vlm:
            try:
                from ..patches.mlx_vlm_mtp import (
                    apply_mlx_vlm_mtp_patch,
                    apply_mlx_vlm_mtp_runtime_patch,
                )

                apply_mlx_vlm_mtp_patch()
                apply_mlx_vlm_mtp_runtime_patch()
            except Exception as e:
                logger.debug("mlx-vlm MTP patches skipped: %s", e)

    if for_vlm and model_type and model_type.startswith("qwen3_5_moe"):
        try:
            from ..patches.qwen3_6_nested_visual import (
                apply_qwen3_6_nested_visual_patch,
            )

            if apply_qwen3_6_nested_visual_patch():
                logger.info(
                    "qwen3_6 nested-visual sanitize wrap applied for %s",
                    model_name,
                )
        except Exception as e:
            logger.debug("qwen3_6 nested-visual patch import failed: %s", e)

    if for_vlm and model_type == "minimax_m3_vl":
        try:
            from ..patches.minimax_m3_sparse_attention import (
                apply_minimax_m3_sparse_attention_patch,
            )

            if apply_minimax_m3_sparse_attention_patch():
                logger.info(
                    "minimax_m3 sparse-attention left-padding patch applied for %s",
                    model_name,
                )
        except Exception as e:
            logger.debug("minimax_m3 sparse-attention patch import failed: %s", e)


def maybe_load_custom_quantization(model_name, *, is_vlm=False):
    return None


def expand_per_layer_quant_keys(config):
    pass


def get_tokenizer_config(model_name, *, trust_remote_code=False):
    return None
