# SPDX-License-Identifier: Apache-2.0
"""Model loading utilities."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
                for k in f:
                    if k.startswith(_MTP_WEIGHT_PREFIXES):
                        return True
        except Exception as e:
            logger.debug("Failed to read %s header for mtp weight scan: %s", shard, e)
    return False


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


def maybe_load_custom_quantization(model_name, *, is_vlm=False):
    return None


def expand_per_layer_quant_keys(config):
    pass


def get_tokenizer_config(model_name, *, trust_remote_code=False):
    return None
