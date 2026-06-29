# SPDX-License-Identifier: Apache-2.0
"""Model loading helpers with post-load transforms."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)

_VLM_TEXT_PREFIX = "language_model."

_MLX_LM_LOAD_CONFIG_PATCHED = False


def expand_per_layer_quant_keys(cfg: dict) -> dict:
    for config_key in ("quantization", "quantization_config"):
        quant = cfg.get(config_key)
        if not isinstance(quant, dict):
            continue
        extras: dict[str, dict] = {}
        for key, val in quant.items():
            if not isinstance(val, dict):
                continue
            prefixed = _VLM_TEXT_PREFIX + key
            if not key.startswith(_VLM_TEXT_PREFIX) and prefixed not in quant:
                extras[prefixed] = val
            elif key.startswith(_VLM_TEXT_PREFIX):
                short = key[len(_VLM_TEXT_PREFIX) :]
                if short not in quant:
                    extras[short] = val
        if extras:
            quant.update(extras)
    return cfg


def _patch_mlx_lm_load_config() -> None:
    global _MLX_LM_LOAD_CONFIG_PATCHED
    if _MLX_LM_LOAD_CONFIG_PATCHED:
        return

    try:
        import mlx_lm.utils as _lu
    except ImportError:
        return

    _original = _lu.load_config

    def _patched(model_path, *args, **kwargs):
        cfg = _original(model_path, *args, **kwargs)
        expand_per_layer_quant_keys(cfg)
        return cfg

    _lu.load_config = _patched
    _MLX_LM_LOAD_CONFIG_PATCHED = True


def _has_mtp_heads(config: dict) -> bool:
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
                for k in f.keys():
                    if k.startswith(_MTP_WEIGHT_PREFIXES):
                        return True
        except Exception as e:
            logger.debug("Failed to read %s header for mtp weight scan: %s", shard, e)
    return False


def _is_mtp_compatible(config: dict, model_type: str | None) -> bool:
    if not _has_mtp_heads(config):
        return False
    if not model_type:
        return False
    return (
        model_type.startswith("qwen3_5")
        or model_type.startswith("qwen3_6")
        or model_type.startswith("deepseek_v4")
    )


def maybe_apply_pre_load_patches(
    model_name: str,
    model_settings: Any | None = None,
    for_vlm: bool = False,
) -> None:
    try:
        from ..patches.mlx_lm_mtp import set_mtp_active
        set_mtp_active(False)
    except ImportError:
        pass

    _patch_mlx_lm_load_config()

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
        try:
            from ..patches.deepseek_v4 import apply_deepseek_v4_patch
            if apply_deepseek_v4_patch():
                logger.info("DeepSeek V4 pre-load patch applied for %s", model_name)
        except ImportError:
            pass

    if model_type == "step3p7":
        try:
            from ..patches.step3p7 import apply_step3p7_patch
            if apply_step3p7_patch():
                logger.info("Step 3.7 pre-load patch applied for %s", model_name)
        except ImportError:
            pass

    text_config = config.get("text_config")
    text_model_type = (
        text_config.get("model_type") if isinstance(text_config, dict) else None
    )
    if model_type == "llama4" or text_model_type == "llama4":
        try:
            from ..patches.llama4_attention import apply_llama4_attention_patch
            if apply_llama4_attention_patch():
                logger.info("Llama 4 attention patch applied for %s", model_name)
        except ImportError:
            pass

    if model_type == "glm_moe_dsa":
        try:
            from ..patches.glm_moe_dsa import apply_glm_moe_dsa_patch
            if apply_glm_moe_dsa_patch():
                logger.info("GLM MoE DSA pre-load patch applied for %s", model_name)
        except ImportError:
            pass

    if for_vlm and model_type == "diffusion_gemma":
        try:
            from ..patches.mlx_vlm_diffusion import apply_mlx_vlm_diffusion_patch
            if apply_mlx_vlm_diffusion_patch():
                logger.info("mlx-vlm diffusion patch applied for %s", model_name)
        except ImportError:
            pass

    minimax_m3_types = {"minimax_m3", "minimax_m3_vl"}
    if for_vlm and (
        model_type in minimax_m3_types or text_model_type in minimax_m3_types
    ):
        try:
            from ..patches.mlx_vlm_minimax_m3_compat import (
                apply_mlx_vlm_minimax_m3_compat_patch,
            )
            if apply_mlx_vlm_minimax_m3_compat_patch():
                logger.info(
                    "MiniMax M3 mlx-vlm compatibility patch applied for %s",
                    model_name,
                )
        except ImportError:
            pass

        try:
            from ..patches.minimax_m3_sparse_attention import (
                apply_minimax_m3_sparse_attention_patch,
            )
            if apply_minimax_m3_sparse_attention_patch():
                logger.info(
                    "MiniMax M3 sparse attention patch applied for %s",
                    model_name,
                )
        except ImportError:
            pass

    if _is_mtp_compatible(config, model_type):
        mtp_enabled = bool(
            model_settings is not None and getattr(model_settings, "mtp_enabled", False)
        )
        try:
            from ..patches.mlx_lm_mtp import (
                apply_mlx_lm_mtp_patch,
                set_mtp_active,
            )
        except ImportError:
            pass
        else:
            if apply_mlx_lm_mtp_patch():
                set_mtp_active(mtp_enabled)
                if mtp_enabled:
                    logger.info(
                        "Native MTP patch applied for %s (model_type=%s, active)",
                        model_name,
                        model_type,
                    )
                else:
                    logger.debug(
                        "Native MTP patch applied for %s for sanitize correctness "
                        "(model has MTP heads but mtp_enabled=False; head not attached)",
                        model_name,
                    )

        if for_vlm:
            try:
                from ..patches.mlx_vlm_mtp import (
                    apply_mlx_vlm_mtp_patch,
                    apply_mlx_vlm_mtp_runtime_patch,
                    set_mtp_attach_enabled,
                )
            except ImportError:
                pass
            else:
                has_mtp_weights = _checkpoint_has_mtp_weights(model_name)
                set_mtp_attach_enabled(has_mtp_weights)

                if apply_mlx_vlm_mtp_patch():
                    if mtp_enabled:
                        logger.info(
                            "mlx-vlm MTP sanitize patch applied for %s",
                            model_name,
                        )
                    else:
                        logger.debug(
                            "mlx-vlm MTP sanitize patch applied for %s "
                            "(mtp_enabled=False; allows persisted mtp.* "
                            "weights to bind)",
                            model_name,
                        )
                if apply_mlx_vlm_mtp_runtime_patch():
                    if not has_mtp_weights:
                        logger.info(
                            "mlx-vlm runtime MTP patch applied for %s "
                            "(config declares mtp heads but checkpoint "
                            "ships no mtp.* weights; MTPModule attachment "
                            "skipped to keep strict load_weights happy)",
                            model_name,
                        )
                    elif mtp_enabled:
                        logger.info(
                            "mlx-vlm runtime MTP patch applied for %s",
                            model_name,
                        )
                    else:
                        logger.debug(
                            "mlx-vlm runtime MTP patch applied for %s "
                            "(mtp_enabled=False; head attached for weight "
                            "load only)",
                            model_name,
                        )
    elif model_settings is not None and getattr(model_settings, "mtp_enabled", False):
        logger.warning(
            "mtp_enabled=True for %s but model is incompatible "
            "(model_type=%r, mtp_heads=%s); MTP path will be inactive",
            model_name,
            model_type,
            _has_mtp_heads(config),
        )

    if (
        for_vlm
        and model_type
        and model_type.startswith("qwen3_5_moe")
        and not _is_mtp_compatible(config, model_type)
    ):
        try:
            from ..patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_patch
        except ImportError:
            pass
        else:
            if apply_mlx_vlm_mtp_patch():
                logger.debug(
                    "mlx-vlm qwen3_6 MoE VLM sanitize patch applied for %s "
                    "(no MTP heads; switch_mlp load correctness)",
                    model_name,
                )

    if for_vlm and model_type and model_type.startswith("qwen3_5_moe"):
        try:
            from ..patches.qwen3_6_nested_visual import (
                apply_qwen3_6_nested_visual_patch,
            )
        except ImportError:
            pass
        else:
            if apply_qwen3_6_nested_visual_patch():
                logger.info(
                    "qwen3_6 nested-visual sanitize wrap applied for %s",
                    model_name,
                )


def load_text_model(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
    model_settings: Any | None = None,
):
    maybe_apply_pre_load_patches(model_name, model_settings=model_settings)
    from mlx_lm import load

    trust_remote_code = (
        bool(getattr(model_settings, "trust_remote_code", False))
        if model_settings is not None
        else False
    )
    return load(
        model_name,
        tokenizer_config=tokenizer_config,
        trust_remote_code=trust_remote_code,
    )


def materialize_lazy_state(state: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in state.items():
        if hasattr(value, "materialize"):
            value.materialize()
        result[key] = value
    return result


def materialize_model_arrays(model: Any) -> None:
    arrays = [v for _, v in tree_flatten(model) if isinstance(v, mx.array)]
    if arrays:
        mx.eval(arrays)


def apply_post_load_transforms(model: Any, model_settings: Any = None) -> Any:
    if model_settings is None:
        return model

    index_cache_freq = getattr(model_settings, "index_cache_freq", None)
    if index_cache_freq is not None and index_cache_freq >= 2:
        try:
            from ..patches.index_cache import apply_index_cache
        except ImportError:
            pass
        else:
            applied = apply_index_cache(model, index_cache_freq)
            if applied:
                logger.info(f"IndexCache applied: freq={index_cache_freq}")

    return model


def maybe_load_custom_quantization(
    model_name: str,
    *,
    is_vlm: bool,
) -> tuple[Any, Any] | None:
    config_path = Path(model_name) / "config.json"
    if not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        logger.debug(
            "Could not read %s for custom quantization dispatch: %s",
            config_path,
            e,
        )
        return None

    quant_config = config.get("quantization_config")
    quant_method = quant_config.get("quant_method") if quant_config else None

    if not quant_method:
        return None

    if quant_method.lower() == "paroquant":
        try:
            from paroquant.inference.backends.mlx.load import load as paro_load
        except ImportError as e:
            raise ImportError(
                "This model uses ParoQuant. Install it separately with: "
                'pip install "paroquant[mlx]"'
            ) from e

        model, processor, loaded_is_vlm = paro_load(model_name, force_text=not is_vlm)
        if is_vlm and not loaded_is_vlm:
            raise ValueError(
                "ParoQuant loader returned a text-only model for VLM load: "
                f"{model_name}"
            )
    else:
        return None

    return model, processor
