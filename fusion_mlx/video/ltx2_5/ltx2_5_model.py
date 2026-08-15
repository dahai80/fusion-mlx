# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 transformer (P3 reuse layer).
# LTX-2.5 reuses the LTX-2 transformer block / AdaLayerNorm / RoPE skeleton
# verbatim (same family, 48 layers vs 2/2.3's 13/19). LTXModel already accepts
# any LTXModelConfig subclass, so LTX2_5Model is a thin typed wrapper whose only
# NEW logic is from_pretrained: the 2.5 single-file checkpoint
# (diffusion_models/ltx-2.5-22b-{distilled,dev}-transformer-bf16.safetensors)
# ships WITHOUT a sibling config.json (config lives in the gated diffusers
# repo), so we synthesize the config from default_ltx2_5_config() instead of
# reading it from disk.
#
# UNVERIFIED against real 22B weights (gated, 403). The key-tree mapping in
# LTXModel.sanitize() is inherited unchanged from ltx2; per AR doc §4.2 MLX
# from_pretrained silently zero-inits on key mismatch (Cosmos/Wan2 已踩),
# so the unmatched/missing-key audit log below is the failure surface to watch
# on first real-weight load.
from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from ..ltx2.ltx2_model import LTXModel, X0Model
from .config import LTX2_5ModelConfig, LTX2_5Variant, default_ltx2_5_config

logger = logging.getLogger(__name__)


class LTX2_5Model(LTXModel):
    # LTX-2.5 22B transformer。复用 ltx2 LTXModel 全部前向逻辑，仅覆写加载入口。

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str | Path,
        config: LTX2_5ModelConfig | None = None,
        variant: LTX2_5Variant | str = LTX2_5Variant.DISTILLED,
        strict: bool = True,
    ) -> LTX2_5Model:
        # 2.5 单文件 checkpoint 无 config.json，配置由 default_ltx2_5_config 合成。
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"LTX-2.5 transformer weights not found: {weights_path}"
            )

        if config is None:
            config = default_ltx2_5_config(variant)
        logger.info(
            "ltx2_5 from_pretrained: weights=%s variant=%s layers=%d caption=%d",
            weights_path.name,
            LTX2_5Variant.from_str(variant).value,
            config.num_layers,
            config.caption_channels,
        )

        model = cls(config)

        # 单文件或目录均支持（目录则 glob *.safetensors）。
        if weights_path.is_file() and weights_path.suffix == ".safetensors":
            weights = mx.load(str(weights_path))
            weight_files = [weights_path]
        else:
            weight_files = sorted(weights_path.glob("*.safetensors"))
            if not weight_files:
                raise FileNotFoundError(
                    f"no .safetensors under {weights_path}"
                )
            weights = {}
            for wf in weight_files:
                weights.update(mx.load(str(wf)))
        logger.info(
            "ltx2_5 from_pretrained: loaded %d weight keys from %d files",
            len(weights),
            len(weight_files),
        )

        sanitized = model.sanitize(weights)
        # float32 -> bfloat16 对齐 ltx2 行为（bf16 checkpoint）。
        sanitized = {
            k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
            for k, v in sanitized.items()
        }

        # 键树审计：unmatched/missing 即静默零初始化风险点（AR §4.2）。
        try:
            model_params = dict(tree_flatten(model.parameters()))
            sanitized_keys = set(sanitized.keys())
            model_keys = set(model_params.keys())
            unmatched = sorted(sanitized_keys - model_keys)
            missing = sorted(model_keys - sanitized_keys)
            logger.info(
                "ltx2_5 from_pretrained: weights=%d model_params=%d "
                "unmatched=%d missing=%d",
                len(sanitized_keys),
                len(model_keys),
                len(unmatched),
                len(missing),
            )
            if unmatched:
                logger.warning(
                    "ltx2_5 unmatched weight keys (first 30): %s", unmatched[:30]
                )
            if missing:
                logger.warning(
                    "ltx2_5 missing model params (first 30): %s", missing[:30]
                )
            if strict and (unmatched or missing):
                # fail visible (Rule 12)：键树不匹配意味着权重未正确加载，
                # 真实模型首跑前必须定位修复。仅当 strict=False 时放行用于调试。
                raise RuntimeError(
                    f"LTX-2.5 weight key-tree mismatch: "
                    f"unmatched={len(unmatched)} missing={len(missing)} "
                    f"(see warning log above). Single-file checkpoint key tree "
                    f"differs from MLX module tree — needs convert/reshard step. "
                    f"Re-run with strict=False to inspect."
                )
        except RuntimeError:
            raise
        except Exception as audit_err:
            logger.warning("ltx2_5 weight audit skipped: %s", audit_err)

        model.load_weights(list(sanitized.items()), strict=False)
        mx.eval(model.parameters())
        model.eval()
        logger.info("ltx2_5 from_pretrained: load complete (strict=%s)", strict)
        return model


class LTX2_5X0Model(X0Model):
    # X0 预测器复用 ltx2 X0Model，velocity_model 改为 LTX2_5Model。
    pass
