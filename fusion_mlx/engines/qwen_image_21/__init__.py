# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 image backend (transparent RGBA PNG support).

Vendors mflux 0.20.0 PR#736's pure-MLX Qwen-Image-2.1 (7.1B DiT + Qwen3-VL-8B
text encoder + 4-channel RGBA VAE) behind fusion-mlx's ImageGenEngine. The
VAE decoder conv_out has out_channels=4, so ImageUtil._numpy_to_pil yields an
RGBA PIL image automatically — transparent PNG output is native, no post-fab
alpha synthesis. Parity with mlx-serve's `transparent:true` Qwen-Image-2.1 path.

mflux-fusion 0.18.0 (fusion-mlx's pinned mflux fork) predates PR#736, so the
qwen21 package is vendored here with self-imports rewritten to relative form;
common deps (weight loading, config, tokenizer, qwen3_vl) resolve against the
installed mflux-fusion, whose shared modules are byte-identical to 0.20.0.
"""

import logging

from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig

logger = logging.getLogger(__name__)

_QWEN21_MODEL_NAME = "Qwen/Qwen-Image-2.1"
_QWEN21_KEY = "qwen-image-2.1"


def _register_config() -> None:
    if _QWEN21_KEY not in AVAILABLE_MODELS:
        AVAILABLE_MODELS[_QWEN21_KEY] = ModelConfig(
            priority=29,
            aliases=["qwen-image-2.1", "qwen-2.1", "qwen-image-21"],
            model_name=_QWEN21_MODEL_NAME,
            base_model=None,
            controlnet_model=None,
            custom_transformer_model=None,
            num_train_steps=None,
            max_sequence_length=None,
            supports_guidance=True,
            requires_sigma_shift=True,
            sigma_max_shift=0.9,
            sigma_max_seq_len=8192,
            sigma_shift_terminal=0.02,
        )
    if not hasattr(ModelConfig, "qwen_image_21"):
        ModelConfig.qwen_image_21 = staticmethod(  # type: ignore[attr-defined]
            lambda: AVAILABLE_MODELS[_QWEN21_KEY]
        )


_register_config()

from .qwen21.variants.txt2img.qwen_image_21 import QwenImage21  # noqa: E402

__all__ = ["QwenImage21"]
