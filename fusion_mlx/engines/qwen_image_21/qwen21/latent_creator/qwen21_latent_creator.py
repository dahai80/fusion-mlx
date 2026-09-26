# SPDX-License-Identifier: Apache-2.0
# Vendored from mflux 0.20.0 (https://github.com/filipstrand/mflux) PR#736
# MIT-licensed pure-MLX Qwen-Image-2.1 reference (txt2img + 4-ch RGBA VAE).
# Pinned to fusion-mlx's mlx stack. Upstream-tracked; do not modify forward
# numerics without parity-testing against mflux qwen21.

import mlx.core as mx

from mflux.models.common.config import ModelConfig


class Qwen21LatentCreator:
    # 2.1 consumes latents unpatched: packing is a plain spatial flatten of the (1, 64, H/16, W/16) latents.
    # All entry points cast to the model precision so the transformer stream stays bf16.

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        return mx.random.normal(
            shape=[1, (height // 16) * (width // 16), 64],
            key=mx.random.key(seed),
        ).astype(ModelConfig.precision)

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int, num_channels_latents: int = 64) -> mx.array:
        latents = mx.reshape(latents, (1, num_channels_latents, height // 16, width // 16))
        latents = mx.transpose(latents, (0, 2, 3, 1))
        return mx.reshape(latents, (1, (width // 16) * (height // 16), num_channels_latents)).astype(
            ModelConfig.precision
        )

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, height // 16, width // 16, 64))
        latents = mx.transpose(latents, (0, 3, 1, 2))
        return mx.reshape(latents, (1, 64, 1, height // 16, width // 16))
