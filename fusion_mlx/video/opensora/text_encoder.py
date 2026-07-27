# SPDX-License-Identifier: Apache-2.0
# Open-Sora V2 dual text encoder: pure-MLX UMT5 + CLIP-ViT-Large-patch14.
# Reuses skyreels_v3 encoders — zero torch dependency for inference.

import logging

import mlx.core as mx

from fusion_mlx.video.skyreels_v3.text_encoder import CLIPTextEncoder, UMT5Encoder

logger = logging.getLogger(__name__)


class DualTextEncoder:
    # Combines UMT5 (context) + CLIP (vector) encoders, pure MLX.

    def __init__(
        self,
        t5_path: str | None = None,
        clip_path: str | None = None,
        t5_max_length: int = 512,
        clip_max_length: int = 77,
    ):
        self.t5 = UMT5Encoder()
        self.clip = CLIPTextEncoder(clip_path)
        self.t5_max_length = t5_max_length
        self.clip_max_length = clip_max_length

    def encode(self, text: str | list[str]):
        if isinstance(text, str):
            text = [text]

        # UMT5: per-prompt encode_text -> pad to max_length -> stack
        context_parts = []
        for t in text:
            emb = self.t5.encode_text(t, max_length=self.t5_max_length)
            # emb: [1, L_valid, d_model], pad to [1, t5_max_length, d_model]
            L_valid = emb.shape[1]
            if L_valid < self.t5_max_length:
                pad_len = self.t5_max_length - L_valid
                pad = mx.zeros((1, pad_len, emb.shape[2]), dtype=emb.dtype)
                emb = mx.concatenate([emb, pad], axis=1)
            elif L_valid > self.t5_max_length:
                emb = emb[:, : self.t5_max_length]
            context_parts.append(emb[0])
        context = mx.stack(context_parts, axis=0)
        mx.eval(context)

        # CLIP: batch encode_text
        y_vec = self.clip.encode_text(text, max_length=self.clip_max_length)
        # Normalize to [B, dim]: if stub returns [B, L, dim], take mean over L
        if y_vec.ndim == 3:
            logger.warning(
                "CLIP stub mode: y_vec %s -> pooling to [B, dim]",
                y_vec.shape,
            )
            y_vec = y_vec.mean(axis=1)
        mx.eval(y_vec)

        logger.info(
            "DualTextEncoder: context=%s y_vec=%s", context.shape, y_vec.shape
        )
        return context, y_vec
