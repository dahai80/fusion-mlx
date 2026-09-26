# SPDX-License-Identifier: Apache-2.0
# ACE-Step MLX pipeline (issue #988): ConditionGenerationModel + flow-match
# sampling loop. Wires DiT + condition encoder + tokenizer/detokenizer.
# Ported from ACE-Step/Ace-Step1.5 modeling_acestep_v15_turbo.py (MIT).
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from .condition import ConditionEncoder
from .config import AceStepConfig
from .modeling import AceStepDiTModel
from .tokenizer import AceStepAudioTokenizer, AudioTokenDetokenizer

logger = logging.getLogger(__name__)


SHIFT_TIMESTEPS = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [1.0, 0.9333, 0.8571, 0.7692, 0.6667, 0.5455, 0.4, 0.2222],
    3.0: [1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3],
}


class AceStepModel(nn.Module):
    # Full ACE-Step condition generation model: decoder (DiT) + encoder
    # (text+lyric+timbre) + tokenizer (FSQ) + detokenizer. Flow-match sampling.
    def __init__(self, config: AceStepConfig):
        super().__init__()
        self.config = config
        self.decoder = AceStepDiTModel(config)
        self.encoder = ConditionEncoder(config)
        self.tokenizer = AceStepAudioTokenizer(config)
        self.detokenizer = AudioTokenDetokenizer(config)
        self.null_condition_emb = mx.random.normal((1, 1, config.hidden_size)) * 0.02

    def prepare_condition(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: mx.array | None,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: mx.array,
        refer_audio_packed: mx.array | None,
        refer_audio_order_mask: mx.array | None,
        src_latents: mx.array,
        chunk_masks: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        enc_h, enc_m = self.encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_packed=refer_audio_packed,
            refer_audio_order_mask=refer_audio_order_mask,
        )
        # tokenize src_latents -> 5Hz quantized -> detokenize to 25Hz lm_hints.
        # pad src_latents to multiple of pool_window_size for tokenize.
        P = self.config.pool_window_size
        T = src_latents.shape[1]
        pad_len = (P - (T % P)) % P
        if pad_len > 0:
            src_padded = mx.concatenate(
                [
                    src_latents,
                    mx.zeros(
                        (src_latents.shape[0], pad_len, src_latents.shape[-1]),
                        dtype=src_latents.dtype,
                    ),
                ],
                axis=1,
            )
        else:
            src_padded = src_latents
        q5, idx = self.tokenizer.tokenize(src_padded)
        lm_hints_25 = self.detokenizer(q5)
        lm_hints_25 = lm_hints_25[:, :T, :]
        # context = [src_latents, chunk_masks] concat on feature
        context_latents = mx.concatenate(
            [src_latents, chunk_masks.astype(src_latents.dtype)], axis=-1
        )
        return enc_h, enc_m, context_latents

    @staticmethod
    def _get_x0(zt: mx.array, vt: mx.array, t: mx.array) -> mx.array:
        t_ = t[:, None, None]
        return zt - vt * t_

    @staticmethod
    def _renoise(x: mx.array, t: mx.array, noise: mx.array | None = None) -> mx.array:
        if noise is None:
            noise = mx.random.normal(x.shape)
        t_ = t[:, None, None]
        return t_ * noise + (1.0 - t_) * x

    def generate(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: mx.array,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: mx.array,
        src_latents: mx.array,
        chunk_masks: mx.array,
        refer_audio_packed: mx.array | None = None,
        refer_audio_order_mask: mx.array | None = None,
        attention_mask: mx.array | None = None,
        seed: int | None = None,
        shift: float = 3.0,
        infer_method: str = "ode",
    ) -> mx.array:
        bsz = src_latents.shape[0]
        if attention_mask is None:
            attention_mask = mx.ones(
                (bsz, src_latents.shape[1]), dtype=src_latents.dtype
            )

        enc_h, enc_m, context_latents = self.prepare_condition(
            text_hidden_states,
            text_attention_mask,
            lyric_hidden_states,
            lyric_attention_mask,
            refer_audio_packed,
            refer_audio_order_mask,
            src_latents,
            chunk_masks,
        )

        if seed is not None:
            mx.random.seed(int(seed))
        # noise shape = src_latents shape (context feat is 2x src feat; src half = noise)
        noise = mx.random.normal(src_latents.shape, dtype=src_latents.dtype)

        shift = min([1.0, 2.0, 3.0], key=lambda x: abs(x - shift))
        t_schedule = SHIFT_TIMESTEPS[shift]
        num_steps = len(t_schedule)

        xt = noise
        for step_idx in range(num_steps):
            t_curr = mx.array([t_schedule[step_idx]] * bsz, dtype=src_latents.dtype)
            vt = self.decoder(
                hidden_states=xt,
                timestep=t_curr,
                timestep_r=t_curr,
                attention_mask=attention_mask,
                encoder_hidden_states=enc_h,
                encoder_attention_mask=enc_m,
                context_latents=context_latents,
            )
            if step_idx == num_steps - 1:
                xt = self._get_x0(xt, vt, t_curr)
                break
            if infer_method == "sde":
                pred_clean = self._get_x0(xt, vt, t_curr)
                t_next = mx.array(
                    [t_schedule[step_idx + 1]] * bsz, dtype=src_latents.dtype
                )
                xt = self._renoise(pred_clean, t_next)
            else:  # ode Euler
                dt = t_schedule[step_idx] - t_schedule[step_idx + 1]
                xt = xt - vt * dt
            logger.info(
                "acestep step %d/%d t=%.4f",
                step_idx + 1,
                num_steps,
                t_schedule[step_idx],
            )
        return xt


__all__ = ["AceStepModel"]
