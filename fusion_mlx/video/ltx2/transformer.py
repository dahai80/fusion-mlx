# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 AV transformer block (vendored from mlx-video).
# Phase 4 LTX-2 direct-MLX port: model-layer foundation.
import logging
from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention
from .config import LTXRopeType, TransformerConfig
from .feed_forward import FeedForward
from .utils import rms_norm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Modality:
    latent: mx.array
    timesteps: mx.array
    positions: mx.array
    context: mx.array
    enabled: bool = True
    context_mask: mx.array | None = None
    positional_embeddings: tuple[mx.array, mx.array] | None = None
    sigma: mx.array | None = None


@dataclass(frozen=True)
class TransformerArgs:
    x: mx.array
    context: mx.array
    context_mask: mx.array | None
    timesteps: mx.array
    embedded_timestep: mx.array
    positional_embeddings: tuple[mx.array, mx.array]
    cross_positional_embeddings: tuple[mx.array, mx.array] | None
    cross_scale_shift_timestep: mx.array | None
    cross_gate_timestep: mx.array | None
    enabled: bool
    prompt_timesteps: mx.array | None = None
    prompt_embedded_timestep: mx.array | None = None


class BasicAVTransformerBlock(nn.Module):

    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        has_prompt_adaln: bool = False,
    ):
        super().__init__()

        self.idx = idx
        self.norm_eps = norm_eps
        self.has_prompt_adaln = has_prompt_adaln

        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            num_ada_params = 9 if has_prompt_adaln else 6
            self.scale_shift_table = mx.zeros((num_ada_params, video.dim))

            if has_prompt_adaln:
                self.prompt_scale_shift_table = mx.zeros((2, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            num_audio_ada_params = 9 if has_prompt_adaln else 6
            self.audio_scale_shift_table = mx.zeros((num_audio_ada_params, audio.dim))

            if has_prompt_adaln:
                self.audio_prompt_scale_shift_table = mx.zeros((2, audio.dim))

        if audio is not None and video is not None:
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            logger.debug(
                "video_to_audio_attn: query_dim=%d context_dim=%d heads=%d dim_head=%d inner_dim=%d",
                audio.dim,
                video.dim,
                audio.heads,
                audio.d_head,
                audio.d_head * audio.heads,
            )
            self.scale_shift_table_a2v_ca_audio = mx.zeros((5, audio.dim))
            self.scale_shift_table_a2v_ca_video = mx.zeros((5, video.dim))

    def get_ada_values(
        self,
        scale_shift_table: mx.array,
        batch_size: int,
        timestep: mx.array,
        indices: slice,
    ) -> tuple[mx.array, ...]:
        num_ada_params = scale_shift_table.shape[0]

        table_slice = scale_shift_table[indices]
        table_expanded = mx.expand_dims(mx.expand_dims(table_slice, axis=0), axis=0)

        timestep_reshaped = mx.reshape(
            timestep, (batch_size, timestep.shape[1], num_ada_params, -1)
        )

        timestep_slice = timestep_reshaped[:, :, indices, :]

        ada_values = table_expanded + timestep_slice

        num_sliced = ada_values.shape[2]
        result = tuple(ada_values[:, :, i, :] for i in range(num_sliced))

        return result

    def get_av_ca_ada_values(
        self,
        scale_shift_table: mx.array,
        batch_size: int,
        scale_shift_timestep: mx.array,
        gate_timestep: mx.array,
        num_scale_shift_values: int = 4,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        scale_shift_ada = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            scale_shift_timestep,
            slice(None, None),
        )

        gate_ada = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            batch_size,
            gate_timestep,
            slice(None, None),
        )

        scale_shift_squeezed = tuple(
            mx.squeeze(t, axis=1) if t.shape[1] == 1 else t for t in scale_shift_ada
        )
        gate_squeezed = tuple(
            mx.squeeze(t, axis=1) if t.shape[1] == 1 else t for t in gate_ada
        )

        return (*scale_shift_squeezed, *gate_squeezed)

    def __call__(
        self,
        video: TransformerArgs | None = None,
        audio: TransformerArgs | None = None,
        skip_video_self_attn: bool = False,
        skip_audio_self_attn: bool = False,
        skip_cross_modal: bool = False,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        batch_size = video.x.shape[0] if video is not None else audio.x.shape[0]

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.size > 0
        run_ax = audio is not None and audio.enabled and ax.size > 0
        run_a2v = (
            run_vx
            and (audio is not None and audio.enabled and ax.size > 0)
            and not skip_cross_modal
        )
        run_v2a = (
            run_ax
            and (video is not None and video.enabled and vx.size > 0)
            and not skip_cross_modal
        )

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )

            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            vx = (
                vx
                + self.attn1(
                    norm_vx,
                    pe=video.positional_embeddings,
                    skip_attention=skip_video_self_attn,
                )
                * vgate_msa
            )

            if self.has_prompt_adaln:
                vshift_q, vscale_q, vgate_q = self.get_ada_values(
                    self.scale_shift_table, vx.shape[0], video.timesteps, slice(6, 9)
                )
                vprompt_shift_kv, vprompt_scale_kv = self.get_ada_values(
                    self.prompt_scale_shift_table,
                    vx.shape[0],
                    video.prompt_timesteps,
                    slice(0, 2),
                )
                attn_input = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_q) + vshift_q
                encoder_hidden_states = (
                    video.context * (1 + vprompt_scale_kv) + vprompt_shift_kv
                )
                vx = (
                    vx
                    + self.attn2(
                        attn_input,
                        context=encoder_hidden_states,
                        mask=video.context_mask,
                    )
                    * vgate_q
                )
            else:
                vx = vx + self.attn2(
                    rms_norm(vx, eps=self.norm_eps),
                    context=video.context,
                    mask=video.context_mask,
                )

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            ax = (
                ax
                + self.audio_attn1(
                    norm_ax,
                    pe=audio.positional_embeddings,
                    skip_attention=skip_audio_self_attn,
                )
                * agate_msa
            )

            if self.has_prompt_adaln:
                ashift_q, ascale_q, agate_q = self.get_ada_values(
                    self.audio_scale_shift_table,
                    ax.shape[0],
                    audio.timesteps,
                    slice(6, 9),
                )
                aprompt_shift_kv, aprompt_scale_kv = self.get_ada_values(
                    self.audio_prompt_scale_shift_table,
                    ax.shape[0],
                    audio.prompt_timesteps,
                    slice(0, 2),
                )
                attn_input_a = (
                    rms_norm(ax, eps=self.norm_eps) * (1 + ascale_q) + ashift_q
                )
                encoder_hidden_states_a = (
                    audio.context * (1 + aprompt_scale_kv) + aprompt_shift_kv
                )
                ax = (
                    ax
                    + self.audio_attn2(
                        attn_input_a,
                        context=encoder_hidden_states_a,
                        mask=audio.context_mask,
                    )
                    * agate_q
                )
            else:
                ax = ax + self.audio_attn2(
                    rms_norm(ax, eps=self.norm_eps),
                    context=audio.context,
                    mask=audio.context_mask,
                )

        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            (
                scale_ca_audio_a2v,
                shift_ca_audio_a2v,
                scale_ca_audio_v2a,
                shift_ca_audio_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            (
                scale_ca_video_a2v,
                shift_ca_video_a2v,
                scale_ca_video_v2a,
                shift_ca_video_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                )

            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                ax = ax + (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                )

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        video_out = replace(video, x=vx) if video is not None else None
        audio_out = replace(audio, x=ax) if audio is not None else None

        return video_out, audio_out
