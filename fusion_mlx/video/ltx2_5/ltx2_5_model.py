# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 AV diffusion model (independent from LTX-2).
# 与 ltx2 的差异：has_prompt_adaln=True（prompt_adaln_single + to_gate_logits）、
# ff_bias/audio_ff_bias 不对称、keyframes_abs_pos_embedding、video/audio
# embeddings_connector（258 keys，sanitize 不再跳过）。键树已代码验证匹配 4349。
import logging
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .adaln import AdaLayerNormSingle
from .config import LTX2_5ModelConfig, LTX2_5Variant, LTXRopeType, default_ltx2_5_config
from .embeddings_connector import Embeddings1DConnector
from .rope import precompute_freqs_cis
from .text_projection import PixArtAlphaTextProjection
from .transformer import BasicAVTransformerBlock, Modality, TransformerArgs
from .utils import to_denoised

logger = logging.getLogger(__name__)


def _read_quant_config(model_dir: Path) -> tuple[int, int]:
    # 从 split_model.json 读 quantization 参数 (#762)。缺失时回退 q8 默认
    # (group_size=64, bits=8)，与 dgrauet/ltx-2.5-mlx-q8 一致。
    import json

    split_model = model_dir / "split_model.json"
    group_size = 64
    bits = 8
    if split_model.exists():
        try:
            with open(split_model) as f:
                cfg = json.load(f)
            if cfg.get("quantization_group_size"):
                group_size = int(cfg["quantization_group_size"])
            if cfg.get("quantization_bits"):
                bits = int(cfg["quantization_bits"])
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("ltx2_5 quant config read failed: %s", exc)
    return group_size, bits


class TransformerArgsPreprocessor:
    def __init__(
        self,
        patchify_proj: nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection | None,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        double_precision_rope: bool = False,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ):
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.prompt_adaln = prompt_adaln
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope

    def _prepare_timestep(
        self,
        timestep: mx.array,
        batch_size: int,
        hidden_dtype: mx.Dtype = None,
    ) -> tuple[mx.array, mx.array]:

        timestep = timestep * self.timestep_scale_multiplier
        timestep_emb, embedded_timestep = self.adaln(
            timestep.reshape(-1), hidden_dtype=hidden_dtype
        )

        timestep_emb = mx.reshape(
            timestep_emb, (batch_size, -1, timestep_emb.shape[-1])
        )
        embedded_timestep = mx.reshape(
            embedded_timestep, (batch_size, -1, embedded_timestep.shape[-1])
        )

        return timestep_emb, embedded_timestep

    def _prepare_timestep_with_adaln(
        self,
        adaln: AdaLayerNormSingle,
        timestep: mx.array,
        batch_size: int,
        hidden_dtype: mx.Dtype = None,
    ) -> tuple[mx.array, mx.array]:
        timestep = timestep * self.timestep_scale_multiplier
        timestep_emb, embedded_timestep = adaln(
            timestep.reshape(-1), hidden_dtype=hidden_dtype
        )
        timestep_emb = mx.reshape(
            timestep_emb, (batch_size, -1, timestep_emb.shape[-1])
        )
        embedded_timestep = mx.reshape(
            embedded_timestep, (batch_size, -1, embedded_timestep.shape[-1])
        )
        return timestep_emb, embedded_timestep

    def _prepare_context(
        self,
        context: mx.array,
        x: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        batch_size = x.shape[0]

        if self.caption_projection is not None:
            context = self.caption_projection(context)
        context = mx.reshape(context, (batch_size, -1, x.shape[-1]))
        return context, attention_mask

    def _prepare_attention_mask(
        self,
        attention_mask: mx.array | None,
        x_dtype: mx.Dtype,
    ) -> mx.array | None:
        if attention_mask is None:
            return None

        if attention_mask.dtype in [mx.float16, mx.float32, mx.bfloat16]:
            return attention_mask

        mask = (attention_mask.astype(x_dtype) - 1) * 1e9
        mask = mx.reshape(
            mask, (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        )
        return mask

    def _prepare_positional_embeddings(
        self,
        positions: mx.array,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
    ) -> tuple[mx.array, mx.array]:
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            double_precision=self.double_precision_rope,
        )
        return pe

    def prepare(self, modality: Modality) -> TransformerArgs:
        x = self.patchify_proj(modality.latent)
        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps, x.shape[0], hidden_dtype=x.dtype
        )
        context, attention_mask = self._prepare_context(
            modality.context, x, modality.context_mask
        )
        attention_mask = self._prepare_attention_mask(
            attention_mask, modality.latent.dtype
        )

        if modality.positional_embeddings is not None:
            pe = modality.positional_embeddings
        else:
            pe = self._prepare_positional_embeddings(
                positions=modality.positions,
                inner_dim=self.inner_dim,
                max_pos=self.max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                num_attention_heads=self.num_attention_heads,
            )

        prompt_timestep = None
        prompt_embedded_timestep = None
        if self.prompt_adaln is not None and modality.sigma is not None:
            prompt_timestep, prompt_embedded_timestep = (
                self._prepare_timestep_with_adaln(
                    self.prompt_adaln,
                    modality.sigma,
                    x.shape[0],
                    hidden_dtype=x.dtype,
                )
            )

        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
            prompt_timesteps=prompt_timestep,
            prompt_embedded_timestep=prompt_embedded_timestep,
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(
        self,
        patchify_proj: nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection | None,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
        double_precision_rope: bool = False,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ):
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            prompt_adaln=prompt_adaln,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(self, modality: Modality) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)

        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
        )

        cross_scale_shift_timestep, cross_gate_timestep = (
            self._prepare_cross_attention_timestep(
                timestep=modality.timesteps,
                timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
                batch_size=transformer_args.x.shape[0],
                hidden_dtype=transformer_args.x.dtype,
            )
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: mx.array,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: mx.Dtype = None,
    ) -> tuple[mx.array, mx.array]:
        timestep = timestep * timestep_scale_multiplier

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.reshape(-1), hidden_dtype=hidden_dtype
        )
        scale_shift_timestep = mx.reshape(
            scale_shift_timestep, (batch_size, -1, scale_shift_timestep.shape[-1])
        )

        gate_timestep, _ = self.cross_gate_adaln(
            timestep.reshape(-1) * av_ca_factor, hidden_dtype=hidden_dtype
        )
        gate_timestep = mx.reshape(
            gate_timestep, (batch_size, -1, gate_timestep.shape[-1])
        )

        return scale_shift_timestep, gate_timestep


class LTX2_5Model(nn.Module):
    def __init__(self, config: LTX2_5ModelConfig):

        super().__init__()

        self.config = config
        self.model_type = config.model_type
        self.use_middle_indices_grid = config.use_middle_indices_grid
        self.rope_type = config.rope_type
        self.timestep_scale_multiplier = config.timestep_scale_multiplier
        self.positional_embedding_theta = config.positional_embedding_theta

        cross_pe_max_pos = None

        if config.model_type.is_video_enabled():
            self.positional_embedding_max_pos = config.positional_embedding_max_pos
            self.num_attention_heads = config.num_attention_heads
            self.inner_dim = config.inner_dim
            self._init_video(config)

        if config.model_type.is_audio_enabled():
            self.audio_positional_embedding_max_pos = (
                config.audio_positional_embedding_max_pos
            )
            self.audio_num_attention_heads = config.audio_num_attention_heads
            self.audio_inner_dim = config.audio_inner_dim
            self._init_audio(config)

        if (
            config.model_type.is_video_enabled()
            and config.model_type.is_audio_enabled()
        ):
            cross_pe_max_pos = max(
                config.positional_embedding_max_pos[0],
                config.audio_positional_embedding_max_pos[0],
            )
            self.av_ca_timestep_scale_multiplier = (
                config.av_ca_timestep_scale_multiplier
            )
            self.audio_cross_attention_dim = config.audio_cross_attention_dim
            self._init_audio_video(config)

        # 2.5 delta: video/audio embeddings connector（strict load 需要）。
        self._init_connectors(config)

        # 2.5 delta: keyframes absolute pos embedding（1 key，[1, inner_dim]）。
        if config.use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = mx.zeros((1, self.inner_dim))

        self._init_preprocessors(config, cross_pe_max_pos)

        self._init_transformer_blocks(config)

    def _init_video(self, config: LTX2_5ModelConfig) -> None:
        self.patchify_proj = nn.Linear(config.in_channels, self.inner_dim, bias=True)

        adaln_coefficient = 9 if config.has_prompt_adaln else 6
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, embedding_coefficient=adaln_coefficient
        )

        if config.has_prompt_adaln:
            self.prompt_adaln_single = AdaLayerNormSingle(
                self.inner_dim, embedding_coefficient=2
            )
        else:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=config.caption_channels,
                hidden_size=self.inner_dim,
            )

        self.scale_shift_table = mx.zeros((2, self.inner_dim))
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=config.norm_eps, affine=False)
        self.proj_out = nn.Linear(self.inner_dim, config.out_channels)

    def _init_audio(self, config: LTX2_5ModelConfig) -> None:
        self.audio_patchify_proj = nn.Linear(
            config.audio_in_channels, self.audio_inner_dim, bias=True
        )

        audio_adaln_coefficient = 9 if config.has_prompt_adaln else 6
        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim, embedding_coefficient=audio_adaln_coefficient
        )

        if config.has_prompt_adaln:
            self.audio_prompt_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim, embedding_coefficient=2
            )
        else:
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=config.audio_caption_channels,
                hidden_size=self.audio_inner_dim,
            )

        self.audio_scale_shift_table = mx.zeros((2, self.audio_inner_dim))
        self.audio_norm_out = nn.LayerNorm(
            self.audio_inner_dim, eps=config.norm_eps, affine=False
        )
        self.audio_proj_out = nn.Linear(self.audio_inner_dim, config.audio_out_channels)

    def _init_audio_video(self, config: LTX2_5ModelConfig) -> None:
        num_scale_shift_values = 4

        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_connectors(self, config: LTX2_5ModelConfig) -> None:
        # 2.5 delta: 消费 Gemma4 caption embedding 的 1D transformer 连接器。
        # video connector inner_dim=4096，audio connector inner_dim=2048，各 8 层。
        if config.model_type.is_video_enabled():
            self.video_embeddings_connector = Embeddings1DConnector(
                attention_head_dim=config.attention_head_dim,
                num_attention_heads=config.num_attention_heads,
                num_layers=config.connector_num_layers,
                positional_embedding_theta=config.positional_embedding_theta,
                positional_embedding_max_pos=config.connector_positional_embedding_max_pos,
                num_learnable_registers=config.connector_num_learnable_registers,
                rope_type=config.rope_type,
                double_precision_rope=config.double_precision_rope,
                apply_gated_attention=config.connector_apply_gated_attention,
                ff_bias=True,
                norm_eps=config.norm_eps,
            )
        if config.model_type.is_audio_enabled():
            self.audio_embeddings_connector = Embeddings1DConnector(
                attention_head_dim=config.audio_attention_head_dim,
                num_attention_heads=config.audio_num_attention_heads,
                num_layers=config.connector_num_layers,
                positional_embedding_theta=config.positional_embedding_theta,
                positional_embedding_max_pos=config.connector_positional_embedding_max_pos,
                num_learnable_registers=config.connector_num_learnable_registers,
                rope_type=config.rope_type,
                double_precision_rope=config.double_precision_rope,
                apply_gated_attention=config.connector_apply_gated_attention,
                ff_bias=True,
                norm_eps=config.norm_eps,
            )

    def _init_preprocessors(
        self, config: LTX2_5ModelConfig, cross_pe_max_pos: int | None
    ) -> None:
        if (
            config.model_type.is_video_enabled()
            and config.model_type.is_audio_enabled()
        ):
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=getattr(self, "caption_projection", None),
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=config.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=config.use_middle_indices_grid,
                audio_cross_attention_dim=config.audio_cross_attention_dim,
                timestep_scale_multiplier=config.timestep_scale_multiplier,
                positional_embedding_theta=config.positional_embedding_theta,
                rope_type=config.rope_type,
                av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
                double_precision_rope=config.double_precision_rope,
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=getattr(self, "audio_caption_projection", None),
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=config.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=config.use_middle_indices_grid,
                audio_cross_attention_dim=config.audio_cross_attention_dim,
                timestep_scale_multiplier=config.timestep_scale_multiplier,
                positional_embedding_theta=config.positional_embedding_theta,
                rope_type=config.rope_type,
                av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
                double_precision_rope=config.double_precision_rope,
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif config.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=getattr(self, "caption_projection", None),
                inner_dim=self.inner_dim,
                max_pos=config.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=config.use_middle_indices_grid,
                timestep_scale_multiplier=config.timestep_scale_multiplier,
                positional_embedding_theta=config.positional_embedding_theta,
                rope_type=config.rope_type,
                double_precision_rope=config.double_precision_rope,
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
        elif config.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=getattr(self, "audio_caption_projection", None),
                inner_dim=self.audio_inner_dim,
                max_pos=config.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=config.use_middle_indices_grid,
                timestep_scale_multiplier=config.timestep_scale_multiplier,
                positional_embedding_theta=config.positional_embedding_theta,
                rope_type=config.rope_type,
                double_precision_rope=config.double_precision_rope,
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(self, config: LTX2_5ModelConfig) -> None:
        video_config = config.get_video_config()
        audio_config = config.get_audio_config()

        self.transformer_blocks = {
            idx: BasicAVTransformerBlock(
                idx=idx,
                video=video_config,
                audio=audio_config,
                rope_type=config.rope_type,
                norm_eps=config.norm_eps,
                has_prompt_adaln=config.has_prompt_adaln,
                ff_bias=config.ff_bias,
                audio_ff_bias=config.audio_ff_bias,
            )
            for idx in range(config.num_layers)
        }

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        stg_video_blocks: list[int] | None = None,
        stg_audio_blocks: list[int] | None = None,
        skip_cross_modal: bool = False,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        stg_v_set = set(stg_video_blocks) if stg_video_blocks else set()
        stg_a_set = set(stg_audio_blocks) if stg_audio_blocks else set()
        for idx, block in self.transformer_blocks.items():
            video, audio = block(
                video=video,
                audio=audio,
                skip_video_self_attn=(idx in stg_v_set),
                skip_audio_self_attn=(idx in stg_a_set),
                skip_cross_modal=skip_cross_modal,
            )
        return video, audio

    def _process_output(
        self,
        scale_shift_table: mx.array,
        norm_out: nn.LayerNorm,
        proj_out: nn.Linear,
        x: mx.array,
        embedded_timestep: mx.array,
    ) -> mx.array:

        table_expanded = scale_shift_table[None, None, :, :]
        timestep_expanded = embedded_timestep[:, :, None, :]

        scale_shift_values = table_expanded + timestep_expanded

        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)

        return x

    def __call__(
        self,
        video: Modality | None = None,
        audio: Modality | None = None,
        stg_video_blocks: list[int] | None = None,
        stg_audio_blocks: list[int] | None = None,
        skip_cross_modal: bool = False,
    ) -> tuple[mx.array | None, mx.array | None]:

        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = (
            self.video_args_preprocessor.prepare(video) if video is not None else None
        )
        audio_args = (
            self.audio_args_preprocessor.prepare(audio) if audio is not None else None
        )

        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            stg_video_blocks=stg_video_blocks,
            stg_audio_blocks=stg_audio_blocks,
            skip_cross_modal=skip_cross_modal,
        )

        vx = (
            self._process_output(
                self.scale_shift_table,
                self.norm_out,
                self.proj_out,
                video_out.x,
                video_out.embedded_timestep,
            )
            if video_out is not None
            else None
        )

        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )

        return vx, ax

    @staticmethod
    def _remap_raw(k: str) -> str:
        # 原始 diffusers 键名 -> MLX 模块树键名（canon 与 flat connector 共用）。
        k = k.replace(".to_out.0.", ".to_out.")
        k = k.replace(".ff.net.0.proj.", ".ff.proj_in.")
        k = k.replace(".ff.net.2.", ".ff.proj_out.")
        k = k.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
        k = k.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
        k = k.replace(".linear_1.", ".linear1.")
        k = k.replace(".linear_2.", ".linear2.")
        return k

    def sanitize(self, weights: dict) -> dict:
        # 与 ltx2 的差异：不跳过 connector keys（2.5 connector 是 model 子模块）。
        # 三种布局：
        #   1. canon Comfy：model.diffusion_model. 前缀 + 原始键名 -> 剥前缀 + remap。
        #   2. flat diffusers (#762)：transformer. 前缀（键名已 remapped，仅剥前缀）
        #      + connector. 前缀（原始键名，剥前缀 + remap），两文件合并后传入。
        #   3. 其他：原样返回。
        sanitized = {}

        has_raw_prefix = any(k.startswith("model.diffusion_model.") for k in weights)
        if has_raw_prefix:
            for key, value in weights.items():
                if not key.startswith("model.diffusion_model."):
                    continue
                sanitized[
                    self._remap_raw(key.replace("model.diffusion_model.", ""))
                ] = value
            return sanitized

        has_flat_transformer = any(k.startswith("transformer.") for k in weights)
        has_flat_connector = any(k.startswith("connector.") for k in weights)
        if has_flat_transformer or has_flat_connector:
            for key, value in weights.items():
                if key.startswith("transformer."):
                    # flat transformer 键名已是 MLX 格式（to_out 非 to_out.0），
                    # 仅剥前缀，不再 remap。
                    sanitized[key[len("transformer.") :]] = value
                elif key.startswith("connector."):
                    sanitized[self._remap_raw(key[len("connector.") :])] = value
            return sanitized

        return weights

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str | Path,
        config: LTX2_5ModelConfig | None = None,
        variant: LTX2_5Variant | str = LTX2_5Variant.DISTILLED,
        strict: bool = True,
        connector_weights: str | Path | None = None,
    ) -> "LTX2_5Model":
        # 2.5 单文件 checkpoint 无 config.json，配置由 default_ltx2_5_config 合成。
        # flat diffusers 布局 (#762) 把 connector 单列为独立文件，通过
        # connector_weights 传入并合并到 transformer 键集中统一 sanitize。
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

        if weights_path.is_file() and weights_path.suffix == ".safetensors":
            weights = mx.load(str(weights_path))
            weight_files = [weights_path]
        else:
            weight_files = sorted(weights_path.glob("*.safetensors"))
            if not weight_files:
                raise FileNotFoundError(f"no .safetensors under {weights_path}")
            weights = {}
            for wf in weight_files:
                weights.update(mx.load(str(wf)))
        if connector_weights is not None:
            connector_weights = Path(connector_weights)
            if connector_weights.exists():
                cw = mx.load(str(connector_weights))
                logger.info(
                    "ltx2_5 from_pretrained: merged connector %s (%d keys)",
                    connector_weights.name,
                    len(cw),
                )
                weights.update(cw)
        logger.info(
            "ltx2_5 from_pretrained: loaded %d weight keys from %d files",
            len(weights),
            len(weight_files),
        )

        sanitized = model.sanitize(weights)
        sanitized = {
            k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
            for k, v in sanitized.items()
        }

        # 量化检测 (#762): checkpoint 含 .scales 键 → 用 checkpoint 驱动的
        # class_predicate 在 load_weights 前 nn.quantize，使 Linear 变为
        # QuantizedLinear（带 .scales/.biases 参数）以匹配 q8 键。predicate
        # 用模块树路径 path（与 checkpoint 键前缀一致）判断该 Linear 是否在
        # checkpoint 中有对应 .scales。group_size/bits 从 split_model.json 读。
        is_quantized = any(k.endswith(".scales") for k in sanitized)
        if is_quantized:
            group_size, bits = _read_quant_config(weights_path.parent)
            sanitized_keys_set = set(sanitized.keys())

            def _quant_predicate(path, module):
                return isinstance(module, nn.Linear) and (
                    f"{path}.scales" in sanitized_keys_set
                )

            nn.quantize(
                model,
                group_size=group_size,
                bits=bits,
                class_predicate=_quant_predicate,
            )
            logger.info(
                "ltx2_5 from_pretrained: quantized group_size=%d bits=%d",
                group_size,
                bits,
            )

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
                raise RuntimeError(
                    f"LTX-2.5 weight key-tree mismatch: "
                    f"unmatched={len(unmatched)} missing={len(missing)} "
                    f"(see warning log above). Re-run with strict=False to inspect."
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


class LTX2_5X0Model(nn.Module):
    def __init__(self, velocity_model: LTX2_5Model):

        super().__init__()
        self.velocity_model = velocity_model

    def __call__(
        self,
        video: Modality | None = None,
        audio: Modality | None = None,
        stg_video_blocks: list[int] | None = None,
        stg_audio_blocks: list[int] | None = None,
        skip_cross_modal: bool = False,
    ) -> tuple[mx.array | None, mx.array | None]:

        vx, ax = self.velocity_model(
            video,
            audio,
            stg_video_blocks=stg_video_blocks,
            stg_audio_blocks=stg_audio_blocks,
            skip_cross_modal=skip_cross_modal,
        )

        denoised_video = (
            to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        )
        denoised_audio = (
            to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        )

        return denoised_video, denoised_audio
