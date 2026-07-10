import logging
import math

import mlx.core as mx

from .conditioning import LatentState, apply_denoise_mask
from .guidance import apg_delta
from .ltx2_model import LTXModel
from .transformer import Modality

logger = logging.getLogger(__name__)


def denoise_distilled(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer: LTXModel,
    sigmas: list,
    verbose: bool = True,
    state: LatentState | None = None,
    audio_latents: mx.array | None = None,
    audio_positions: mx.array | None = None,
    audio_embeddings: mx.array | None = None,
    audio_frozen: bool = False,
) -> tuple[mx.array, mx.array | None]:
    dtype = latents.dtype
    enable_audio = audio_latents is not None

    if state is not None:
        latents = state.latent

    latents = latents.astype(mx.float32)
    if enable_audio:
        audio_latents = audio_latents.astype(mx.float32)

    desc = "Denoising A/V" if enable_audio else "Denoising"
    num_steps = len(sigmas) - 1

    if verbose:
        logger.info("%s: %d steps", desc, num_steps)

    for i in range(num_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        b, c, f, h, w = latents.shape
        num_tokens = f * h * w
        latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1)).astype(
            dtype
        )

        if state is not None:
            denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
            timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

        video_modality = Modality(
            latent=latents_flat,
            timesteps=timesteps,
            positions=positions,
            context=text_embeddings,
            context_mask=None,
            enabled=True,
            sigma=mx.full((b,), sigma, dtype=dtype),
        )

        audio_modality = None
        if enable_audio:
            ab, ac, at, af = audio_latents.shape
            audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
            audio_flat = mx.reshape(audio_flat, (ab, at, ac * af)).astype(dtype)

            a_ts = (
                mx.zeros((ab, at), dtype=dtype)
                if audio_frozen
                else mx.full((ab, at), sigma, dtype=dtype)
            )
            a_sig = (
                mx.zeros((ab,), dtype=dtype)
                if audio_frozen
                else mx.full((ab,), sigma, dtype=dtype)
            )
            audio_modality = Modality(
                latent=audio_flat,
                timesteps=a_ts,
                positions=audio_positions,
                context=audio_embeddings,
                context_mask=None,
                enabled=True,
                sigma=a_sig,
            )

        velocity, audio_velocity = transformer(
            video=video_modality, audio=audio_modality
        )
        mx.eval(velocity)
        if audio_velocity is not None:
            mx.eval(audio_velocity)

        sigma_f32 = mx.array(sigma, dtype=mx.float32)
        latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
        timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
        x0_f32 = latents_flat_f32 - timesteps_f32 * velocity.astype(mx.float32)
        denoised = mx.reshape(mx.transpose(x0_f32, (0, 2, 1)), (b, c, f, h, w))

        audio_denoised = None
        if enable_audio and audio_velocity is not None and not audio_frozen:
            ab, ac, at, af = audio_latents.shape
            audio_velocity = mx.reshape(audio_velocity, (ab, at, ac, af))
            audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
            audio_denoised = audio_latents - sigma_f32 * audio_velocity.astype(
                mx.float32
            )

        if state is not None:
            denoised = apply_denoise_mask(
                denoised, state.clean_latent.astype(mx.float32), state.denoise_mask
            )

        mx.eval(denoised)
        if audio_denoised is not None:
            mx.eval(audio_denoised)

        if sigma_next > 0:
            sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
            latents = denoised + sigma_next_f32 * (latents - denoised) / sigma_f32
            if enable_audio and audio_denoised is not None and not audio_frozen:
                audio_latents = (
                    audio_denoised
                    + sigma_next_f32 * (audio_latents - audio_denoised) / sigma_f32
                )
        else:
            latents = denoised
            if enable_audio and audio_denoised is not None and not audio_frozen:
                audio_latents = audio_denoised

        mx.eval(latents)
        if enable_audio:
            mx.eval(audio_latents)

        if verbose:
            logger.info("step %d/%d", i + 1, num_steps)

    return latents.astype(dtype), audio_latents.astype(dtype) if enable_audio else None


def denoise_dev(
    latents: mx.array,
    positions: mx.array,
    text_embeddings_pos: mx.array,
    text_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 4.0,
    cfg_rescale: float = 0.0,
    verbose: bool = True,
    state: LatentState | None = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    stg_scale: float = 0.0,
    stg_blocks: list | None = None,
) -> mx.array:
    from .rope import precompute_freqs_cis

    dtype = latents.dtype
    if state is not None:
        latents = state.latent

    latents = latents.astype(mx.float32)

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0
    use_stg = stg_scale != 0.0 and stg_blocks is not None
    num_steps = len(sigmas_list) - 1

    precomputed_rope = precompute_freqs_cis(
        positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_rope)

    passes = ["CFG"] if use_cfg else []
    if use_stg:
        passes.append("STG")
    label = "+".join(passes) if passes else "uncond"
    if verbose:
        logger.info("Denoising (%s): %d steps", label, num_steps)

    for i in range(num_steps):
        sigma = sigmas_list[i]
        sigma_next = sigmas_list[i + 1]

        b, c, f, h, w = latents.shape
        num_tokens = f * h * w
        latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1)).astype(
            dtype
        )

        if state is not None:
            denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
            timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

        sigma_array = mx.full((b,), sigma, dtype=dtype)

        video_modality_pos = Modality(
            latent=latents_flat,
            timesteps=timesteps,
            positions=positions,
            context=text_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_rope,
            sigma=sigma_array,
        )
        velocity_pos, _ = transformer(video=video_modality_pos, audio=None)

        latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
        timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
        x0_pos_f32 = latents_flat_f32 - timesteps_f32 * velocity_pos.astype(mx.float32)

        x0_guided_f32 = x0_pos_f32

        if use_cfg:
            video_modality_neg = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=text_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_rope,
                sigma=sigma_array,
            )
            velocity_neg, _ = transformer(video=video_modality_neg, audio=None)

            x0_neg_f32 = latents_flat_f32 - timesteps_f32 * velocity_neg.astype(
                mx.float32
            )

            if use_apg:
                x0_guided_f32 = x0_pos_f32 + apg_delta(
                    x0_pos_f32,
                    x0_neg_f32,
                    cfg_scale,
                    eta=apg_eta,
                    norm_threshold=apg_norm_threshold,
                )
            else:
                x0_guided_f32 = x0_pos_f32 + (cfg_scale - 1.0) * (
                    x0_pos_f32 - x0_neg_f32
                )

        if use_stg:
            velocity_ptb, _ = transformer(
                video=video_modality_pos,
                audio=None,
                stg_video_blocks=stg_blocks,
            )
            mx.eval(velocity_ptb)

            x0_ptb_f32 = latents_flat_f32 - timesteps_f32 * velocity_ptb.astype(
                mx.float32
            )
            x0_guided_f32 = x0_guided_f32 + stg_scale * (x0_pos_f32 - x0_ptb_f32)

        if cfg_rescale > 0.0 and (use_cfg or use_stg):
            v_factor = x0_pos_f32.std() / (x0_guided_f32.std() + 1e-8)
            v_factor = cfg_rescale * v_factor + (1.0 - cfg_rescale)
            x0_guided_f32 = x0_guided_f32 * v_factor

        denoised = mx.reshape(mx.transpose(x0_guided_f32, (0, 2, 1)), (b, c, f, h, w))

        sigma_f32 = mx.array(sigma, dtype=mx.float32)

        if state is not None:
            denoised = apply_denoise_mask(
                denoised, state.clean_latent.astype(mx.float32), state.denoise_mask
            )

        if sigma_next > 0:
            sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
            latents = denoised + sigma_next_f32 * (latents - denoised) / sigma_f32
        else:
            latents = denoised

        mx.eval(latents)
        if verbose:
            logger.info("step %d/%d", i + 1, num_steps)

    return latents.astype(dtype)


def denoise_dev_av(
    video_latents: mx.array,
    audio_latents: mx.array,
    video_positions: mx.array,
    audio_positions: mx.array,
    video_embeddings_pos: mx.array,
    video_embeddings_neg: mx.array,
    audio_embeddings_pos: mx.array,
    audio_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 4.0,
    audio_cfg_scale: float = 7.0,
    cfg_rescale: float = 0.0,
    verbose: bool = True,
    video_state: LatentState | None = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    stg_scale: float = 0.0,
    stg_video_blocks: list | None = None,
    stg_audio_blocks: list | None = None,
    modality_scale: float = 1.0,
    audio_frozen: bool = False,
) -> tuple[mx.array, mx.array]:
    from .rope import precompute_freqs_cis

    dtype = video_latents.dtype
    if video_state is not None:
        video_latents = video_state.latent

    video_latents = video_latents.astype(mx.float32)
    audio_latents = audio_latents.astype(mx.float32)

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0
    use_stg = stg_scale != 0.0 and stg_video_blocks is not None
    use_modality = modality_scale != 1.0
    num_steps = len(sigmas_list) - 1

    precomputed_video_rope = precompute_freqs_cis(
        video_positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )

    precomputed_audio_rope = precompute_freqs_cis(
        audio_positions,
        dim=transformer.audio_inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.audio_positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.audio_num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_video_rope, precomputed_audio_rope)

    passes = ["CFG"] if use_cfg else []
    if use_stg:
        passes.append("STG")
    if use_modality:
        passes.append("Mod")
    label = "+".join(passes) if passes else "uncond"
    if verbose:
        logger.info("Denoising A/V (%s): %d steps", label, num_steps)

    for i in range(num_steps):
        sigma = sigmas_list[i]
        sigma_next = sigmas_list[i + 1]

        b, c, f, h, w = video_latents.shape
        num_video_tokens = f * h * w
        video_flat = mx.transpose(
            mx.reshape(video_latents, (b, c, -1)), (0, 2, 1)
        ).astype(dtype)

        ab, ac, at, af = audio_latents.shape
        audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
        audio_flat = mx.reshape(audio_flat, (ab, at, ac * af)).astype(dtype)

        if video_state is not None:
            denoise_mask_flat = mx.reshape(video_state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))
            video_timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            video_timesteps = mx.full((b, num_video_tokens), sigma, dtype=dtype)

        audio_timesteps = (
            mx.zeros((ab, at), dtype=dtype)
            if audio_frozen
            else mx.full((ab, at), sigma, dtype=dtype)
        )

        sigma_array = mx.full((b,), sigma, dtype=dtype)
        audio_sigma_array = (
            mx.zeros((ab,), dtype=dtype)
            if audio_frozen
            else mx.full((ab,), sigma, dtype=dtype)
        )
        video_modality_pos = Modality(
            latent=video_flat,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_video_rope,
            sigma=sigma_array,
        )
        audio_modality_pos = Modality(
            latent=audio_flat,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_audio_rope,
            sigma=audio_sigma_array,
        )
        video_vel_pos, audio_vel_pos = transformer(
            video=video_modality_pos, audio=audio_modality_pos
        )
        mx.eval(video_vel_pos, audio_vel_pos)

        video_flat_f32 = mx.transpose(mx.reshape(video_latents, (b, c, -1)), (0, 2, 1))
        audio_flat_f32 = mx.reshape(
            mx.transpose(audio_latents, (0, 2, 1, 3)), (ab, at, ac * af)
        )
        video_timesteps_f32 = mx.expand_dims(
            video_timesteps.astype(mx.float32), axis=-1
        )
        audio_timesteps_f32 = mx.expand_dims(
            audio_timesteps.astype(mx.float32), axis=-1
        )

        video_x0_pos_f32 = video_flat_f32 - video_timesteps_f32 * video_vel_pos.astype(
            mx.float32
        )
        audio_x0_pos_f32 = audio_flat_f32 - audio_timesteps_f32 * audio_vel_pos.astype(
            mx.float32
        )

        video_x0_guided_f32 = video_x0_pos_f32
        audio_x0_guided_f32 = audio_x0_pos_f32

        if use_cfg:
            video_modality_neg = Modality(
                latent=video_flat,
                timesteps=video_timesteps,
                positions=video_positions,
                context=video_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_video_rope,
                sigma=sigma_array,
            )
            audio_modality_neg = Modality(
                latent=audio_flat,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=audio_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_audio_rope,
                sigma=audio_sigma_array,
            )
            video_vel_neg, audio_vel_neg = transformer(
                video=video_modality_neg, audio=audio_modality_neg
            )
            mx.eval(video_vel_neg, audio_vel_neg)

            video_x0_neg_f32 = (
                video_flat_f32 - video_timesteps_f32 * video_vel_neg.astype(mx.float32)
            )
            audio_x0_neg_f32 = (
                audio_flat_f32 - audio_timesteps_f32 * audio_vel_neg.astype(mx.float32)
            )

            if use_apg:
                video_x0_guided_f32 = video_x0_pos_f32 + apg_delta(
                    video_x0_pos_f32,
                    video_x0_neg_f32,
                    cfg_scale,
                    eta=apg_eta,
                    norm_threshold=apg_norm_threshold,
                )
            else:
                video_x0_guided_f32 = video_x0_pos_f32 + (cfg_scale - 1.0) * (
                    video_x0_pos_f32 - video_x0_neg_f32
                )
            audio_x0_guided_f32 = audio_x0_pos_f32 + (audio_cfg_scale - 1.0) * (
                audio_x0_pos_f32 - audio_x0_neg_f32
            )

        if use_stg:
            video_vel_ptb, audio_vel_ptb = transformer(
                video=video_modality_pos,
                audio=audio_modality_pos,
                stg_video_blocks=stg_video_blocks,
                stg_audio_blocks=stg_audio_blocks,
            )
            mx.eval(video_vel_ptb, audio_vel_ptb)

            video_x0_ptb_f32 = (
                video_flat_f32 - video_timesteps_f32 * video_vel_ptb.astype(mx.float32)
            )
            audio_x0_ptb_f32 = (
                audio_flat_f32 - audio_timesteps_f32 * audio_vel_ptb.astype(mx.float32)
            )

            video_x0_guided_f32 = video_x0_guided_f32 + stg_scale * (
                video_x0_pos_f32 - video_x0_ptb_f32
            )
            audio_x0_guided_f32 = audio_x0_guided_f32 + stg_scale * (
                audio_x0_pos_f32 - audio_x0_ptb_f32
            )

        if use_modality:
            video_vel_iso, audio_vel_iso = transformer(
                video=video_modality_pos,
                audio=audio_modality_pos,
                skip_cross_modal=True,
            )
            mx.eval(video_vel_iso, audio_vel_iso)

            video_x0_iso_f32 = (
                video_flat_f32 - video_timesteps_f32 * video_vel_iso.astype(mx.float32)
            )
            audio_x0_iso_f32 = (
                audio_flat_f32 - audio_timesteps_f32 * audio_vel_iso.astype(mx.float32)
            )

            video_x0_guided_f32 = video_x0_guided_f32 + (modality_scale - 1.0) * (
                video_x0_pos_f32 - video_x0_iso_f32
            )
            audio_x0_guided_f32 = audio_x0_guided_f32 + (modality_scale - 1.0) * (
                audio_x0_pos_f32 - audio_x0_iso_f32
            )

        if cfg_rescale > 0.0 and (use_cfg or use_stg or use_modality):
            v_factor = video_x0_pos_f32.std() / (video_x0_guided_f32.std() + 1e-8)
            v_factor = cfg_rescale * v_factor + (1.0 - cfg_rescale)
            video_x0_guided_f32 = video_x0_guided_f32 * v_factor
            a_factor = audio_x0_pos_f32.std() / (audio_x0_guided_f32.std() + 1e-8)
            a_factor = cfg_rescale * a_factor + (1.0 - cfg_rescale)
            audio_x0_guided_f32 = audio_x0_guided_f32 * a_factor

        video_denoised_f32 = mx.reshape(
            mx.transpose(video_x0_guided_f32, (0, 2, 1)), (b, c, f, h, w)
        )
        audio_denoised_f32 = mx.reshape(audio_x0_guided_f32, (ab, at, ac, af))
        audio_denoised_f32 = mx.transpose(audio_denoised_f32, (0, 2, 1, 3))

        sigma_f32 = mx.array(sigma, dtype=mx.float32)

        if video_state is not None:
            clean_f32 = video_state.clean_latent.astype(mx.float32)
            mask_f32 = video_state.denoise_mask.astype(mx.float32)
            video_denoised_f32 = video_denoised_f32 * mask_f32 + clean_f32 * (
                1.0 - mask_f32
            )

        mx.eval(video_denoised_f32, audio_denoised_f32)

        if sigma_next > 0:
            sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
            dt_f32 = sigma_next_f32 - sigma_f32

            video_velocity_f32 = (video_latents - video_denoised_f32) / sigma_f32
            video_latents = video_latents + video_velocity_f32 * dt_f32

            if not audio_frozen:
                audio_velocity_f32 = (audio_latents - audio_denoised_f32) / sigma_f32
                audio_latents = audio_latents + audio_velocity_f32 * dt_f32
        else:
            video_latents = video_denoised_f32
            if not audio_frozen:
                audio_latents = audio_denoised_f32

        mx.eval(video_latents, audio_latents)
        if verbose:
            logger.info("step %d/%d", i + 1, num_steps)

    return video_latents, audio_latents


def denoise_res2s_av(
    video_latents: mx.array,
    audio_latents: mx.array,
    video_positions: mx.array,
    audio_positions: mx.array,
    video_embeddings_pos: mx.array,
    video_embeddings_neg: mx.array,
    audio_embeddings_pos: mx.array,
    audio_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 3.0,
    audio_cfg_scale: float = 7.0,
    cfg_rescale: float = 0.45,
    audio_cfg_rescale: float | None = None,
    verbose: bool = True,
    video_state: LatentState | None = None,
    stg_scale: float = 0.0,
    stg_video_blocks: list | None = None,
    stg_audio_blocks: list | None = None,
    modality_scale: float = 1.0,
    noise_seed: int = 42,
    bongmath: bool = True,
    bongmath_max_iter: int = 100,
    audio_frozen: bool = False,
) -> tuple[mx.array, mx.array]:
    from .rope import precompute_freqs_cis
    from .samplers import (
        get_new_noise,
        get_res2s_coefficients,
        sde_noise_step,
    )

    if audio_cfg_rescale is None:
        audio_cfg_rescale = cfg_rescale

    dtype = video_latents.dtype
    if video_state is not None:
        video_latents = video_state.latent

    video_latents = video_latents.astype(mx.float32)
    audio_latents = audio_latents.astype(mx.float32)

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0
    use_stg = stg_scale != 0.0 and stg_video_blocks is not None
    use_modality = modality_scale != 1.0
    n_full_steps = len(sigmas_list) - 1

    if sigmas_list[-1] == 0:
        sigmas_list = sigmas_list[:-1] + [0.0011, 0.0]

    hs = [-math.log(sigmas_list[i + 1] / sigmas_list[i]) for i in range(n_full_steps)]

    precomputed_video_rope = precompute_freqs_cis(
        video_positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    precomputed_audio_rope = precompute_freqs_cis(
        audio_positions,
        dim=transformer.audio_inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.audio_positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.audio_num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_video_rope, precomputed_audio_rope)

    phi_cache = {}
    c2 = 0.5

    step_noise_key = mx.random.key(noise_seed)
    substep_noise_key = mx.random.key(noise_seed + 10000)

    def _eval_guided_denoise(v_latents, a_latents, sigma):
        b, c, f, h, w = v_latents.shape
        num_video_tokens = f * h * w
        video_flat = mx.transpose(mx.reshape(v_latents, (b, c, -1)), (0, 2, 1)).astype(
            dtype
        )

        ab, ac, at, af = a_latents.shape
        audio_flat = mx.transpose(a_latents, (0, 2, 1, 3))
        audio_flat = mx.reshape(audio_flat, (ab, at, ac * af)).astype(dtype)

        if video_state is not None:
            denoise_mask_flat = mx.reshape(video_state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))
            video_timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            video_timesteps = mx.full((b, num_video_tokens), sigma, dtype=dtype)
        audio_timesteps = (
            mx.zeros((ab, at), dtype=dtype)
            if audio_frozen
            else mx.full((ab, at), sigma, dtype=dtype)
        )

        sigma_array = mx.full((b,), sigma, dtype=dtype)
        audio_sigma_array = (
            mx.zeros((ab,), dtype=dtype)
            if audio_frozen
            else mx.full((ab,), sigma, dtype=dtype)
        )

        video_modality_pos = Modality(
            latent=video_flat,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_video_rope,
            sigma=sigma_array,
        )
        audio_modality_pos = Modality(
            latent=audio_flat,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_audio_rope,
            sigma=audio_sigma_array,
        )
        video_vel_pos, audio_vel_pos = transformer(
            video=video_modality_pos, audio=audio_modality_pos
        )
        mx.eval(video_vel_pos, audio_vel_pos)

        video_flat_f32 = mx.transpose(mx.reshape(v_latents, (b, c, -1)), (0, 2, 1))
        audio_flat_f32 = mx.reshape(
            mx.transpose(a_latents, (0, 2, 1, 3)), (ab, at, ac * af)
        )
        video_ts_f32 = mx.expand_dims(video_timesteps.astype(mx.float32), axis=-1)
        audio_ts_f32 = mx.expand_dims(audio_timesteps.astype(mx.float32), axis=-1)

        video_x0_pos = video_flat_f32 - video_ts_f32 * video_vel_pos.astype(mx.float32)
        audio_x0_pos = audio_flat_f32 - audio_ts_f32 * audio_vel_pos.astype(mx.float32)

        video_x0_guided = video_x0_pos
        audio_x0_guided = audio_x0_pos

        if use_cfg:
            video_modality_neg = Modality(
                latent=video_flat,
                timesteps=video_timesteps,
                positions=video_positions,
                context=video_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_video_rope,
                sigma=sigma_array,
            )
            audio_modality_neg = Modality(
                latent=audio_flat,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=audio_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_audio_rope,
                sigma=audio_sigma_array,
            )
            video_vel_neg, audio_vel_neg = transformer(
                video=video_modality_neg, audio=audio_modality_neg
            )
            mx.eval(video_vel_neg, audio_vel_neg)

            video_x0_neg = video_flat_f32 - video_ts_f32 * video_vel_neg.astype(
                mx.float32
            )
            audio_x0_neg = audio_flat_f32 - audio_ts_f32 * audio_vel_neg.astype(
                mx.float32
            )

            video_x0_guided = video_x0_pos + (cfg_scale - 1.0) * (
                video_x0_pos - video_x0_neg
            )
            audio_x0_guided = audio_x0_pos + (audio_cfg_scale - 1.0) * (
                audio_x0_pos - audio_x0_neg
            )

        if use_stg:
            video_vel_ptb, audio_vel_ptb = transformer(
                video=video_modality_pos,
                audio=audio_modality_pos,
                stg_video_blocks=stg_video_blocks,
                stg_audio_blocks=stg_audio_blocks,
            )
            mx.eval(video_vel_ptb, audio_vel_ptb)

            video_x0_ptb = video_flat_f32 - video_ts_f32 * video_vel_ptb.astype(
                mx.float32
            )
            audio_x0_ptb = audio_flat_f32 - audio_ts_f32 * audio_vel_ptb.astype(
                mx.float32
            )

            video_x0_guided = video_x0_guided + stg_scale * (
                video_x0_pos - video_x0_ptb
            )
            audio_x0_guided = audio_x0_guided + stg_scale * (
                audio_x0_pos - audio_x0_ptb
            )

        if use_modality:
            video_vel_iso, audio_vel_iso = transformer(
                video=video_modality_pos,
                audio=audio_modality_pos,
                skip_cross_modal=True,
            )
            mx.eval(video_vel_iso, audio_vel_iso)

            video_x0_iso = video_flat_f32 - video_ts_f32 * video_vel_iso.astype(
                mx.float32
            )
            audio_x0_iso = audio_flat_f32 - audio_ts_f32 * audio_vel_iso.astype(
                mx.float32
            )

            video_x0_guided = video_x0_guided + (modality_scale - 1.0) * (
                video_x0_pos - video_x0_iso
            )
            audio_x0_guided = audio_x0_guided + (modality_scale - 1.0) * (
                audio_x0_pos - audio_x0_iso
            )

        if cfg_rescale > 0.0 and (use_cfg or use_stg or use_modality):
            v_factor = video_x0_pos.std() / (video_x0_guided.std() + 1e-8)
            v_factor = cfg_rescale * v_factor + (1.0 - cfg_rescale)
            video_x0_guided = video_x0_guided * v_factor
        if audio_cfg_rescale > 0.0 and (use_cfg or use_stg or use_modality):
            a_factor = audio_x0_pos.std() / (audio_x0_guided.std() + 1e-8)
            a_factor = audio_cfg_rescale * a_factor + (1.0 - audio_cfg_rescale)
            audio_x0_guided = audio_x0_guided * a_factor

        video_denoised = mx.reshape(
            mx.transpose(video_x0_guided, (0, 2, 1)), (b, c, f, h, w)
        )
        audio_denoised = mx.reshape(audio_x0_guided, (ab, at, ac, af))
        audio_denoised = mx.transpose(audio_denoised, (0, 2, 1, 3))

        if video_state is not None:
            clean_f32 = video_state.clean_latent.astype(mx.float32)
            mask_f32 = video_state.denoise_mask.astype(mx.float32)
            video_denoised = video_denoised * mask_f32 + clean_f32 * (1.0 - mask_f32)

        mx.eval(video_denoised, audio_denoised)
        return video_denoised, audio_denoised

    passes = ["res2s"]
    if use_cfg:
        passes.append("CFG")
    if use_stg:
        passes.append("STG")
    if use_modality:
        passes.append("Mod")
    label = "+".join(passes)
    if verbose:
        logger.info("Denoising A/V (%s): %d steps", label, n_full_steps)

    for step_idx in range(n_full_steps):
        sigma = sigmas_list[step_idx]
        sigma_next = sigmas_list[step_idx + 1]
        h = hs[step_idx]

        x_anchor_video = video_latents
        x_anchor_audio = audio_latents

        denoised_video_1, denoised_audio_1 = _eval_guided_denoise(
            video_latents, audio_latents, sigma
        )

        a21, b1, b2 = get_res2s_coefficients(h, phi_cache, c2)

        sub_sigma = math.sqrt(sigma * sigma_next)

        eps_1_video = denoised_video_1 - x_anchor_video
        x_mid_video = x_anchor_video + h * a21 * eps_1_video

        if not audio_frozen:
            eps_1_audio = denoised_audio_1 - x_anchor_audio
            x_mid_audio = x_anchor_audio + h * a21 * eps_1_audio
        else:
            eps_1_audio = None
            x_mid_audio = audio_latents

        substep_noise_key, key1, key2 = mx.random.split(substep_noise_key, 3)
        substep_noise_v = get_new_noise(video_latents.shape, key1)

        x_mid_video = sde_noise_step(
            x_anchor_video, x_mid_video, sigma, sub_sigma, substep_noise_v
        )
        if not audio_frozen:
            substep_noise_a = get_new_noise(audio_latents.shape, key2)
            x_mid_audio = sde_noise_step(
                x_anchor_audio, x_mid_audio, sigma, sub_sigma, substep_noise_a
            )
        mx.eval(x_mid_video, x_mid_audio)

        if bongmath and h < 0.5 and sigma > 0.03:
            for _ in range(bongmath_max_iter):
                x_anchor_video = x_mid_video - h * a21 * eps_1_video
                eps_1_video = denoised_video_1 - x_anchor_video
                if not audio_frozen:
                    x_anchor_audio = x_mid_audio - h * a21 * eps_1_audio
                    eps_1_audio = denoised_audio_1 - x_anchor_audio
            if audio_frozen:
                mx.eval(x_anchor_video, eps_1_video)
            else:
                mx.eval(x_anchor_video, x_anchor_audio, eps_1_video, eps_1_audio)

        denoised_video_2, denoised_audio_2 = _eval_guided_denoise(
            x_mid_video.astype(mx.float32),
            x_mid_audio.astype(mx.float32),
            sub_sigma,
        )

        eps_2_video = denoised_video_2 - x_anchor_video
        x_next_video = x_anchor_video + h * (b1 * eps_1_video + b2 * eps_2_video)

        step_noise_key, key1, key2 = mx.random.split(step_noise_key, 3)
        step_noise_v = get_new_noise(video_latents.shape, key1)
        x_next_video = sde_noise_step(
            x_anchor_video, x_next_video, sigma, sigma_next, step_noise_v
        )

        video_latents = x_next_video.astype(mx.float32)
        if not audio_frozen:
            eps_2_audio = denoised_audio_2 - x_anchor_audio
            x_next_audio = x_anchor_audio + h * (b1 * eps_1_audio + b2 * eps_2_audio)
            step_noise_a = get_new_noise(audio_latents.shape, key2)
            x_next_audio = sde_noise_step(
                x_anchor_audio, x_next_audio, sigma, sigma_next, step_noise_a
            )
            audio_latents = x_next_audio.astype(mx.float32)

        mx.eval(video_latents, audio_latents)
        if verbose:
            logger.info("step %d/%d", step_idx + 1, n_full_steps)

    if sigmas.tolist()[-1] == 0:
        denoised_video, denoised_audio = _eval_guided_denoise(
            video_latents, audio_latents, sigmas_list[n_full_steps]
        )
        video_latents = denoised_video
        if not audio_frozen:
            audio_latents = denoised_audio
        mx.eval(video_latents, audio_latents)

    return video_latents, audio_latents
