import argparse
import json
import logging
import time
from enum import Enum
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from fusion_mlx.cache.latent_cache import (
    get_image_latent_cache,
    image_latent_key,
)

from .audio import (
    AUDIO_SAMPLE_RATE,
    load_audio_decoder,
    load_vocoder_model,
    mux_video_audio,
    save_audio,
)
from .conditioning import (
    LatentState,
    VideoConditionByLatentIndex,
    apply_conditioning,
)
from .denoise import denoise_dev_av, denoise_distilled, denoise_res2s_av
from .lora import load_and_merge_lora
from .ltx2_model import LTXModel
from .positions import (
    AUDIO_HOP_LENGTH,
    AUDIO_LATENT_CHANNELS,
    AUDIO_LATENT_SAMPLE_RATE,
    AUDIO_MEL_BINS,
    compute_audio_frames,
    create_audio_position_grid,
    create_position_grid,
)
from .scheduler import ltx2_scheduler
from .text_encoder import LTX2TextEncoder
from .upsampler import load_upsampler, upsample_latents
from .utils import get_model_path, load_image, prepare_image_for_encoding
from .video_vae import VideoEncoder
from .video_vae.decoder import VideoDecoder
from .video_vae.tiling import TilingConfig

logger = logging.getLogger(__name__)


class PipelineType(Enum):
    DISTILLED = "distilled"
    DEV = "dev"
    DEV_TWO_STAGE = "dev-two-stage"
    DEV_TWO_STAGE_HQ = "dev-two-stage-hq"


STAGE_1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


def _build_i2v_conditionings(
    image_latent,
    image_frame_idx: int,
    image_strength: float,
    end_image_latent=None,
    end_image_strength: float = 1.0,
):
    conditionings = []
    if image_latent is not None:
        idx = 0 if end_image_latent is not None else image_frame_idx
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=image_latent, frame_idx=idx, strength=image_strength
            )
        )
    if end_image_latent is not None:
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=end_image_latent, frame_idx=-1, strength=end_image_strength
            )
        )
    return conditionings


def generate_video(
    model_repo: str,
    text_encoder_repo: str | None,
    prompt: str,
    pipeline: PipelineType = PipelineType.DISTILLED,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    height: int = 512,
    width: int = 512,
    num_frames: int = 33,
    num_inference_steps: int = 40,
    cfg_scale: float = 4.0,
    audio_cfg_scale: float = 7.0,
    cfg_rescale: float = 0.0,
    seed: int = 42,
    fps: int = 24,
    output_path: str = "output.mp4",
    save_frames: bool = False,
    verbose: bool = True,
    enhance_prompt: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.7,
    image: str | None = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
    end_image: str | None = None,
    end_image_strength: float | None = None,
    tiling: str = "auto",
    stream: bool = False,
    audio: bool = False,
    output_audio_path: str | None = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    stg_scale: float = 1.0,
    stg_blocks: list | None = None,
    modality_scale: float = 3.0,
    lora_path: str | None = None,
    lora_strength: float = 1.0,
    lora_strength_stage_1: float | None = None,
    lora_strength_stage_2: float | None = None,
    audio_file: str | None = None,
    audio_start_time: float = 0.0,
    spatial_upscaler: str | None = None,
    session_id: str | None = None,
):
    start_time = time.time()

    is_two_stage = pipeline in (
        PipelineType.DISTILLED,
        PipelineType.DEV_TWO_STAGE,
        PipelineType.DEV_TWO_STAGE_HQ,
    )
    divisor = 64 if is_two_stage else 32
    assert height % divisor == 0, f"Height must be divisible by {divisor}, got {height}"
    assert width % divisor == 0, f"Width must be divisible by {divisor}, got {width}"

    if num_frames % 8 != 1:
        adjusted_num_frames = round((num_frames - 1) / 8) * 8 + 1
        logger.warning(
            "Number of frames must be 1 + 8*k. Using: %d", adjusted_num_frames
        )
        num_frames = adjusted_num_frames

    is_i2v = image is not None or end_image is not None
    has_end_image = end_image is not None
    if end_image_strength is None:
        end_image_strength = image_strength
    is_a2v = audio_file is not None
    if is_a2v and audio:
        raise ValueError(
            "Cannot use both --audio-file (A2V) and --audio (generate audio). Choose one."
        )
    if is_a2v:
        audio = True
    mode_str = "I2V" if is_i2v else "T2V"
    if has_end_image and image is not None:
        mode_str = "I2V(first+last)"
    elif has_end_image:
        mode_str = "I2V(last)"
    if is_a2v:
        mode_str = "A2V" + ("+I2V" if is_i2v else "")
    elif audio:
        mode_str += "+Audio"

    pipeline_names = {
        PipelineType.DISTILLED: "DISTILLED",
        PipelineType.DEV: "DEV",
        PipelineType.DEV_TWO_STAGE: "DEV-TWO-STAGE",
        PipelineType.DEV_TWO_STAGE_HQ: "DEV-TWO-STAGE-HQ",
    }
    pipeline_name = pipeline_names[pipeline]
    logger.info("=" * 60)
    logger.info(
        "[%s] [%s] %dx%d - %d frames",
        pipeline_name,
        mode_str,
        width,
        height,
        num_frames,
    )
    logger.info("Prompt: %s", prompt[:80] + ("..." if len(prompt) > 80 else ""))

    if pipeline in (
        PipelineType.DEV,
        PipelineType.DEV_TWO_STAGE,
        PipelineType.DEV_TWO_STAGE_HQ,
    ):
        audio_cfg_info = f", Audio CFG: {audio_cfg_scale}" if audio else ""
        stg_info = f", STG: {stg_scale} blocks={stg_blocks}" if stg_scale != 0.0 else ""
        mod_info = f", Modality: {modality_scale}" if modality_scale != 1.0 else ""
        logger.info(
            "Steps: %d, CFG: %s%s, Rescale: %s%s%s",
            num_inference_steps,
            cfg_scale,
            audio_cfg_info,
            cfg_rescale,
            stg_info,
            mod_info,
        )

    if is_i2v:
        if image is not None:
            logger.info(
                "First image: %s (strength=%s, frame=%d)",
                image,
                image_strength,
                image_frame_idx,
            )
        if has_end_image:
            logger.info(
                "Last image: %s (strength=%s, frame=-1)", end_image, end_image_strength
            )

    audio_frames = compute_audio_frames(num_frames, fps)
    if audio:
        logger.info("Audio: %d latent frames @ %dHz", audio_frames, AUDIO_SAMPLE_RATE)

    model_path = get_model_path(model_repo)
    text_encoder_path = (
        model_path if text_encoder_repo is None else get_model_path(text_encoder_repo)
    )

    upscaler_path = None
    upscaler_scale = 2.0
    if is_two_stage:
        if spatial_upscaler is not None:
            upscaler_path = (
                model_path / spatial_upscaler
                if not Path(spatial_upscaler).is_absolute()
                else Path(spatial_upscaler)
            )
            if not upscaler_path.exists():
                upscaler_path = model_path / spatial_upscaler
            if "x1.5" in str(upscaler_path):
                upscaler_scale = 1.5
            elif "x2" in str(upscaler_path):
                upscaler_scale = 2.0
        else:
            upscaler_files = sorted(
                model_path.glob("*spatial-upscaler-x2*.safetensors")
            )
            if upscaler_files:
                upscaler_path = upscaler_files[0]
                upscaler_scale = 2.0

    if is_two_stage:
        stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
        stage2_h = int(stage1_h * upscaler_scale)
        stage2_w = int(stage1_w * upscaler_scale)
    else:
        latent_h, latent_w = height // 32, width // 32
    latent_frames = 1 + (num_frames - 1) // 8

    mx.random.seed(seed)

    transformer_config_path = model_path / "transformer" / "config.json"
    has_prompt_adaln = False
    if transformer_config_path.exists():
        with open(transformer_config_path) as f:
            has_prompt_adaln = json.load(f).get("has_prompt_adaln", False)

    logger.info("Loading text encoder...")
    text_encoder = LTX2TextEncoder(has_prompt_adaln=has_prompt_adaln)
    text_encoder.load(model_path=model_path, text_encoder_path=text_encoder_path)
    mx.eval(text_encoder.parameters())
    logger.info("Text encoder loaded")

    if enhance_prompt:
        logger.info("Enhancing prompt")
        prompt = text_encoder.enhance_t2v(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            verbose=verbose,
        )
        logger.info("Enhanced: %s", prompt[:150] + ("..." if len(prompt) > 150 else ""))

    if pipeline in (
        PipelineType.DEV,
        PipelineType.DEV_TWO_STAGE,
        PipelineType.DEV_TWO_STAGE_HQ,
    ):
        video_embeddings_pos, audio_embeddings_pos = text_encoder(
            prompt, return_audio_embeddings=True
        )
        video_embeddings_neg, audio_embeddings_neg = text_encoder(
            negative_prompt, return_audio_embeddings=True
        )
        model_dtype = video_embeddings_pos.dtype
        mx.eval(
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
        )
        if pipeline in (PipelineType.DEV_TWO_STAGE, PipelineType.DEV_TWO_STAGE_HQ):
            text_embeddings = video_embeddings_pos
    else:
        text_embeddings, audio_embeddings = text_encoder(
            prompt, return_audio_embeddings=True
        )
        mx.eval(text_embeddings, audio_embeddings)
        model_dtype = text_embeddings.dtype

    del text_encoder
    mx.clear_cache()

    logger.info(
        "Loading %s transformer%s...",
        pipeline_name.lower(),
        " (A/V mode)" if audio else "",
    )
    transformer = LTXModel.from_pretrained(
        model_path=model_path / "transformer", strict=True
    )
    logger.info("Transformer loaded")

    if stg_blocks is None and stg_scale != 0.0:
        if transformer.config.has_prompt_adaln:
            stg_blocks = [28]
        else:
            stg_blocks = [29]
        logger.info(
            "Auto-detected STG blocks: %s (model=%s)",
            stg_blocks,
            "2.3" if transformer.config.has_prompt_adaln else "2",
        )

    a2v_audio_latents = None
    a2v_waveform = None
    a2v_sr = None
    if is_a2v:
        # Stage E: audio_vae + convert_audio_encoder not yet ported to fusion.
        from .audio_vae import AudioEncoder
        from .audio_vae.audio_processor import (
            ensure_stereo,
            load_audio,
            waveform_to_mel,
        )
        from .utils import convert_audio_encoder

        logger.info("Loading and encoding input audio (A2V)...")
        video_duration = num_frames / fps

        waveform, sr = load_audio(
            audio_file,
            target_sr=AUDIO_LATENT_SAMPLE_RATE,
            start_time=audio_start_time,
            max_duration=video_duration,
        )
        waveform = ensure_stereo(waveform)
        a2v_waveform = waveform.copy()
        a2v_sr = sr

        mel = waveform_to_mel(
            waveform,
            sample_rate=sr,
            n_fft=1024,
            hop_length=AUDIO_HOP_LENGTH,
            n_mels=64,
        )

        encoder_dir = convert_audio_encoder(model_path, source_repo="Lightricks/LTX-2")
        audio_encoder = AudioEncoder.from_pretrained(encoder_dir)
        mx.eval(audio_encoder.parameters())

        encoded = audio_encoder(mel)
        mx.eval(encoded)

        a2v_audio_latents = mx.transpose(encoded, (0, 3, 1, 2)).astype(model_dtype)

        t_encoded = a2v_audio_latents.shape[2]
        if t_encoded > audio_frames:
            a2v_audio_latents = a2v_audio_latents[:, :, :audio_frames, :]
        elif t_encoded < audio_frames:
            pad_size = audio_frames - t_encoded
            padding = mx.zeros(
                (1, AUDIO_LATENT_CHANNELS, pad_size, AUDIO_MEL_BINS),
                dtype=model_dtype,
            )
            a2v_audio_latents = mx.concatenate([a2v_audio_latents, padding], axis=2)
        mx.eval(a2v_audio_latents)

        del audio_encoder
        mx.clear_cache()
        logger.info(
            "Audio encoded (%d frames from %s)", a2v_audio_latents.shape[2], audio_file
        )

    if pipeline == PipelineType.DISTILLED:
        stage1_image_latent = None
        stage2_image_latent = None
        stage1_end_image_latent = None
        stage2_end_image_latent = None
        if is_i2v:
            # Phase-2: session tail→first-frame reuse (skip VAE encode on hit)
            session_tail_hit = False
            if session_id is not None:
                from fusion_mlx.cache.latent_cache import get_session_tail

                tail = get_session_tail(session_id, model_repo)
                if tail is not None:
                    t_h, t_w = tail.shape[3], tail.shape[4]
                    if t_h == stage2_h and t_w == stage2_w:
                        stage2_image_latent = tail
                        session_tail_hit = True
                        logger.info(
                            "session tail cache hit: %s stage2 %dx%d",
                            session_id, t_h, t_w,
                        )
                    else:
                        logger.info(
                            "session tail cache skip: shape mismatch %dx%d vs %dx%d",
                            t_h, t_w, stage2_h, stage2_w,
                        )

            if not session_tail_hit:
                logger.info("Loading VAE encoder and encoding image(s)...")
            # UMA Radix Latent cache (#2 Phase-1): repeat I2V requests with
            # the same image+resolution reuse the cached VAE latent and skip
            # the VAE encoder load + forward entirely (zero-copy on UMA).
            latent_cache = get_image_latent_cache(model_repo)
            vae_encoder = None

            s1_h, s1_w = stage1_h * 32, stage1_w * 32
            s2_h, s2_w = stage2_h * 32, stage2_w * 32

            def _encode_image_latent(src, h, w):
                nonlocal vae_encoder
                key = image_latent_key(model_repo, src, h, w, model_dtype)
                if latent_cache is not None:
                    cached = latent_cache.get(key)
                    if cached is not None:
                        logger.info("latent cache hit: %dx%d (%s)", h, w, key)
                        return cached
                if vae_encoder is None:
                    vae_encoder = VideoEncoder.from_pretrained(
                        model_path / "vae" / "encoder"
                    )
                loaded = load_image(src, height=h, width=w, dtype=model_dtype)
                latent = vae_encoder(
                    prepare_image_for_encoding(loaded, h, w, dtype=model_dtype)
                )
                mx.eval(latent)
                if latent_cache is not None:
                    latent_cache.put(key, latent)
                    logger.info("latent cache miss+insert: %dx%d (%s)", h, w, key)
                return latent

            if image is not None:
                stage1_image_latent = _encode_image_latent(image, s1_h, s1_w)
                stage2_image_latent = _encode_image_latent(image, s2_h, s2_w)

            if has_end_image:
                stage1_end_image_latent = _encode_image_latent(end_image, s1_h, s1_w)
                stage2_end_image_latent = _encode_image_latent(end_image, s2_h, s2_w)

            if vae_encoder is not None:
                del vae_encoder
                mx.clear_cache()
            logger.info("VAE encoder loaded and image(s) encoded")

        logger.info(
            "Stage 1: Generating at %dx%d (8 steps)", stage1_w * 32, stage1_h * 32
        )
        mx.random.seed(seed)

        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
        mx.eval(positions)

        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS)
            ).astype(model_dtype)
        )
        mx.eval(audio_positions, audio_latents)

        state1 = None
        if is_i2v and (
            stage1_image_latent is not None or stage1_end_image_latent is not None
        ):
            latent_shape = (1, 128, latent_frames, stage1_h, stage1_w)
            state1 = LatentState(
                latent=mx.zeros(latent_shape, dtype=model_dtype),
                clean_latent=mx.zeros(latent_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                stage1_image_latent,
                image_frame_idx,
                image_strength,
                stage1_end_image_latent,
                end_image_strength,
            )
            state1 = apply_conditioning(state1, conditionings)

            noise = mx.random.normal(latent_shape, dtype=model_dtype)
            noise_scale = mx.array(STAGE_1_SIGMAS[0], dtype=model_dtype)
            scaled_mask = state1.denoise_mask * noise_scale
            state1 = LatentState(
                latent=noise * scaled_mask
                + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state1.clean_latent,
                denoise_mask=state1.denoise_mask,
            )
            latents = state1.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(
                (1, 128, latent_frames, stage1_h, stage1_w), dtype=model_dtype
            )
            mx.eval(latents)

        latents, audio_latents = denoise_distilled(
            latents,
            positions,
            text_embeddings,
            transformer,
            STAGE_1_SIGMAS,
            verbose=verbose,
            state=state1,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_embeddings,
            audio_frozen=is_a2v,
        )

        logger.info("Upsampling latents %dx...", upscaler_scale)
        if upscaler_path is None or not upscaler_path.exists():
            raise FileNotFoundError(f"No spatial upscaler found in {model_path}")
        upsampler, upscaler_scale = load_upsampler(str(upscaler_path))
        mx.eval(upsampler.parameters())

        vae_decoder = VideoDecoder.from_pretrained(str(model_path / "vae" / "decoder"))

        latents = upsample_latents(
            latents,
            upsampler,
            vae_decoder.per_channel_statistics.mean,
            vae_decoder.per_channel_statistics.std,
        )
        mx.eval(latents)

        del upsampler
        mx.clear_cache()
        logger.info("Latents upsampled")

        logger.info(
            "Stage 2: Refining at %dx%d (3 steps)", stage2_w * 32, stage2_h * 32
        )
        positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
        mx.eval(positions)

        state2 = None
        if is_i2v and (
            stage2_image_latent is not None or stage2_end_image_latent is not None
        ):
            state2 = LatentState(
                latent=latents,
                clean_latent=mx.zeros_like(latents),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                stage2_image_latent,
                image_frame_idx,
                image_strength,
                stage2_end_image_latent,
                end_image_strength,
            )
            state2 = apply_conditioning(state2, conditionings)

            noise = mx.random.normal(latents.shape).astype(model_dtype)
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            scaled_mask = state2.denoise_mask * noise_scale
            state2 = LatentState(
                latent=noise * scaled_mask
                + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state2.clean_latent,
                denoise_mask=state2.denoise_mask,
            )
            latents = state2.latent
            mx.eval(latents)
        else:
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            one_minus_scale = mx.array(1.0 - STAGE_2_SIGMAS[0], dtype=model_dtype)
            noise = mx.random.normal(latents.shape).astype(model_dtype)
            latents = noise * noise_scale + latents * one_minus_scale
            mx.eval(latents)

        if audio_latents is not None and not is_a2v:
            audio_noise = mx.random.normal(audio_latents.shape, dtype=model_dtype)
            audio_noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            audio_latents = audio_noise * audio_noise_scale + audio_latents * (
                mx.array(1.0, dtype=model_dtype) - audio_noise_scale
            )
            mx.eval(audio_latents)

        latents, audio_latents = denoise_distilled(
            latents,
            positions,
            text_embeddings,
            transformer,
            STAGE_2_SIGMAS,
            verbose=verbose,
            state=state2,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_embeddings,
            audio_frozen=is_a2v,
        )

        # Phase-2: capture tail-frame latent for multi-shot session reuse
        if session_id is not None:
            from fusion_mlx.cache.latent_cache import put_session_tail

            tail = latents[:, :, -1:, :, :]
            put_session_tail(session_id, model_repo, tail)

    elif pipeline == PipelineType.DEV:
        image_latent = None
        end_image_latent = None
        if is_i2v:
            # Phase-2: check session tail cache first (reuse previous shot's
            # denoised tail-frame latent as this shot's first-frame conditioning).
            from fusion_mlx.cache.latent_cache import get_session_tail

            session_tail_reused = False
            if session_id is not None and image is not None:
                tail_latent = get_session_tail(session_id, model_repo)
                if tail_latent is not None:
                    # Resize if resolution changed between shots
                    if (
                        tail_latent.shape[2] == 1
                        and tail_latent.shape[3] == latent_h
                        and tail_latent.shape[4] == latent_w
                    ):
                        image_latent = tail_latent
                        session_tail_reused = True
                        logger.info(
                            "session tail reused as first-frame latent (skip VAE encode)"
                        )
                    else:
                        logger.info(
                            "session tail shape mismatch: got %s expected [1,C,1,%d,%d], "
                            "falling back to image encode",
                            tail_latent.shape,
                            latent_h,
                            latent_w,
                        )

            if not session_tail_reused:
                logger.info("Loading VAE encoder and encoding image(s)...")
                # UMA Radix Latent cache (#2 Phase-1): repeat I2V requests with
                # the same image+resolution reuse the cached VAE latent and skip
                # the VAE encoder load + forward entirely (zero-copy on UMA).
                latent_cache = get_image_latent_cache(model_repo)
                vae_encoder = None

                def _encode_image_latent(src, h, w):
                    nonlocal vae_encoder
                    key = image_latent_key(model_repo, src, h, w, model_dtype)
                    if latent_cache is not None:
                        cached = latent_cache.get(key)
                        if cached is not None:
                            logger.info("latent cache hit: %dx%d (%s)", h, w, key)
                            return cached
                    if vae_encoder is None:
                        vae_encoder = VideoEncoder.from_pretrained(
                            model_path / "vae" / "encoder"
                        )
                    loaded = load_image(src, height=h, width=w, dtype=model_dtype)
                    latent = vae_encoder(
                        prepare_image_for_encoding(loaded, h, w, dtype=model_dtype)
                    )
                    mx.eval(latent)
                    if latent_cache is not None:
                        latent_cache.put(key, latent)
                        logger.info("latent cache miss+insert: %dx%d (%s)", h, w, key)
                    return latent

                if image is not None:
                    image_latent = _encode_image_latent(image, height, width)

                if has_end_image:
                    end_image_latent = _encode_image_latent(end_image, height, width)

                if vae_encoder is not None:
                    del vae_encoder
                    mx.clear_cache()
                logger.info("VAE encoder loaded and image(s) encoded")

        sigmas = ltx2_scheduler(steps=num_inference_steps)
        mx.eval(sigmas)
        logger.info(
            "Sigma schedule: %.4f -> %.4f -> %.4f",
            sigmas[0].item(),
            sigmas[-2].item(),
            sigmas[-1].item(),
        )

        logger.info(
            "Generating: %dx%d (%d steps, CFG=%s, rescale=%s)",
            width,
            height,
            num_inference_steps,
            cfg_scale,
            cfg_rescale,
        )
        mx.random.seed(seed)

        video_positions = create_position_grid(1, latent_frames, latent_h, latent_w)
        mx.eval(video_positions)

        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS),
                dtype=model_dtype,
            )
        )
        mx.eval(audio_positions, audio_latents)

        video_state = None
        video_latent_shape = (1, 128, latent_frames, latent_h, latent_w)
        if is_i2v and (image_latent is not None or end_image_latent is not None):
            video_state = LatentState(
                latent=mx.zeros(video_latent_shape, dtype=model_dtype),
                clean_latent=mx.zeros(video_latent_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                image_latent,
                image_frame_idx,
                image_strength,
                end_image_latent,
                end_image_strength,
            )
            video_state = apply_conditioning(video_state, conditionings)

            noise = mx.random.normal(video_latent_shape, dtype=model_dtype)
            noise_scale = sigmas[0]
            scaled_mask = video_state.denoise_mask * noise_scale
            video_state = LatentState(
                latent=noise * scaled_mask
                + video_state.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=video_state.clean_latent,
                denoise_mask=video_state.denoise_mask,
            )
            latents = video_state.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(video_latent_shape, dtype=model_dtype)
            mx.eval(latents)

        latents, audio_latents = denoise_dev_av(
            latents,
            audio_latents,
            video_positions,
            audio_positions,
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
            transformer,
            sigmas,
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            cfg_rescale=cfg_rescale,
            verbose=verbose,
            video_state=video_state,
            use_apg=use_apg,
            apg_eta=apg_eta,
            apg_norm_threshold=apg_norm_threshold,
            stg_scale=stg_scale,
            stg_video_blocks=stg_blocks,
            stg_audio_blocks=stg_blocks,
            modality_scale=modality_scale,
            audio_frozen=is_a2v,
        )

        # Phase-2: capture tail-frame latent for multi-shot session reuse
        if session_id is not None:
            from fusion_mlx.cache.latent_cache import put_session_tail

            tail = latents[:, :, -1:, :, :]
            put_session_tail(session_id, model_repo, tail)

        vae_decoder = VideoDecoder.from_pretrained(str(model_path / "vae" / "decoder"))

    elif pipeline == PipelineType.DEV_TWO_STAGE:
        stage1_image_latent = None
        stage2_image_latent = None
        stage1_end_image_latent = None
        stage2_end_image_latent = None
        if is_i2v:
            # Phase-2: session tail→first-frame reuse (skip VAE encode on hit)
            session_tail_hit = False
            if session_id is not None:
                from fusion_mlx.cache.latent_cache import get_session_tail

                tail = get_session_tail(session_id, model_repo)
                if tail is not None:
                    t_h, t_w = tail.shape[3], tail.shape[4]
                    if t_h == stage2_h and t_w == stage2_w:
                        stage2_image_latent = tail
                        session_tail_hit = True
                        logger.info(
                            "session tail cache hit: %s stage2 %dx%d",
                            session_id, t_h, t_w,
                        )
                    else:
                        logger.info(
                            "session tail cache skip: shape mismatch %dx%d vs %dx%d",
                            t_h, t_w, stage2_h, stage2_w,
                        )

            if not session_tail_hit:
                logger.info("Loading VAE encoder and encoding image(s)...")
            # UMA Radix Latent cache (#2 Phase-1): repeat I2V requests with
            # the same image+resolution reuse the cached VAE latent and skip
            # the VAE encoder load + forward entirely (zero-copy on UMA).
            latent_cache = get_image_latent_cache(model_repo)
            vae_encoder = None

            s1_h, s1_w = stage1_h * 32, stage1_w * 32
            s2_h, s2_w = stage2_h * 32, stage2_w * 32

            def _encode_image_latent(src, h, w):
                nonlocal vae_encoder
                key = image_latent_key(model_repo, src, h, w, model_dtype)
                if latent_cache is not None:
                    cached = latent_cache.get(key)
                    if cached is not None:
                        logger.info("latent cache hit: %dx%d (%s)", h, w, key)
                        return cached
                if vae_encoder is None:
                    vae_encoder = VideoEncoder.from_pretrained(
                        model_path / "vae" / "encoder"
                    )
                loaded = load_image(src, height=h, width=w, dtype=model_dtype)
                latent = vae_encoder(
                    prepare_image_for_encoding(loaded, h, w, dtype=model_dtype)
                )
                mx.eval(latent)
                if latent_cache is not None:
                    latent_cache.put(key, latent)
                    logger.info("latent cache miss+insert: %dx%d (%s)", h, w, key)
                return latent

            if image is not None:
                stage1_image_latent = _encode_image_latent(image, s1_h, s1_w)
                stage2_image_latent = _encode_image_latent(image, s2_h, s2_w)

            if has_end_image:
                stage1_end_image_latent = _encode_image_latent(end_image, s1_h, s1_w)
                stage2_end_image_latent = _encode_image_latent(end_image, s2_h, s2_w)

            if vae_encoder is not None:
                del vae_encoder
                mx.clear_cache()
            logger.info("VAE encoder loaded and image(s) encoded")

        sigmas = ltx2_scheduler(steps=num_inference_steps)
        mx.eval(sigmas)
        logger.info(
            "Stage 1 sigma schedule: %.4f -> %.4f -> %.4f",
            sigmas[0].item(),
            sigmas[-2].item(),
            sigmas[-1].item(),
        )

        logger.info(
            "Stage 1: Dev generating at %dx%d (%d steps, CFG=%s, rescale=%s)",
            stage1_w * 32,
            stage1_h * 32,
            num_inference_steps,
            cfg_scale,
            cfg_rescale,
        )
        mx.random.seed(seed)

        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
        mx.eval(positions)

        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS),
                dtype=model_dtype,
            )
        )
        mx.eval(audio_positions, audio_latents)

        state1 = None
        stage1_shape = (1, 128, latent_frames, stage1_h, stage1_w)
        if is_i2v and (
            stage1_image_latent is not None or stage1_end_image_latent is not None
        ):
            state1 = LatentState(
                latent=mx.zeros(stage1_shape, dtype=model_dtype),
                clean_latent=mx.zeros(stage1_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                stage1_image_latent,
                image_frame_idx,
                image_strength,
                stage1_end_image_latent,
                end_image_strength,
            )
            state1 = apply_conditioning(state1, conditionings)

            noise = mx.random.normal(stage1_shape, dtype=model_dtype)
            noise_scale = sigmas[0]
            scaled_mask = state1.denoise_mask * noise_scale
            state1 = LatentState(
                latent=noise * scaled_mask
                + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state1.clean_latent,
                denoise_mask=state1.denoise_mask,
            )
            latents = state1.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(stage1_shape, dtype=model_dtype)
            mx.eval(latents)

        latents, audio_latents = denoise_dev_av(
            latents,
            audio_latents,
            positions,
            audio_positions,
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
            transformer,
            sigmas,
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            cfg_rescale=cfg_rescale,
            verbose=verbose,
            video_state=state1,
            use_apg=use_apg,
            apg_eta=apg_eta,
            apg_norm_threshold=apg_norm_threshold,
            stg_scale=stg_scale,
            stg_video_blocks=stg_blocks,
            stg_audio_blocks=stg_blocks,
            modality_scale=modality_scale,
            audio_frozen=is_a2v,
        )

        mx.eval(audio_latents)

        logger.info("Upsampling latents %dx...", upscaler_scale)
        if upscaler_path is None or not upscaler_path.exists():
            raise FileNotFoundError(f"No spatial upscaler found in {model_path}")
        upsampler, upscaler_scale = load_upsampler(str(upscaler_path))
        mx.eval(upsampler.parameters())

        vae_decoder = VideoDecoder.from_pretrained(str(model_path / "vae" / "decoder"))

        latents = upsample_latents(
            latents,
            upsampler,
            vae_decoder.per_channel_statistics.mean,
            vae_decoder.per_channel_statistics.std,
        )
        mx.eval(latents)

        del upsampler
        mx.clear_cache()
        logger.info("Latents upsampled")

        if lora_path is None:
            lora_files = sorted(model_path.glob("*distilled-lora*.safetensors"))
            if lora_files:
                lora_path = str(lora_files[0])
                logger.info("Auto-detected LoRA: %s", Path(lora_path).name)
            else:
                logger.warning("No LoRA file found. Stage 2 will use base weights.")

        if lora_path is not None:
            logger.info("Merging distilled LoRA weights...")
            load_and_merge_lora(transformer, lora_path, strength=lora_strength)

        logger.info(
            "Stage 2: Distilled refining at %dx%d (3 steps, no CFG)", width, height
        )
        positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
        mx.eval(positions)

        state2 = None
        if is_i2v and (
            stage2_image_latent is not None or stage2_end_image_latent is not None
        ):
            state2 = LatentState(
                latent=latents,
                clean_latent=mx.zeros_like(latents),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                stage2_image_latent,
                image_frame_idx,
                image_strength,
                stage2_end_image_latent,
                end_image_strength,
            )
            state2 = apply_conditioning(state2, conditionings)

            noise = mx.random.normal(latents.shape).astype(model_dtype)
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            scaled_mask = state2.denoise_mask * noise_scale
            state2 = LatentState(
                latent=noise * scaled_mask
                + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state2.clean_latent,
                denoise_mask=state2.denoise_mask,
            )
            latents = state2.latent
            mx.eval(latents)
        else:
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            one_minus_scale = mx.array(1.0 - STAGE_2_SIGMAS[0], dtype=model_dtype)
            noise = mx.random.normal(latents.shape).astype(model_dtype)
            latents = noise * noise_scale + latents * one_minus_scale
            mx.eval(latents)

        if audio_latents is not None and not is_a2v:
            audio_noise = mx.random.normal(audio_latents.shape, dtype=model_dtype)
            audio_noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            audio_latents = audio_noise * audio_noise_scale + audio_latents * (
                mx.array(1.0, dtype=model_dtype) - audio_noise_scale
            )
            mx.eval(audio_latents)

        latents, audio_latents = denoise_distilled(
            latents,
            positions,
            text_embeddings,
            transformer,
            STAGE_2_SIGMAS,
            verbose=verbose,
            state=state2,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_embeddings_pos,
            audio_frozen=is_a2v,
        )

        # Phase-2: capture tail-frame latent for multi-shot session reuse
        if session_id is not None:
            from fusion_mlx.cache.latent_cache import put_session_tail

            tail = latents[:, :, -1:, :, :]
            put_session_tail(session_id, model_repo, tail)

    elif pipeline == PipelineType.DEV_TWO_STAGE_HQ:
        hq_lora_strength_s1 = (
            lora_strength_stage_1 if lora_strength_stage_1 is not None else 0.25
        )
        hq_lora_strength_s2 = (
            lora_strength_stage_2 if lora_strength_stage_2 is not None else 0.5
        )
        hq_cfg_rescale = cfg_rescale if cfg_rescale != 0.7 else 0.45
        hq_steps = num_inference_steps if num_inference_steps != 30 else 15
        hq_stg_scale = stg_scale if stg_scale != 1.0 else 0.0

        stage1_image_latent = None
        stage2_image_latent = None
        stage1_end_image_latent = None
        stage2_end_image_latent = None
        if is_i2v:
            # Phase-2: session tail→first-frame reuse (skip VAE encode on hit)
            session_tail_hit = False
            if session_id is not None:
                from fusion_mlx.cache.latent_cache import get_session_tail

                tail = get_session_tail(session_id, model_repo)
                if tail is not None:
                    t_h, t_w = tail.shape[3], tail.shape[4]
                    if t_h == stage2_h and t_w == stage2_w:
                        stage2_image_latent = tail
                        session_tail_hit = True
                        logger.info(
                            "session tail cache hit: %s stage2 %dx%d",
                            session_id, t_h, t_w,
                        )
                    else:
                        logger.info(
                            "session tail cache skip: shape mismatch %dx%d vs %dx%d",
                            t_h, t_w, stage2_h, stage2_w,
                        )

            if not session_tail_hit:
                logger.info("Loading VAE encoder and encoding image(s)...")
            # UMA Radix Latent cache (#2 Phase-1): repeat I2V requests with
            # the same image+resolution reuse the cached VAE latent and skip
            # the VAE encoder load + forward entirely (zero-copy on UMA).
            latent_cache = get_image_latent_cache(model_repo)
            vae_encoder = None

            s1_h, s1_w = stage1_h * 32, stage1_w * 32
            s2_h, s2_w = stage2_h * 32, stage2_w * 32

            def _encode_image_latent(src, h, w):
                nonlocal vae_encoder
                key = image_latent_key(model_repo, src, h, w, model_dtype)
                if latent_cache is not None:
                    cached = latent_cache.get(key)
                    if cached is not None:
                        logger.info("latent cache hit: %dx%d (%s)", h, w, key)
                        return cached
                if vae_encoder is None:
                    vae_encoder = VideoEncoder.from_pretrained(
                        model_path / "vae" / "encoder"
                    )
                loaded = load_image(src, height=h, width=w, dtype=model_dtype)
                latent = vae_encoder(
                    prepare_image_for_encoding(loaded, h, w, dtype=model_dtype)
                )
                mx.eval(latent)
                if latent_cache is not None:
                    latent_cache.put(key, latent)
                    logger.info("latent cache miss+insert: %dx%d (%s)", h, w, key)
                return latent

            if image is not None:
                stage1_image_latent = _encode_image_latent(image, s1_h, s1_w)
                stage2_image_latent = _encode_image_latent(image, s2_h, s2_w)

            if has_end_image:
                stage1_end_image_latent = _encode_image_latent(end_image, s1_h, s1_w)
                stage2_end_image_latent = _encode_image_latent(end_image, s2_h, s2_w)

            if vae_encoder is not None:
                del vae_encoder
                mx.clear_cache()
            logger.info("VAE encoder loaded and image(s) encoded")

        if lora_path is None:
            lora_files = sorted(model_path.glob("*distilled-lora*.safetensors"))
            if lora_files:
                lora_path = str(lora_files[0])
                logger.info("Auto-detected LoRA: %s", Path(lora_path).name)
            else:
                logger.warning(
                    "No LoRA file found. HQ pipeline works best with distilled LoRA."
                )

        if lora_path is not None:
            logger.info(
                "Merging distilled LoRA (stage 1, strength=%s)...", hq_lora_strength_s1
            )
            load_and_merge_lora(transformer, lora_path, strength=hq_lora_strength_s1)

        num_tokens = latent_frames * stage1_h * stage1_w
        sigmas = ltx2_scheduler(steps=hq_steps, num_tokens=num_tokens)
        mx.eval(sigmas)
        logger.info(
            "Stage 1 sigma schedule: %.4f -> %.4f -> %.4f (tokens=%d)",
            sigmas[0].item(),
            sigmas[-2].item(),
            sigmas[-1].item(),
            num_tokens,
        )

        logger.info(
            "Stage 1: res_2s at %dx%d (%d steps, CFG=%s, rescale=%s)",
            stage1_w * 32,
            stage1_h * 32,
            hq_steps,
            cfg_scale,
            hq_cfg_rescale,
        )
        mx.random.seed(seed)

        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
        mx.eval(positions)

        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS),
                dtype=model_dtype,
            )
        )
        mx.eval(audio_positions, audio_latents)

        state1 = None
        stage1_shape = (1, 128, latent_frames, stage1_h, stage1_w)
        if is_i2v and (
            stage1_image_latent is not None or stage1_end_image_latent is not None
        ):
            state1 = LatentState(
                latent=mx.zeros(stage1_shape, dtype=model_dtype),
                clean_latent=mx.zeros(stage1_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                stage1_image_latent,
                image_frame_idx,
                image_strength,
                stage1_end_image_latent,
                end_image_strength,
            )
            state1 = apply_conditioning(state1, conditionings)

            noise = mx.random.normal(stage1_shape, dtype=model_dtype)
            noise_scale = sigmas[0]
            scaled_mask = state1.denoise_mask * noise_scale
            state1 = LatentState(
                latent=noise * scaled_mask
                + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state1.clean_latent,
                denoise_mask=state1.denoise_mask,
            )
            latents = state1.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(stage1_shape, dtype=model_dtype)
            mx.eval(latents)

        latents, audio_latents = denoise_res2s_av(
            latents,
            audio_latents,
            positions,
            audio_positions,
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
            transformer,
            sigmas,
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            cfg_rescale=hq_cfg_rescale,
            audio_cfg_rescale=1.0,
            verbose=verbose,
            video_state=state1,
            stg_scale=hq_stg_scale,
            stg_video_blocks=stg_blocks,
            stg_audio_blocks=stg_blocks,
            modality_scale=modality_scale,
            noise_seed=seed,
            audio_frozen=is_a2v,
        )

        mx.eval(audio_latents)

        logger.info("Upsampling latents %dx...", upscaler_scale)
        if upscaler_path is None or not upscaler_path.exists():
            raise FileNotFoundError(f"No spatial upscaler found in {model_path}")
        upsampler, upscaler_scale = load_upsampler(str(upscaler_path))
        mx.eval(upsampler.parameters())

        vae_decoder = VideoDecoder.from_pretrained(str(model_path / "vae" / "decoder"))

        latents = upsample_latents(
            latents,
            upsampler,
            vae_decoder.per_channel_statistics.mean,
            vae_decoder.per_channel_statistics.std,
        )
        mx.eval(latents)

        del upsampler
        mx.clear_cache()
        logger.info("Latents upsampled")

        if lora_path is not None:
            additional_strength = hq_lora_strength_s2 - hq_lora_strength_s1
            if additional_strength > 0:
                logger.info(
                    "Adjusting LoRA (stage 2, total=%s)...", hq_lora_strength_s2
                )
                load_and_merge_lora(
                    transformer, lora_path, strength=additional_strength
                )

        logger.info(
            "Stage 2: res_2s refining at %dx%d (3 steps, no CFG)",
            stage2_w * 32,
            stage2_h * 32,
        )
        positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
        mx.eval(positions)

        state2 = None
        if is_i2v and (
            stage2_image_latent is not None or stage2_end_image_latent is not None
        ):
            state2 = LatentState(
                latent=latents,
                clean_latent=mx.zeros_like(latents),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                stage2_image_latent,
                image_frame_idx,
                image_strength,
                stage2_end_image_latent,
                end_image_strength,
            )
            state2 = apply_conditioning(state2, conditionings)

            noise = mx.random.normal(latents.shape).astype(model_dtype)
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            scaled_mask = state2.denoise_mask * noise_scale
            state2 = LatentState(
                latent=noise * scaled_mask
                + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state2.clean_latent,
                denoise_mask=state2.denoise_mask,
            )
            latents = state2.latent
            mx.eval(latents)
        else:
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            one_minus_scale = mx.array(1.0 - STAGE_2_SIGMAS[0], dtype=model_dtype)
            noise = mx.random.normal(latents.shape).astype(model_dtype)
            latents = noise * noise_scale + latents * one_minus_scale
            mx.eval(latents)

        if audio_latents is not None and not is_a2v:
            audio_noise = mx.random.normal(audio_latents.shape, dtype=model_dtype)
            audio_noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            audio_latents = audio_noise * audio_noise_scale + audio_latents * (
                mx.array(1.0, dtype=model_dtype) - audio_noise_scale
            )
            mx.eval(audio_latents)

        stage2_sigmas = mx.array(STAGE_2_SIGMAS, dtype=mx.float32)
        latents, audio_latents = denoise_res2s_av(
            latents,
            audio_latents,
            positions,
            audio_positions,
            video_embeddings_pos,
            video_embeddings_pos,
            audio_embeddings_pos,
            audio_embeddings_pos,
            transformer,
            stage2_sigmas,
            cfg_scale=1.0,
            audio_cfg_scale=1.0,
            cfg_rescale=0.0,
            verbose=verbose,
            video_state=state2,
            noise_seed=seed + 1,
            audio_frozen=is_a2v,
        )

        # Phase-2: capture tail-frame latent for multi-shot session reuse
        if session_id is not None:
            from fusion_mlx.cache.latent_cache import put_session_tail

            tail = latents[:, :, -1:, :, :]
            put_session_tail(session_id, model_repo, tail)

    del transformer
    mx.clear_cache()

    logger.info("Decoding video...")

    if tiling == "none":
        tiling_config = None
    elif tiling == "auto":
        tiling_config = TilingConfig.auto(height, width, num_frames)
    elif tiling == "default":
        tiling_config = TilingConfig.default()
    elif tiling == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif tiling == "conservative":
        tiling_config = TilingConfig.conservative()
    elif tiling == "spatial":
        tiling_config = TilingConfig.spatial_only()
    elif tiling == "temporal":
        tiling_config = TilingConfig.temporal_only()
    else:
        logger.warning("Unknown tiling mode '%s', using auto", tiling)
        tiling_config = TilingConfig.auto(height, width, num_frames)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_writer = None
    stream_frame_count = 0

    if stream and tiling_config is not None:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        logger.info("Streaming frames to %s as decoded", output_path)

        def on_frames_ready(frames: mx.array, _start_idx: int):
            nonlocal stream_frame_count
            frames = mx.squeeze(frames, axis=0)
            frames = mx.transpose(frames, (1, 2, 3, 0))
            frames = mx.clip((frames + 1.0) / 2.0, 0.0, 1.0)
            frames = (frames * 255).astype(mx.uint8)
            frames_np = np.array(frames)

            for frame in frames_np:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                stream_frame_count += 1
                if stream_frame_count % 8 == 0:
                    logger.info("streamed %d/%d frames", stream_frame_count, num_frames)

    else:
        on_frames_ready = None

    if tiling_config is not None:
        spatial_info = (
            f"{tiling_config.spatial_config.tile_size_in_pixels}px"
            if tiling_config.spatial_config
            else "none"
        )
        temporal_info = (
            f"{tiling_config.temporal_config.tile_size_in_frames}f"
            if tiling_config.temporal_config
            else "none"
        )
        logger.info(
            "Tiling (%s): spatial=%s, temporal=%s", tiling, spatial_info, temporal_info
        )
        video = vae_decoder.decode_tiled(
            latents,
            tiling_config=tiling_config,
            tiling_mode=tiling,
            debug=verbose,
            on_frames_ready=on_frames_ready,
        )
    else:
        logger.info("Tiling: disabled")
        video = vae_decoder(latents)
    mx.eval(video)
    mx.clear_cache()

    if video_writer is not None:
        video_writer.release()
        logger.info("Streamed video to %s", output_path)
        video = mx.squeeze(video, axis=0)
        video = mx.transpose(video, (1, 2, 3, 0))
        video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
        video = (video * 255).astype(mx.uint8)
        video_np = np.array(video)
    else:
        video = mx.squeeze(video, axis=0)
        video = mx.transpose(video, (1, 2, 3, 0))
        video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
        video = (video * 255).astype(mx.uint8)
        video_np = np.array(video)

        if audio:
            temp_video_path = output_path.with_suffix(".temp.mp4")
            save_path = temp_video_path
        else:
            save_path = output_path

        try:
            import cv2

            h, w = video_np.shape[1], video_np.shape[2]
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
            for frame in video_np:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            if not audio:
                logger.info("Saved video to %s", output_path)
        except Exception as e:
            logger.error("Could not save video: %s", e)

    audio_np = None
    vocoder_sample_rate = AUDIO_SAMPLE_RATE
    if audio and audio_latents is not None:
        if is_a2v and a2v_waveform is not None:
            audio_np = a2v_waveform
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]
            vocoder_sample_rate = a2v_sr or AUDIO_LATENT_SAMPLE_RATE
            logger.info("Using original input audio (A2V)")
        else:
            logger.info("Decoding audio...")
            audio_decoder = load_audio_decoder(model_path, pipeline)
            vocoder = load_vocoder_model(model_path, pipeline)
            mx.eval(audio_decoder.parameters(), vocoder.parameters())

            mel_spectrogram = audio_decoder(audio_latents)
            mx.eval(mel_spectrogram)
            logger.info(
                "Mel spectrogram: shape=%s, std=%.4f, mean=%.4f",
                mel_spectrogram.shape,
                mel_spectrogram.std().item(),
                mel_spectrogram.mean().item(),
            )

            audio_waveform = vocoder(mel_spectrogram)
            mx.eval(audio_waveform)

            audio_np = np.array(audio_waveform.astype(mx.float32))
            if audio_np.ndim == 3:
                audio_np = audio_np[0]

            vocoder_sample_rate = getattr(
                vocoder, "output_sampling_rate", AUDIO_SAMPLE_RATE
            )

            del audio_decoder, vocoder
            mx.clear_cache()
            logger.info("Audio decoded")

        audio_path = (
            Path(output_audio_path)
            if output_audio_path
            else output_path.with_suffix(".wav")
        )
        save_audio(audio_np, audio_path, vocoder_sample_rate)
        logger.info("Saved audio to %s", audio_path)

        logger.info("Combining video and audio...")
        temp_video_path = output_path.with_suffix(".temp.mp4")
        success = mux_video_audio(temp_video_path, audio_path, output_path)
        if success:
            logger.info("Saved video with audio to %s", output_path)
            temp_video_path.unlink()
        else:
            temp_video_path.rename(output_path)
            logger.warning("Saved video without audio to %s", output_path)

    del vae_decoder
    mx.clear_cache()

    if save_frames:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(video_np):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        logger.info("Saved %d frames to %s", len(video_np), frames_dir)

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    time_str = f"{int(minutes)}m {seconds:.1f}s" if minutes >= 1 else f"{seconds:.1f}s"
    logger.info("Done! Generated in %s (%.2fs/frame)", time_str, elapsed / num_frames)
    logger.info("Peak memory: %.2f GB", mx.get_peak_memory() / (1024**3))

    if audio:
        return video_np, audio_np
    return video_np


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos with MLX LTX-2 (Distilled or Dev pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distilled pipeline (two-stage, fast, no CFG)
  python -m fusion_mlx.video.ltx2.generate --prompt "A cat walking on grass"
  python -m fusion_mlx.video.ltx2.generate --prompt "Ocean waves" --pipeline distilled

  # Dev pipeline (single-stage, CFG, higher quality)
  python -m fusion_mlx.video.ltx2.generate --prompt "A cat walking" --pipeline dev --cfg-scale 3.0
  python -m fusion_mlx.video.ltx2.generate --prompt "Ocean waves" --pipeline dev --steps 40

  # Dev two-stage pipeline (dev + LoRA refinement)
  python -m fusion_mlx.video.ltx2.generate --prompt "A cat walking" --pipeline dev-two-stage --cfg-scale 3.0

  # Image-to-Video (works with both pipelines)
  python -m fusion_mlx.video.ltx2.generate --prompt "A person dancing" --image photo.jpg
  python -m fusion_mlx.video.ltx2.generate --prompt "Waves crashing" --image beach.png --pipeline dev

  # With Audio (works with both pipelines)
  python -m fusion_mlx.video.ltx2.generate --prompt "Ocean waves crashing" --audio
  python -m fusion_mlx.video.ltx2.generate --prompt "A jazz band playing" --audio --pipeline dev
        """,
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Text description of the video to generate",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="distilled",
        choices=["distilled", "dev", "dev-two-stage", "dev-two-stage-hq"],
        help="Pipeline type: distilled (fast), dev (CFG), dev-two-stage (dev + LoRA), dev-two-stage-hq (res_2s + LoRA both stages)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for CFG (dev pipeline only)",
    )
    parser.add_argument(
        "--height", "-H", type=int, default=512, help="Output video height"
    )
    parser.add_argument(
        "--width", "-W", type=int, default=512, help="Output video width"
    )
    parser.add_argument(
        "--num-frames", "-n", type=int, default=33, help="Number of frames"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (dev pipeline only, default 30)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="CFG guidance scale for video (dev pipeline only, default 3.0)",
    )
    parser.add_argument(
        "--audio-cfg-scale",
        type=float,
        default=7.0,
        help="CFG guidance scale for audio (default 7.0, PyTorch default)",
    )
    parser.add_argument(
        "--cfg-rescale",
        type=float,
        default=0.7,
        help="CFG rescale factor (0.0-1.0, dev pipeline only, default 0.7)",
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument(
        "--output-path", "-o", type=str, default="output.mp4", help="Output video path"
    )
    parser.add_argument(
        "--save-frames", action="store_true", help="Save individual frames as images"
    )
    parser.add_argument(
        "--model-repo", type=str, default="Lightricks/LTX-2", help="Model repository"
    )
    parser.add_argument(
        "--text-encoder-repo", type=str, default=None, help="Text encoder repository"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--enhance-prompt", action="store_true", help="Enhance the prompt using Gemma"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max tokens for prompt enhancement"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for prompt enhancement",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to conditioning image for I2V",
    )
    parser.add_argument(
        "--image-strength",
        type=float,
        default=1.0,
        help="Conditioning strength for I2V",
    )
    parser.add_argument(
        "--image-frame-idx",
        type=int,
        default=0,
        help="Frame index to condition for I2V (ignored when --end-image is set)",
    )
    parser.add_argument(
        "--end-image",
        type=str,
        default=None,
        help="Path to conditioning image for the last frame (I2V end-frame control)",
    )
    parser.add_argument(
        "--end-image-strength",
        type=float,
        default=None,
        help="Conditioning strength for end frame (defaults to --image-strength)",
    )
    parser.add_argument(
        "--tiling",
        type=str,
        default="auto",
        choices=[
            "auto",
            "none",
            "default",
            "aggressive",
            "conservative",
            "spatial",
            "temporal",
        ],
        help="Tiling mode for VAE decoding",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream frames to output as they're decoded",
    )
    parser.add_argument(
        "--audio",
        "-a",
        action="store_true",
        help="Enable synchronized audio generation",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to audio file for A2V (audio-to-video) conditioning",
    )
    parser.add_argument(
        "--audio-start-time",
        type=float,
        default=0.0,
        help="Start time in seconds for audio file (default: 0.0)",
    )
    parser.add_argument(
        "--output-audio", type=str, default=None, help="Output audio path"
    )
    parser.add_argument(
        "--apg",
        action="store_true",
        help="Use Adaptive Projected Guidance instead of CFG (more stable for I2V)",
    )
    parser.add_argument(
        "--apg-eta",
        type=float,
        default=1.0,
        help="APG parallel component weight (1.0 = keep full parallel)",
    )
    parser.add_argument(
        "--apg-norm-threshold",
        type=float,
        default=0.0,
        help="APG guidance norm clamp (0 = no clamping)",
    )
    parser.add_argument(
        "--stg-scale",
        type=float,
        default=1.0,
        help="STG (Spatiotemporal Guidance) scale (default 1.0, 0.0 = disabled)",
    )
    parser.add_argument(
        "--stg-blocks",
        type=int,
        nargs="+",
        default=None,
        help="Transformer block indices for STG perturbation (default: [29] for LTX-2, [28] for LTX-2.3)",
    )
    parser.add_argument(
        "--modality-scale",
        type=float,
        default=3.0,
        help="Cross-modal guidance scale (default 3.0, 1.0 = disabled)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA safetensors file (dev-two-stage pipeline)",
    )
    parser.add_argument(
        "--lora-strength",
        type=float,
        default=1.0,
        help="LoRA merge strength (dev-two-stage pipeline, default 1.0)",
    )
    parser.add_argument(
        "--lora-strength-stage-1",
        type=float,
        default=0.25,
        help="LoRA strength for HQ stage 1 (default 0.25)",
    )
    parser.add_argument(
        "--lora-strength-stage-2",
        type=float,
        default=0.5,
        help="LoRA strength for HQ stage 2 (default 0.5)",
    )
    parser.add_argument(
        "--spatial-upscaler",
        type=str,
        default=None,
        help="Spatial upscaler filename (e.g. ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors). Auto-detects x2 by default.",
    )
    args = parser.parse_args()

    pipeline_map = {
        "distilled": PipelineType.DISTILLED,
        "dev": PipelineType.DEV,
        "dev-two-stage": PipelineType.DEV_TWO_STAGE,
        "dev-two-stage-hq": PipelineType.DEV_TWO_STAGE_HQ,
    }
    pipeline = pipeline_map[args.pipeline]

    generate_video(
        model_repo=args.model_repo,
        text_encoder_repo=args.text_encoder_repo,
        prompt=args.prompt,
        pipeline=pipeline,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        audio_cfg_scale=args.audio_cfg_scale,
        cfg_rescale=args.cfg_rescale,
        seed=args.seed,
        fps=args.fps,
        output_path=args.output_path,
        save_frames=args.save_frames,
        verbose=args.verbose,
        enhance_prompt=args.enhance_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image=args.image,
        image_strength=args.image_strength,
        image_frame_idx=args.image_frame_idx,
        end_image=args.end_image,
        end_image_strength=args.end_image_strength,
        tiling=args.tiling,
        stream=args.stream,
        audio=args.audio,
        output_audio_path=args.output_audio,
        use_apg=args.apg,
        apg_eta=args.apg_eta,
        apg_norm_threshold=args.apg_norm_threshold,
        stg_scale=args.stg_scale,
        stg_blocks=args.stg_blocks,
        modality_scale=args.modality_scale,
        lora_path=args.lora_path,
        lora_strength=args.lora_strength,
        lora_strength_stage_1=args.lora_strength_stage_1,
        lora_strength_stage_2=args.lora_strength_stage_2,
        audio_file=args.audio_file,
        audio_start_time=args.audio_start_time,
        spatial_upscaler=args.spatial_upscaler,
    )


if __name__ == "__main__":
    main()
