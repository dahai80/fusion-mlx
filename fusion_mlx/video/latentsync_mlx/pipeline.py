"""LatentSync MLX inference pipeline — pure MLX, zero PyTorch.

Face detection uses insightface (CPU), Whisper audio encoding uses the
MuseTalk-MLX WhisperEncoder (pure MLX). UNet denoising + VAE encode/decode
are pure MLX. Only numpy/cv2/soundfile/librosa for I/O.
"""
import gc
import os
import math
import shutil
import subprocess
import logging

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import cv2
import tqdm

from .unet import UNet3DConditionModel
from .vae import Autoencoder
from .sampler import DDIMSampler
from ..musetalk_mlx.whisper.whisper_encoder import WhisperEncoder
from ..musetalk_mlx.whisper.log_mel import log_mel_spectrogram, N_SAMPLES
from ..musetalk_mlx.whisper.audio2feature import apply_pe, get_whisper_chunk

logger = logging.getLogger(__name__)


def write_video_frames(path: str, frames: np.ndarray, fps: int = 25):
    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def read_video_cv2(video_path: str, fps: int = 25):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)


def read_audio_librosa(audio_path: str, sr: int = 16000):
    import librosa
    wav, _ = librosa.load(audio_path, sr=sr)
    return wav


def _affine_transform_frame(frame, face_box, size=256):
    x1, y1, x2, y2 = face_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    scale = min(size / (x2 - x1), size / (y2 - y1)) * 0.9
    M = np.array([
        [scale, 0, size / 2 - cx * scale],
        [0, scale, size / 2 - cy * scale],
    ])
    warped = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR)
    return warped, M


def _prepare_masks_and_images(faces_np, half="lower"):
    """Pure numpy/cv2 mask preparation — replaces PyTorch ImageProcessor.

    faces_np: (F, H, W, 3) uint8 RGB
    Returns: ref_pv, masked_pv, masks — all (F, H, W, C) float32 NHWC [-1, 1]
    """
    h, w = faces_np.shape[1], faces_np.shape[2]
    ref_list, masked_list, mask_list = [], [], []

    for frame in faces_np:
        ref = frame.astype(np.float32) / 255.0
        ref = (ref - 0.5) / 0.5

        mask = np.zeros((h, w, 1), dtype=np.float32)
        if half == "lower":
            mask[h // 2:, :, :] = 1.0
        else:
            mask[:h // 2, :, :] = 1.0

        masked = ref.copy()
        masked = masked * (1.0 - mask) + (-1.0) * mask

        ref_list.append(ref)
        masked_list.append(masked)
        mask_list.append(mask)

    return (
        np.stack(ref_list),
        np.stack(masked_list),
        np.stack(mask_list),
    )


def _resize_mask_batch(masks, target_h, target_w):
    """Resize masks from (F, H, W, 1) to (F, target_h, target_w, 1)."""
    out = []
    for m in masks:
        if m.ndim == 3 and m.shape[-1] == 1:
            m2d = m[:, :, 0]
        else:
            m2d = m
        resized = cv2.resize(m2d, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        out.append(resized[..., np.newaxis])
    return np.stack(out)


def _restore_face_to_frame(synced_face, original_frame, face_box, affine_matrix, size=256):
    """Inverse affine paste-back: place lip-synced face back into original frame."""
    x1, y1, x2, y2 = [int(v) for v in face_box]

    face_rgb = ((synced_face + 1.0) / 2.0 * 255.0)
    face_rgb = np.clip(face_rgb, 0, 255).astype(np.uint8)

    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return original_frame
    face_resized = cv2.resize(face_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = np.zeros((size, size), dtype=np.float32)
    mask[size // 2:, :] = 1.0
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    frame = original_frame.copy()
    if y2 > frame.shape[0] or x2 > frame.shape[1]:
        return frame
    roi = frame[y1:y2, x1:x2]
    blended = (face_resized.astype(np.float32) * mask_resized[..., np.newaxis] +
               roi.astype(np.float32) * (1.0 - mask_resized[..., np.newaxis]))
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame


def _detect_faces_insightface(frames, ctx_id=-1, det_size=(640, 640)):
    """Detect faces using insightface (CPU-only, no torch)."""
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError("insightface required: pip install insightface")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=ctx_id, det_size=det_size)

    all_boxes, all_affines, all_cropped = [], [], []
    for frame in frames:
        faces = app.get(frame)
        if len(faces) == 0:
            h, w = frame.shape[:2]
            box = [0, 0, w, h]
            M = np.array([[w / 256, 0, 0], [0, h / 256, 0]])
            crop = cv2.resize(frame, (256, 256))
        else:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            box = face.bbox.tolist()
            crop, M = _affine_transform_frame(frame, box, size=256)

        all_boxes.append(box)
        all_affines.append(M)
        all_cropped.append(crop)

    return all_cropped, all_boxes, all_affines


class LipsyncPipelineMLX:
    """Pure MLX lip-sync pipeline. Zero PyTorch dependency."""

    def __init__(
        self,
        unet: UNet3DConditionModel,
        vae: Autoencoder,
        sampler: DDIMSampler,
        whisper_encoder: WhisperEncoder,
        resolution: int = 256,
        dtype=None,
    ):
        self.unet = unet
        self.vae = vae
        self.sampler = sampler
        self.whisper_encoder = whisper_encoder
        self.resolution = resolution
        self.vae_scale_factor = 8
        self.dtype = dtype or mx.float16

    @classmethod
    def from_pretrained(cls, model_dir: str, dtype=mx.float16):
        """Load all components from a directory with safetensors weights."""
        model_dir = os.path.expanduser(model_dir)

        unet = UNet3DConditionModel()
        unet_path = os.path.join(model_dir, "unet.safetensors")
        if os.path.exists(unet_path):
            weights = mx.load(unet_path)
            unet.update(weights)
            logger.info(f"Loaded UNet from {unet_path}")
        unet.eval()

        vae = Autoencoder()
        vae_path = os.path.join(model_dir, "vae.safetensors")
        if os.path.exists(vae_path):
            weights = mx.load(vae_path)
            vae.update(weights)
            logger.info(f"Loaded VAE from {vae_path}")
        vae.eval()

        sampler = DDIMSampler()

        whisper_enc = WhisperEncoder()
        whisper_path = os.path.join(model_dir, "whisper_encoder.safetensors")
        if os.path.exists(whisper_path):
            weights = mx.load(whisper_path)
            whisper_enc.update(weights)
            logger.info(f"Loaded WhisperEncoder from {whisper_path}")
        whisper_enc.eval()

        pipe = cls(unet, vae, sampler, whisper_enc, dtype=dtype)
        pipe._cast_all(dtype)
        return pipe

    def _cast_all(self, dtype):
        for m in (self.unet, self.vae, self.whisper_encoder):
            if m is not None:
                m.update(m.parameters())
                mx.eval(m.parameters())
        self.dtype = dtype

    def encode_audio(self, audio_path: str, fps: int = 25):
        """Pure MLX audio encoding: wav -> log-mel -> whisper -> per-frame chunks."""
        import librosa

        wav, sr = librosa.load(audio_path, sr=16000)
        segs = [wav[i:i + N_SAMPLES] for i in range(0, max(len(wav), 1), N_SAMPLES)]
        feats = [self.whisper_encoder(log_mel_spectrogram(mx.array(s))) for s in segs]
        stacked = mx.concatenate(feats, axis=1)
        chunks = get_whisper_chunk(stacked, len(wav), fps=fps)
        return chunks, wav, sr

    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        seed: int = 1247,
        temp_dir: str = "temp_mlx",
    ):
        import soundfile as sf

        logger.info("=== LatentSync MLX Pipeline (pure MLX) ===")

        # --- Stage 1: Audio encoding (pure MLX) ---
        logger.info("Encoding audio with MLX Whisper...")
        whisper_chunks, audio_samples, _ = self.encode_audio(audio_path, fps=video_fps)

        del self.whisper_encoder
        self.whisper_encoder = None
        gc.collect()
        mx.clear_cache()
        logger.info("Freed whisper encoder.")

        # --- Stage 2: Video/face processing (insightface + numpy/cv2) ---
        logger.info("Processing video frames...")
        video_frames = read_video_cv2(video_path, fps=video_fps)

        cropped_faces, boxes, affine_matrices = _detect_faces_insightface(video_frames)

        video_frames, cropped_faces, boxes, affine_matrices = self._loop_video(
            whisper_chunks, video_frames, cropped_faces, boxes, affine_matrices
        )

        # --- Stage 3: MLX denoising ---
        logger.info(f"Running MLX denoising ({num_inference_steps} steps, {len(whisper_chunks)} frames)...")

        mx.random.seed(seed)
        self.sampler.set_timesteps(num_inference_steps)

        do_cfg = guidance_scale > 1.0
        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        latent_h = self.resolution // self.vae_scale_factor
        latent_w = self.resolution // self.vae_scale_factor

        single_noise = mx.random.normal((1, 1, latent_h, latent_w, 4)).astype(self.dtype)
        noise_shape = (1, len(whisper_chunks), latent_h, latent_w, 4)
        all_latents = mx.broadcast_to(single_noise, noise_shape) * self.sampler.init_noise_sigma

        synced_frames_list = []

        for chunk_idx in tqdm.tqdm(range(num_inferences), desc="MLX inference"):
            start = chunk_idx * num_frames
            end = min(start + num_frames, len(whisper_chunks))
            chunk_faces = cropped_faces[start:end]

            # Prepare masks and images (pure numpy)
            ref_pv, masked_pv, masks_np = _prepare_masks_and_images(
                np.stack(chunk_faces)
            )

            # Convert to MLX NHWC
            ref_pv_mlx = mx.array(ref_pv).astype(self.dtype)
            masked_pv_mlx = mx.array(masked_pv).astype(self.dtype)

            # VAE encode
            masked_latents = self._vae_encode(masked_pv_mlx)
            ref_latents = self._vae_encode(ref_pv_mlx)

            # Resize mask to latent space
            mask_latents_np = _resize_mask_batch(masks_np, latent_h, latent_w)
            mask_latents = mx.array(mask_latents_np).astype(self.dtype)

            F_count = masked_latents.shape[0]
            masked_latents = masked_latents[None]
            ref_latents = ref_latents[None]
            mask_latents = mask_latents[None]

            # CFG doubling
            if do_cfg:
                masked_latents = mx.concatenate([masked_latents] * 2, axis=0)
                ref_latents = mx.concatenate([ref_latents] * 2, axis=0)
                mask_latents = mx.concatenate([mask_latents] * 2, axis=0)

            # Audio embeddings (pure MLX)
            audio_embeds_mlx = whisper_chunks[start:end].astype(self.dtype)
            audio_embeds_mlx = audio_embeds_mlx[None]
            if do_cfg:
                null_audio = mx.zeros_like(audio_embeds_mlx)
                audio_embeds_mlx = mx.concatenate([null_audio, audio_embeds_mlx], axis=0)

            latents = all_latents[:, start:end]

            # Denoising loop
            for t in self.sampler.timesteps:
                latent_input = mx.concatenate([latents] * 2, axis=0) if do_cfg else latents
                latent_input = self.sampler.scale_model_input(latent_input, t)

                unet_input = mx.concatenate(
                    [latent_input, mask_latents, masked_latents, ref_latents], axis=-1
                )

                timestep_mx = mx.array([t])
                noise_pred = self.unet(
                    unet_input, timestep_mx, encoder_hidden_states=audio_embeds_mlx
                )

                if do_cfg:
                    pred_uncond, pred_audio = mx.split(noise_pred, 2, axis=0)
                    noise_pred = pred_uncond + guidance_scale * (pred_audio - pred_uncond)

                latents = self.sampler.step(noise_pred, t, latents)
                mx.eval(latents)

            # VAE decode
            decoded = self._vae_decode(latents.reshape(-1, latent_h, latent_w, 4))

            # Compositing
            inv_mask_mlx = mx.array(1.0 - masks_np).astype(self.dtype)
            masks_mlx_mlx = mx.array(masks_np).astype(self.dtype)
            combined = decoded[:F_count] * inv_mask_mlx[:F_count] + ref_pv_mlx[:F_count] * masks_mlx_mlx[:F_count]
            synced_frames_list.append(combined)

        del self.unet
        self.unet = None
        gc.collect()
        mx.clear_cache()
        logger.info("Freed UNet and cleared MLX cache.")

        # --- Stage 4: Restore faces to original frames ---
        logger.info("Restoring faces to video...")
        all_synced = mx.concatenate(synced_frames_list, axis=0)
        all_synced_np = np.array(all_synced.astype(mx.float32))

        del self.vae
        self.vae = None
        gc.collect()
        mx.clear_cache()
        logger.info("Freed VAE and cleared MLX cache.")

        restored = self._restore_video(all_synced_np, video_frames, boxes, affine_matrices)

        # --- Stage 5: Write output ---
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        video_temp = os.path.join(temp_dir, "video.mp4")
        audio_temp = os.path.join(temp_dir, "audio.wav")

        write_video_frames(video_temp, restored, fps=video_fps)

        audio_remain_len = int(restored.shape[0] / video_fps * audio_sample_rate)
        audio_np = audio_samples[:audio_remain_len]
        sf.write(audio_temp, audio_np, audio_sample_rate)

        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
            "-i", video_temp, "-i", audio_temp,
            "-c:v", "libx264", "-crf", "18",
            "-c:a", "aac", "-q:v", "0", "-q:a", "0",
            video_out_path,
        ])
        logger.info(f"Output saved to: {video_out_path}")

    def _loop_video(self, whisper_chunks, video_frames, cropped_faces, boxes, affines):
        if len(whisper_chunks) > len(video_frames):
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_vf, loop_faces, loop_boxes, loop_aff = [], [], [], []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_vf.append(video_frames)
                    loop_faces.append(cropped_faces)
                    loop_boxes += boxes
                    loop_aff += affines
                else:
                    loop_vf.append(video_frames[::-1])
                    loop_faces.append(cropped_faces[::-1])
                    loop_boxes += boxes[::-1]
                    loop_aff += affines[::-1]
            video_frames = np.concatenate(loop_vf)[:len(whisper_chunks)]
            cropped_faces = np.concatenate(loop_faces)[:len(whisper_chunks)]
            boxes = loop_boxes[:len(whisper_chunks)]
            affines = loop_aff[:len(whisper_chunks)]
        else:
            video_frames = video_frames[:len(whisper_chunks)]
            cropped_faces = cropped_faces[:len(whisper_chunks)]
            boxes = boxes[:len(whisper_chunks)]
            affines = affines[:len(whisper_chunks)]
        return video_frames, cropped_faces, boxes, affines

    def _vae_encode(self, images, batch_size=1):
        latents = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]
            mean, _ = self.vae.encode(batch)
            latents.append((mean - self.vae.shift_factor) * self.vae.scaling_factor)
            mx.eval(latents[-1])
        return mx.concatenate(latents, axis=0)

    def _vae_decode(self, latents, batch_size=1):
        scaled = latents / self.vae.scaling_factor + self.vae.shift_factor
        decoded = []
        for i in range(0, scaled.shape[0], batch_size):
            batch = scaled[i:i + batch_size]
            decoded.append(self.vae.decode(batch))
            mx.eval(decoded[-1])
        return mx.concatenate(decoded, axis=0)

    def _restore_video(self, synced_faces_np, video_frames, boxes, affine_matrices):
        """Pure numpy/cv2 face restoration — no PyTorch."""
        video_frames = video_frames[:len(synced_faces_np)]
        out_frames = []
        logger.info(f"Restoring {len(synced_faces_np)} faces...")
        for idx, face in enumerate(tqdm.tqdm(synced_faces_np)):
            frame = _restore_face_to_frame(
                face, video_frames[idx], boxes[idx], affine_matrices[idx], size=self.resolution
            )
            out_frames.append(frame)
        return np.stack(out_frames)
