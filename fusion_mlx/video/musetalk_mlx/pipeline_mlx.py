"""MuseTalk MLX pipeline — single-step latent inpainting.

Core face-level path (parity-gated): crop 256² -> VAE.encode(masked)⊕encode(ref)=8ch
-> UNet(t=0, audio cross-attn) -> VAE.decode -> recon BGR face.

Face detect / crop / blend-paste-back are upstream CPU preprocessing (S3FD/DWPose/
face-parse) wired in by the caller; this module owns the neural generation.

Mirrors musetalk/models/vae.py (VAE wrapper) + scripts/inference.py inference loop.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

# cv2 is a runtime-only dep for face crop/blend I/O. Guard the import
# so the module loads without [video] extra (CI installs only [dev]).
try:
    import cv2
except ImportError:
    cv2 = None

from .config import RESIZED_IMG, UNET_TIMESTEP
from .models.unet import UNet2DConditionModel
from .models.vae import AutoencoderKL
from .utils.weights import (
    load_unet_weights,
    load_vae_weights,
    load_whisper_encoder_weights,
)
from .whisper.audio2feature import apply_pe, extract_prefix_tail, get_whisper_chunk
from .whisper.whisper_encoder import WhisperEncoder

logger = logging.getLogger(__name__)

_NORM_MEAN = 0.5
_NORM_STD = 0.5

# #920: the Metal allocator cache grows unbounded across a sustained realtime
# loop (13.55 GB cache / 1.97 GB active observed) and allocator scans over it
# produce multi-second render spikes. Capping the cache kills the spikes
# (p90 1224ms -> 304ms measured). memory_limit is NOT capped by default — the
# limit is process-global and a hard 3 GiB cap can OOM an LLM sharing the
# process; opt in via FUSION_MUSETALK_MLX_MEMORY_LIMIT.
_DEFAULT_CACHE_LIMIT = 1024**3


def tune_mlx_memory():
    """Apply default MLX allocator caps for sustained inference (#920).

    cache_limit defaults to 1 GiB; both knobs are env-overridable in bytes
    (FUSION_MUSETALK_MLX_CACHE_LIMIT / FUSION_MUSETALK_MLX_MEMORY_LIMIT,
    0 = leave unset). No-op on non-Metal builds (Linux CI).
    """
    if not hasattr(mx, "metal") or not hasattr(mx.metal, "set_cache_limit"):
        logger.debug("[musetalk] mx.metal unavailable — skipping allocator caps")
        return
    cache_limit = int(
        os.environ.get("FUSION_MUSETALK_MLX_CACHE_LIMIT", _DEFAULT_CACHE_LIMIT)
    )
    if cache_limit > 0:
        mx.metal.set_cache_limit(cache_limit)
        logger.info("[musetalk] MLX allocator cache_limit set to %d bytes", cache_limit)
    memory_limit = int(os.environ.get("FUSION_MUSETALK_MLX_MEMORY_LIMIT", "0"))
    if memory_limit > 0 and hasattr(mx.metal, "set_memory_limit"):
        mx.metal.set_memory_limit(memory_limit)
        logger.info(
            "[musetalk] MLX allocator memory_limit set to %d bytes", memory_limit
        )


def get_mask_tensor(size=RESIZED_IMG):
    m = np.zeros((size, size), dtype=np.float32)
    m[: size // 2, :] = 1.0  # keep upper half; lower (mouth) gets masked
    return m


def preprocess_img(img_bgr, half_mask=False, size=RESIZED_IMG):
    """BGR uint8 HxWx3 (already 256²) -> normalized NCHW float32 (1,3,256,256)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # HWC
    x = np.transpose(rgb, (2, 0, 1))  # CHW
    if half_mask:
        x = x * (get_mask_tensor(size) > 0.5)
    x = (x - _NORM_MEAN) / _NORM_STD
    return x[None]  # (1,3,256,256)


def _apply_smart_conv(vae) -> None:
    # Wrap VAE Conv2d with SmartConv2d (per-shape Metal kernel autotune, #919).
    # Default OFF: im2col GEMM accumulates in a different order than mx.conv2d,
    # so per-conv fp16 drift (~1e-4) compounds through the decoder (mid-block
    # attention + 3 upsample stages) into ~0.4 mean / 5.0 max pixel divergence
    # on a [-1,1] output — visible artifacts. VAE decode is already ~20ms/frame
    # on MLX 0.32.0 (the #919 58ms cliff was 0.32.2-specific), so the ~10%
    # speedup is not worth the quality regression on a visual output path.
    # Set FUSION_MUSETALK_SMART_CONV=1 to opt in (perf-critical, drift-tolerant).
    if os.environ.get("FUSION_MUSETALK_SMART_CONV", "0") != "1":
        return
    try:
        from fusion_mlx.graph_opt import apply_smart_conv

        n = apply_smart_conv(vae)
        logger.info("[musetalk] apply_smart_conv wrapped %d conv2d (VAE)", n)
    except Exception as exc:
        logger.warning("[musetalk] apply_smart_conv skipped: %s", exc)


class MuseTalkPipeline:
    def __init__(self, vae, unet, whisper_encoder=None, scaling_factor=None):
        self.vae = vae
        self.unet = unet
        self.whisper_encoder = whisper_encoder
        self.scaling_factor = (
            scaling_factor if scaling_factor is not None else vae.scaling_factor
        )

    def astype(self, dtype):
        """Cast all three nets to dtype (e.g. mx.float16 for realtime/inference)."""
        for m in (self.vae, self.unet, self.whisper_encoder):
            if m is not None:
                m.update(_tree_cast(m.parameters(), dtype))
                mx.eval(m.parameters())
        self._dtype = dtype
        return self

    @classmethod
    def from_pretrained(cls, weights_root: str | Path):
        tune_mlx_memory()
        root = Path(weights_root)
        vae = AutoencoderKL()
        load_vae_weights(vae, root / "sd-vae-ft-mse")
        vae.eval()
        _apply_smart_conv(vae)
        unet = UNet2DConditionModel()
        load_unet_weights(unet, root / "MuseTalk" / "musetalkV15" / "unet.pth")
        unet.eval()
        enc = WhisperEncoder()
        load_whisper_encoder_weights(enc, root / "whisper-tiny")
        enc.eval()
        return cls(vae, unet, enc)

    @classmethod
    def from_pretrained_mlx(cls, dist_dir: str | Path):
        """Load a published MLX variant (bf16 / q8 / q4) — torch-free, self-contained."""
        tune_mlx_memory()
        import json

        import mlx.nn as nn

        from .utils.weights import load_native

        dist_dir = Path(dist_dir)
        meta = json.loads((dist_dir / "config.json").read_text())
        vae = AutoencoderKL()
        load_native(vae, dist_dir / "vae.safetensors")
        vae.eval()
        _apply_smart_conv(vae)
        unet = UNet2DConditionModel()
        q = meta.get("quantization")
        if q:  # quantized UNet: apply nn.quantize before load
            nn.quantize(unet, group_size=q["group_size"], bits=q["bits"])
        load_native(unet, dist_dir / "unet.safetensors")
        unet.eval()
        enc = WhisperEncoder()
        load_native(enc, dist_dir / "whisper_encoder.safetensors")
        enc.eval()
        pipe = cls(vae, unet, enc, scaling_factor=meta.get("scaling_factor"))
        pipe._dtype = mx.float16 if meta.get("dtype") == "float16" else mx.bfloat16
        return pipe

    # ---- VAE wrapper (mirrors musetalk/models/vae.py) ----
    def get_latents_for_unet(self, crop_bgr, deterministic=True):
        """256² BGR face -> 8-ch latent (masked⊕ref). deterministic uses the posterior mean."""
        masked = mx.array(preprocess_img(crop_bgr, half_mask=True))
        ref = mx.array(preprocess_img(crop_bgr, half_mask=False))
        mp, rp = self.vae.encode(masked), self.vae.encode(ref)
        ml = self.scaling_factor * (mp.mean if deterministic else mp.sample())
        rl = self.scaling_factor * (rp.mean if deterministic else rp.sample())
        return mx.concatenate([ml, rl], axis=1)  # (1,8,32,32) NCHW

    def decode_latents(self, latents):
        """4-ch latent -> BGR uint8 (B,256,256,3), matching VAE.decode_latents."""
        img = self.vae.decode(latents / self.scaling_factor)  # NCHW [-1..1]-ish
        img = mx.clip(img / 2 + 0.5, 0, 1)
        img = np.array(
            img.transpose(0, 2, 3, 1).astype(mx.float32)
        )  # NHWC RGB (fp32 for numpy)
        img = (img * 255).round().astype(np.uint8)
        return img[..., ::-1]  # RGB -> BGR

    # ---- generation ----
    def generate_faces(self, latent_batch, audio_chunks):
        """latent_batch: (B,8,32,32) mx; audio_chunks: (B,50,384) mx -> recon BGR uint8 (B,256,256,3)."""
        audio = apply_pe(audio_chunks)
        pred = self.unet(latent_batch, mx.array([UNET_TIMESTEP]), audio)
        mx.eval(pred)
        return self.decode_latents(pred)

    def encode_audio(self, mel, librosa_length, fps=25, prefix=None):
        """mel (1,80,3000) -> per-frame cross-attn chunks (num_frames,50,384).

        #914 prefix-context cache: ``prefix`` (1, prefix_len, 5, 384) is the prior
        5s window's tail embedding spliced as left context to smooth the boundary
        between consecutive windows. Returns ``(chunks, tail)`` where ``tail`` is
        this window's stacked-state tail to cache for the next call. When
        ``prefix`` is None the tail is still returned (caller may ignore it) so
        the first window seeds the second without API branching.
        """
        stacked = self.whisper_encoder(mel)
        chunks = get_whisper_chunk(stacked, librosa_length, fps=fps, prefix=prefix)
        tail = extract_prefix_tail(stacked)
        return chunks, tail

    def encode_audio_from_wav(self, wav_path, fps=25, prefix=None):
        """Torch-free audio frontend: wav -> MLX log-mel -> whisper enc -> cross-attn chunks.

        Mirrors AudioProcessor.get_audio_feature + get_whisper_chunk with no torch/transformers.

        #914: returns ``(chunks, tail)``; pass ``tail`` as ``prefix`` on the next
        5s window to carry boundary context (smooths lip teleport at edges).
        """
        import librosa

        from .whisper.log_mel import N_SAMPLES, log_mel_spectrogram

        wav, _ = librosa.load(str(wav_path), sr=16000)
        segs = [wav[i : i + N_SAMPLES] for i in range(0, max(len(wav), 1), N_SAMPLES)]
        feats = [self.whisper_encoder(log_mel_spectrogram(mx.array(s))) for s in segs]
        stacked = mx.concatenate(feats, axis=1)  # (1, total_seq, 5, 384)
        chunks = get_whisper_chunk(stacked, len(wav), fps=fps, prefix=prefix)
        tail = extract_prefix_tail(stacked)
        return chunks, tail

    def run_batched(self, latent_stack, chunk_stack, batch_size=8):
        """Process N frames in batches (datagen-style). latent_stack:(N,8,32,32),
        chunk_stack:(N,50,384) -> recon BGR uint8 (N,256,256,3)."""
        n = latent_stack.shape[0]
        dtype = getattr(self, "_dtype", mx.float32)
        out = []
        for i in range(0, n, batch_size):
            lb = latent_stack[i : i + batch_size].astype(dtype)
            cb = chunk_stack[i : i + batch_size].astype(dtype)
            out.append(self.generate_faces(lb, cb))
        return np.concatenate(out, axis=0)


def _tree_cast(tree, dtype):
    if isinstance(tree, dict):
        return {k: _tree_cast(v, dtype) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_cast(v, dtype) for v in tree]
    return tree.astype(dtype)
