# SPDX-License-Identifier: Apache-2.0
# Legacy LTX-Video (0.9.x) pure-MLX video backend. Direct MLX port of the
# LTX-Video 0.9.x pipeline (ltx_video/, MIT licensed) - no mlx-video dependency
# for this family. T2V only (VAE encoder not ported). Upstream issue:
# https://github.com/Blaizzy/mlx-video/issues/43

import asyncio
import gc
import logging
import random
from typing import Any

import mlx.core as mx
import numpy as np

from ..._tempfile_safe import managed_tempfile_path
from ...engine_core import get_executor
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)

_DEFAULT_STEPS = 40
_DEFAULT_CFG = 3.0
_MAX_T5_LEN = 256


class LegacyLTXBackend(VideoBackend):
    name = "ltx_video_legacy"
    supports_i2v = False

    def __init__(self, model_name: str, *, dtype: Any = mx.bfloat16, **kwargs: Any):
        self._model_name = model_name
        self._dtype = dtype
        self._loaded = False
        self._transformer = None
        self._vae = None
        self._t5 = None
        self._tokenizer = None
        self._scheduler = None

    @classmethod
    def detect(cls, model_path: str) -> bool:
        p = model_path.lower()
        return "ltx-video" in p or "ltx_video" in p

    async def start(self, model_path: str, **kwargs: Any) -> None:
        if self._loaded:
            return
        logger.info("Starting legacy LTX-Video backend (pure-MLX): %s", model_path)
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), self._load_pipeline, model_path),
            timeout=180.0,
        )
        self._loaded = True
        logger.info("Legacy LTX-Video backend ready: %s", model_path)

    def _load_pipeline(self, model_path: str) -> None:
        from ...video.ltx_video_legacy.scheduler import RectifiedFlowScheduler
        from ...video.ltx_video_legacy.transformer import Transformer3DModel
        from ...video.ltx_video_legacy.vae import LTVideoVAE
        from ...video.t5_encoder import load_t5_encoder, load_t5_tokenizer

        local = _resolve_repo(model_path)
        t_dir, v_dir, e_dir = _component_dirs(local)
        self._transformer = Transformer3DModel.from_pretrained(t_dir, dtype=self._dtype)
        self._vae = LTVideoVAE.from_pretrained(v_dir, dtype=self._dtype)
        self._t5 = load_t5_encoder(e_dir, dtype=self._dtype)
        self._tokenizer = load_t5_tokenizer(e_dir)
        self._scheduler = RectifiedFlowScheduler()
        logger.info(
            "legacy-ltx: loaded transformer=%s vae=%s t5=%s",
            type(self._transformer).__name__,
            type(self._vae).__name__,
            type(self._t5).__name__,
        )

    async def stop(self) -> None:
        if not self._loaded:
            return
        self._loaded = False
        self._transformer = None
        self._vae = None
        self._t5 = None
        self._tokenizer = None
        self._scheduler = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )

    async def generate(self, params: VideoGenParams) -> list[bytes]:
        base_seed = (
            params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        )

        def _generate():
            results: list[bytes] = []
            for i in range(max(1, params.n)):
                mp4_bytes = _generate_one(
                    self._transformer,
                    self._vae,
                    self._t5,
                    self._tokenizer,
                    self._scheduler,
                    self._dtype,
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    num_frames=params.num_frames,
                    width=params.width,
                    height=params.height,
                    fps=params.fps,
                    seed=base_seed + i,
                    num_inference_steps=params.num_inference_steps,
                    cfg_scale=params.cfg_scale,
                )
                results.append(mp4_bytes)
            return results

        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _generate), timeout=600.0
        )

    def constraints(self) -> VideoConstraints:
        return VideoConstraints(
            supports_i2v=False,
            max_n=4,
            dim_divisibility=32,
            num_frames_validator=lambda nf: nf % 8 == 1,
            num_frames_hint="num_frames must satisfy num_frames % 8 == 1",
            dim_hint="width and height must be divisible by 32",
        )


def _resolve_repo(model_path: str) -> str:
    # Local dir wins; otherwise snapshot_download the HF repo. The 0.9.x weight
    # layout is finalized during the real-E2E smoke (Task #13); this resolver is
    # the single seam to adjust.
    from pathlib import Path

    local = Path(model_path)
    if local.exists() and local.is_dir():
        return str(local)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            f"legacy-ltx: huggingface_hub required to resolve {model_path}"
        ) from exc
    return snapshot_download(repo_id=model_path)


def _component_dirs(local: str) -> tuple[str, str, str]:
    from pathlib import Path

    root = Path(local)
    candidates = {
        "t": [root / "transformer", root / "unet"],
        "v": [root / "vae"],
        "e": [root / "text_encoder", root / "t5"],
    }
    out = []
    for key in ("t", "v", "e"):
        picked = next((c for c in candidates[key] if c.exists()), root)
        out.append(str(picked))
    return tuple(out)  # type: ignore[return-value]


def _vae_scale_factors(vae) -> tuple[int, int, int]:
    # Mirror causal_video_autoencoder: spatial = 2^(compress blocks) * patch_size,
    # temporal = 2^(compress blocks) (no patch multiplier). compress_all hits both.
    spatial = 1
    temporal = 1
    for b in vae.config.blocks:
        name = b.get("name", "") if isinstance(b, dict) else getattr(b, "name", "")
        if "space" in name or "all" in name:
            spatial *= 2
        if "time" in name or "all" in name:
            temporal *= 2
    spatial *= vae.config.patch_size
    return temporal, spatial, spatial


def _encode_prompt(t5, tokenizer, prompt: str):
    # Call T5 __call__ directly so the attention mask is retained (encode()
    # discards it). Tokenization mirrors T5Encoder.encode.
    enc = tokenizer(
        prompt if prompt else "",
        padding="max_length",
        truncation=True,
        max_length=_MAX_T5_LEN,
        add_special_tokens=True,
        return_tensors="np",
    )
    input_ids = mx.array(np.asarray(enc["input_ids"], dtype=np.int32))
    attn = mx.array(np.asarray(enc["attention_mask"], dtype=np.int32))
    embeds = t5(input_ids, attn)
    mx.eval(embeds)
    return embeds, attn


def _latent_pixel_coords(lf: int, lh: int, lw: int, scale_factors):
    # SymmetricPatchifier.get_latent_coords (patch_size=1) -> latent_to_pixel_coords.
    # meshgrid (f, h, w) with ij indexing, stack -> (3, lf*lh*lw), scale by VAE factors.
    f = mx.arange(lf, dtype=mx.float32)
    h = mx.arange(lh, dtype=mx.float32)
    w = mx.arange(lw, dtype=mx.float32)
    ff, hh, ww = mx.meshgrid(f, h, w, indexing="ij")
    coords = mx.stack([ff, hh, ww], axis=0).reshape(3, lf * lh * lw)
    coords = coords[None]  # (1, 3, n)
    scales = mx.array(scale_factors, dtype=mx.float32)
    return coords * scales[None, :, None]


def _generate_one(
    transformer,
    vae,
    t5,
    tokenizer,
    scheduler,
    dtype,
    *,
    prompt: str,
    negative_prompt: str | None,
    num_frames: int,
    width: int,
    height: int,
    fps: int,
    seed: int,
    num_inference_steps: int | None,
    cfg_scale: float | None,
) -> bytes:
    from ...video.ltx_video_legacy.denoise import denoise

    steps = int(num_inference_steps) if num_inference_steps else _DEFAULT_STEPS
    cfg = float(cfg_scale) if cfg_scale is not None else _DEFAULT_CFG
    scale_factors = _vae_scale_factors(vae)
    temporal_s, spatial_s, _ = scale_factors
    lf = num_frames // temporal_s + 1
    lh = height // spatial_s
    lw = width // spatial_s
    latent_shape = (1, transformer.cfg.in_channels, lf, lh, lw)
    n_tokens = lf * lh * lw

    mx.random.seed(seed)
    # noise (1, c, lf, lh, lw) -> patchify (patch_size=1) -> (1, n, c)
    noise = (
        mx.random.normal(shape=latent_shape, dtype=mx.float32)
        * scheduler.init_noise_sigma
    )
    latents = mx.transpose(noise, (0, 2, 3, 4, 1)).reshape(1, n_tokens, latent_shape[1])

    pixel_coords = _latent_pixel_coords(lf, lh, lw, scale_factors)

    prompt_embeds, prompt_mask = _encode_prompt(t5, tokenizer, prompt)
    if negative_prompt:
        neg_embeds, neg_mask = _encode_prompt(t5, tokenizer, negative_prompt)
    else:
        neg_embeds = mx.zeros_like(prompt_embeds)
        neg_mask = mx.zeros_like(prompt_mask)

    logger.info(
        "legacy-ltx generate: prompt_len=%d frames=%d %dx%d@%dfps seed=%d "
        "steps=%d cfg=%.2f latent=%s",
        len(prompt),
        num_frames,
        width,
        height,
        fps,
        seed,
        steps,
        cfg,
        latent_shape,
    )

    latents = denoise(
        transformer,
        scheduler,
        latents,
        pixel_coords,
        prompt_embeds,
        prompt_mask,
        neg_embeds,
        neg_mask,
        cfg,
        steps,
        float(fps),
        latent_shape,
        dtype=dtype,
    )

    # unpatchify (1, n, c) -> (1, c, lf, lh, lw) then VAE decode.
    latents_5d = latents.reshape(1, lf, lh, lw, latent_shape[1])
    latents_5d = mx.transpose(latents_5d, (0, 4, 1, 2, 3)).astype(dtype)
    target_shape = (1, 3, lf * temporal_s, lh * spatial_s, lw * spatial_s)
    decoded = vae.decode(latents_5d, target_shape=target_shape)
    mx.eval(decoded)
    decoded = np.asarray(decoded, dtype=np.float32)

    # NCDHW (1, 3, F, H, W) -> crop to requested dims -> list of HxWx3 uint8 frames.
    f_out = min(decoded.shape[2], num_frames)
    h_out = min(decoded.shape[3], height)
    w_out = min(decoded.shape[4], width)
    frames = []
    for fi in range(f_out):
        frame = decoded[0, :, fi, :h_out, :w_out]  # (3, H, W)
        frame = np.clip(frame, 0.0, 1.0)
        frame = np.transpose(frame, (1, 2, 0))  # (H, W, 3)
        frame = (frame * 255.0).astype(np.uint8)
        frames.append(frame)

    with managed_tempfile_path(prefix="fusion_video_", suffix=".mp4") as handle:
        _write_mp4(frames, fps, handle.path)
        with open(handle.path, "rb") as f:
            return f.read()


def _write_mp4(frames: list[np.ndarray], fps: int, path: str) -> None:
    import imageio

    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)
    logger.info(
        "legacy-ltx: wrote mp4 frames=%d fps=%d path=%s", len(frames), fps, path
    )
