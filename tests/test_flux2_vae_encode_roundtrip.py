import asyncio
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# FLUX.2-klein-base-4B ships the full VAE encoder (106 keys in
# diffusion_pytorch_model.safetensors), unlike the wan2 1.3B decoder-only VAE.
# Override with FUSION_653_FLUX2_MODEL for a different model.
REAL_MODEL_DIR = Path(
    os.environ.get(
        "FUSION_653_FLUX2_MODEL",
        str(Path.home() / ".fusion-mlx/models/FLUX.2-klein-base-4B"),
    )
)

pytestmark = pytest.mark.real_model


def _skip_unless_real_model():
    if not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"):
        pytest.skip(
            "set FUSION_MLX_REAL_MODEL_TESTS=1 to run real-model flux2 vae roundtrip"
        )
    if not REAL_MODEL_DIR.exists():
        pytest.skip(f"flux2 model not installed at {REAL_MODEL_DIR}")


def _structured_image(h: int = 256, w: int = 256) -> mx.array:
    # Deterministic spatial gradient in [0,1] (mflux contract: pixels [0,1],
    # image_util.py:131 np.array/255). NHWC (1,H,W,3) per the encode spec
    # contract. A scrambled channel/spatial layout in encode→decode destroys
    # the gradient correlation; a correct roundtrip preserves it.
    # zeros-in→zeros-out passes for the wrong reason, so this is the real
    # guard for the patchify/bn-norm symmetry (issue #653 load-bearing).
    ys = mx.arange(h, dtype=mx.float32) / h
    xs = mx.arange(w, dtype=mx.float32) / w
    frame = (ys[:, None] + xs[None, :]) * 0.5
    chan_r = frame
    chan_g = frame * 0.7
    chan_b = mx.broadcast_to(mx.array(0.3), (h, w))
    img = mx.stack([chan_r, chan_g, chan_b], axis=-1)
    return img[None]


def _to_np(x: mx.array) -> np.ndarray:
    # flux2 VAE runs in bfloat16; numpy's buffer protocol cannot consume bf16
    # ("Item size 2 ... dtype B item size 1"). Cast to float32 in MLX first,
    # then bridge to numpy. Same bridge any numpy/PIL caller downstream needs.
    return np.array(mx.array(x, mx.float32))


def _pixel_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def test_flux2_vae_encode_decode_roundtrip():
    # Load-bearing symmetry test: encode applies VAEUtil.encode→
    # ensure_4d→crop_to_even→patchify→bn_normalize; decode_packed_latents
    # inverts un-bn-norm→unpatchify→vae.decode. If encode skips patchify or
    # bn-norm, or feeds the wrong layout to the NCHW encoder, the roundtrip
    # correlation collapses.
    _skip_unless_real_model()
    from fusion_mlx.public_api import ImageGenEngine

    async def _run():
        eng = ImageGenEngine(str(REAL_MODEL_DIR))
        await eng.start()
        await eng.load_vae()
        pixels = _structured_image(256, 256)
        lat = await eng.encode(pixels)
        assert lat.ndim == 4, lat.shape
        assert lat.shape[1] == 128, lat.shape
        assert not np.any(np.isnan(_to_np(lat))), "encode produced NaN latent"
        out = await eng.decode(lat)
        await eng.unload_vae()
        await eng.stop()
        return pixels, out

    pixels, out = asyncio.run(_run())
    src = _to_np(pixels)
    # decode returns NCHW (1,3,H,W) — mflux vae.decode native layout. Bring
    # to NHWC (1,H,W,3) to match encode's public input contract for a
    # layout-matched pixel correlation.
    arr = _to_np(out).transpose(0, 2, 3, 1)
    assert not np.any(np.isnan(arr)), "decode produced NaN pixels"
    corr = _pixel_corr(src, arr)
    print(f"\nflux2 roundtrip pixel_corr={corr:.4f} out_shape={arr.shape}")
    assert corr >= 0.9, (
        f"flux2 roundtrip pixel_corr {corr:.4f} < 0.9 "
        "(patchify/bn-norm symmetry broken or NHWC/NCHW layout mismatch)"
    )
