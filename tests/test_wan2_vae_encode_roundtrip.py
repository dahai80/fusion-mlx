import asyncio
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# Default is Wan2.1-14B, NOT 1.3B: the 1.3B `vae.safetensors` checkpoint is
# decoder-only (0 encoder weight keys), so its VAE encode runs on random
# conv weights and destroys spatial structure (encode→decode corr ≈ 0). 14B
# ships the full encoder (84 keys) and round-trips a structured gradient at
# corr 0.998+. Override with FUSION_653_WAN2_MODEL for a different model.
REAL_MODEL_DIR = Path(
    os.environ.get(
        "FUSION_653_WAN2_MODEL",
        str(Path.home() / ".fusion-mlx/models/Wan2.1-14B"),
    )
)

pytestmark = pytest.mark.real_model


def _skip_unless_real_model():
    if not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"):
        pytest.skip(
            "set FUSION_MLX_REAL_MODEL_TESTS=1 to run real-model wan2 vae roundtrip"
        )
    if not REAL_MODEL_DIR.exists():
        pytest.skip(f"wan2 model not installed at {REAL_MODEL_DIR}")


def _structured_pixels(t: int, h: int = 256, w: int = 256) -> mx.array:
    # Deterministic spatial+temporal gradient. A scrambled channel/temporal
    # layout in encode→decode destroys the gradient correlation; a correct
    # roundtrip preserves it. zeros-in→zeros-out passes for the wrong
    # reason, so this is the actual guard for the #458 streaming regression.
    ys = mx.arange(h, dtype=mx.float32) / h
    xs = mx.arange(w, dtype=mx.float32) / w
    ts = mx.arange(t, dtype=mx.float32) / max(t - 1, 1)
    frame = (ys[:, None] + xs[None, :]) * 0.5
    frames = []
    for i in range(t):
        chan_r = frame + 0.1 * ts[i]
        chan_g = frame * 0.7
        chan_b = mx.broadcast_to(ts[i], (h, w))
        frames.append(mx.stack([chan_r, chan_g, chan_b], axis=-1))
    return mx.stack(frames, axis=0)[None]


def _pixel_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def test_vae_encode_decode_roundtrip_5frame_streaming():
    # 5 frames = 1+2+2 streaming chunks (iter_=3, odd). Guards the #458
    # streaming-cache regression through the encode path. 1-frame is
    # degenerate (encoder temporal-downsample needs >=kt frames) and 7-frame
    # hits an even-iter flush bug (issue #669), so neither is a valid
    # roundtrip fixture here.
    _skip_unless_real_model()
    from fusion_mlx.public_api import VideoGenEngine

    async def _run():
        eng = VideoGenEngine(str(REAL_MODEL_DIR))
        await eng.start()
        await eng.load_vae()
        pixels = _structured_pixels(5, 256, 256)
        lat = await eng.encode(pixels)
        assert lat.ndim == 5
        assert lat.shape[2] >= 1, lat.shape
        assert not np.any(np.isnan(np.array(lat))), "encode produced NaN latent"
        out = await eng.decode(lat)
        await eng.unload_vae()
        await eng.stop()
        return pixels, out

    pixels, out = asyncio.run(_run())
    src = np.array(pixels)
    arr = np.array(out)
    assert not np.any(np.isnan(arr)), "decode produced NaN pixels"
    corr = _pixel_corr(src, arr)
    print(f"\n5-frame streaming roundtrip pixel_corr={corr:.4f} out_shape={arr.shape}")
    assert (
        corr >= 0.9
    ), f"5-frame streaming roundtrip pixel_corr {corr:.4f} < 0.9 (#458 regression)"


def test_vae_encode_decode_roundtrip_17frame_streaming():
    # 17 frames = 1+2*8 (iter_=9, odd), the standard wan generation count
    # (1+4N). Exercises the full multi-chunk streaming encode path.
    _skip_unless_real_model()
    from fusion_mlx.public_api import VideoGenEngine

    async def _run():
        eng = VideoGenEngine(str(REAL_MODEL_DIR))
        await eng.start()
        await eng.load_vae()
        pixels = _structured_pixels(17, 256, 256)
        lat = await eng.encode(pixels)
        assert lat.ndim == 5
        assert lat.shape[2] >= 1, lat.shape
        assert not np.any(np.isnan(np.array(lat))), "encode produced NaN latent"
        out = await eng.decode(lat)
        await eng.unload_vae()
        await eng.stop()
        return pixels, out

    pixels, out = asyncio.run(_run())
    src = np.array(pixels)
    arr = np.array(out)
    assert not np.any(np.isnan(arr)), "decode produced NaN pixels"
    corr = _pixel_corr(src, arr)
    print(f"\n17-frame streaming roundtrip pixel_corr={corr:.4f} out_shape={arr.shape}")
    assert (
        corr >= 0.9
    ), f"17-frame streaming roundtrip pixel_corr {corr:.4f} < 0.9 (scrambled layout)"
