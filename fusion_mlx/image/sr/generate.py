import logging
import math
import os

import mlx.core as mx
import numpy as np

from fusion_mlx.image.sr.config import RealESRGANConfig
from fusion_mlx.image.sr.weights import load_sr_model

logger = logging.getLogger(__name__)

_NET_CACHE: dict = {}
_TEST_NET = None


def _set_net_for_test(net, scale):
    global _TEST_NET
    _TEST_NET = (net, scale)


def _get_net(model_path, scale, config):
    if _TEST_NET is not None and _TEST_NET[1] == scale:
        return _TEST_NET[0]
    key = (model_path, scale)
    if key not in _NET_CACHE:
        cfg = config or RealESRGANConfig(scale=scale)
        path = model_path or os.path.expanduser(
            "~/.fusion-mlx/models/realesrgan/RealESRGAN_x4plus.safetensors"
        )
        _NET_CACHE[key] = load_sr_model(path, cfg)
        logger.info("sr: loaded net scale=%d from %s", scale, path)
    return _NET_CACHE[key]


def _split_tiles(input_len, tile_size, overlap):
    if tile_size >= input_len:
        return [0], [input_len], []
    n = math.ceil((input_len - overlap) / (tile_size - overlap))
    starts = []
    for i in range(n):
        s = i * (tile_size - overlap)
        if s + tile_size > input_len:
            s = input_len - tile_size
        if s not in starts:
            starts.append(s)
    lens = [tile_size] * len(starts)
    overlaps = []
    for i in range(1, len(starts)):
        overlaps.append(starts[i - 1] + tile_size - starts[i])
    return starts, lens, overlaps


def _feather_weights(th, tw, y_ov, x_ov, i, n_i, j, n_j, scale, dtype):
    wg = np.ones((th, tw, 1), dtype=dtype)
    if i > 0:
        ov = y_ov[i - 1] * scale
        if ov > 0:
            ramp = np.linspace(0.0, 1.0, ov, dtype=dtype)[:, None, None]
            wg[:ov] *= ramp
    if i < n_i - 1:
        ov = y_ov[i] * scale
        if ov > 0:
            ramp = np.linspace(1.0, 0.0, ov, dtype=dtype)[:, None, None]
            wg[th - ov :] *= ramp
    if j > 0:
        ov = x_ov[j - 1] * scale
        if ov > 0:
            ramp = np.linspace(0.0, 1.0, ov, dtype=dtype)[None, :, None]
            wg[:, :ov] *= ramp
    if j < n_j - 1:
        ov = x_ov[j] * scale
        if ov > 0:
            ramp = np.linspace(1.0, 0.0, ov, dtype=dtype)[None, :, None]
            wg[:, tw - ov :] *= ramp
    return wg


def _resolve_one(frame_hwc, net, scale, tile_size, tile_overlap):
    x = frame_hwc[None, ...]
    h, w = x.shape[1], x.shape[2]
    if h <= tile_size and w <= tile_size:
        out = net(x)[0]
        mx.eval(out)
        return out
    y_idx, y_len, y_ov = _split_tiles(h, tile_size, tile_overlap)
    x_idx, x_len, x_ov = _split_tiles(w, tile_size, tile_overlap)
    oh, ow = h * scale, w * scale
    pad = tile_size
    acc = mx.zeros((oh, ow, 3), dtype=mx.float32)
    wsum = mx.zeros((oh, ow, 1), dtype=mx.float32)
    for i, (iy, ly) in enumerate(zip(y_idx, y_len)):
        oy = iy * scale
        iy_p = max(0, iy - pad)
        iy_e = min(h, iy + ly + pad)
        for j, (jx, lx) in enumerate(zip(x_idx, x_len)):
            ox = jx * scale
            jx_p = max(0, jx - pad)
            jx_e = min(w, jx + lx + pad)
            tile_in = x[:, iy_p:iy_e, jx_p:jx_e, :]
            tile_full = net(tile_in)[0]
            mx.eval(tile_full)
            oy0 = (iy - iy_p) * scale
            ox0 = (jx - jx_p) * scale
            tile_out = tile_full[oy0 : oy0 + ly * scale, ox0 : ox0 + lx * scale, :]
            mx.eval(tile_out)
            th, tw = tile_out.shape[0], tile_out.shape[1]
            wg = _feather_weights(
                th, tw, y_ov, x_ov, i, len(y_idx), j, len(x_idx), scale, np.float32
            )
            wg_mx = mx.array(wg)
            acc[oy : oy + th, ox : ox + tw] = (
                acc[oy : oy + th, ox : ox + tw] + tile_out * wg_mx
            )
            wsum[oy : oy + th, ox : ox + tw] = wsum[oy : oy + th, ox : ox + tw] + wg_mx
    out = acc / mx.maximum(wsum, mx.array(1e-8, dtype=mx.float32))
    return out


def super_resolve(
    images, model_path=None, scale=4, tile_size=512, tile_overlap=64, config=None
):
    net = _get_net(model_path, scale, config)
    n, h, w, _ = images.shape
    out = np.empty((n, h * scale, w * scale, 3), dtype=np.float32)
    for i in range(n):
        frame = mx.array(images[i].astype(np.float32))
        sr = _resolve_one(frame, net, scale, tile_size, tile_overlap)
        mx.eval(sr)
        out[i] = np.array(sr)
        logger.info(
            "sr: frame %d/%d in=%dx%d out=%dx%d",
            i + 1,
            n,
            w,
            h,
            out.shape[2],
            out.shape[1],
        )
    return out
