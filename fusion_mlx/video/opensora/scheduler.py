# SPDX-License-Identifier: Apache-2.0
# Open-Sora V2 rectified-flow scheduler with time_shift.

import logging
import math

import mlx.core as mx

logger = logging.getLogger(__name__)


def time_shift(alpha: float, t: mx.array) -> mx.array:
    sigma = alpha / (1 + mx.exp(-t * (1 - t) * alpha))
    return sigma


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    num_frames: int,
    shift: float = 1.0,
    shift_alpha: float = 1.0,
):
    timesteps = mx.linspace(1.0, 0.0, num_steps + 1).tolist()
    timesteps = [float(time_shift(shift_alpha, mx.array(t))) for t in timesteps]
    schedule = []
    for i in range(num_steps):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]
        schedule.append((t_cur, t_next))
    return schedule


def get_noise(
    num_frames: int,
    height: int,
    width: int,
    batch_size: int = 1,
    in_channels: int = 64,
    seed: int | None = None,
) -> mx.array:
    if seed is not None:
        mx.random.seed(seed)
    return mx.random.normal(shape=(batch_size, in_channels, num_frames, height, width))


def pack(x: mx.array, patch_size: int = 2) -> mx.array:
    B, C, T, H, W = x.shape
    ph = pw = patch_size
    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw
    if pad_h > 0 or pad_w > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)])
        H = H + pad_h
        W = W + pad_w
    x = x.reshape(B, C, T, H // ph, ph, W // pw, pw)
    x = x.transpose(0, 2, 3, 5, 4, 6, 1)
    x = x.reshape(B, T * (H // ph) * (W // pw), C * ph * pw)
    return x


def unpack(
    x: mx.array, height: int, width: int, num_frames: int, patch_size: int = 2
) -> mx.array:
    B = x.shape[0]
    C_total = x.shape[2]
    ph = pw = patch_size
    C = C_total // (ph * pw)
    H_patches = height // ph
    W_patches = width // pw
    x = x.reshape(B, num_frames, H_patches, W_patches, ph, pw, C)
    x = x.transpose(0, 6, 1, 2, 4, 3, 5)
    x = x.reshape(B, C, num_frames, H_patches * ph, W_patches * pw)
    return x


def get_image_ids(num_frames: int, height: int, width: int, patch_size: int = 2):
    H_patches = height // patch_size
    W_patches = width // patch_size
    t_ids = mx.arange(num_frames, dtype=mx.float32)
    h_ids = mx.arange(H_patches, dtype=mx.float32)
    w_ids = mx.arange(W_patches, dtype=mx.float32)
    grid_t, grid_h, grid_w = mx.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    ids = mx.stack(
        [grid_t.reshape(-1), grid_h.reshape(-1), grid_w.reshape(-1)], axis=-1
    )
    return ids[None, :, :]


def get_txt_ids(seq_len: int):
    return mx.zeros((1, seq_len, 3), dtype=mx.float32)
