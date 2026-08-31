import logging

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def apply_inpaint_mask(latents, init_latent, mask):
    # #653 Surface C: neutral per-step re-composite. mask=1 -> reactive
    # (keep denoised); mask=0 -> frozen (restore init). Orthogonal to
    # ControlState; never touches conditioning. None -> T2V passthrough.
    if mask is None or init_latent is None:
        return latents
    if latents.shape != init_latent.shape:
        raise ValueError(
            f"apply_inpaint_mask: init_latent shape {tuple(init_latent.shape)}"
            f" != latents {tuple(latents.shape)}"
        )
    return mask * latents + (1.0 - mask) * init_latent


def patch_downsample_mask(mask, vae_stride, patch_size, t_latent, h_latent, w_latent):
    # Average-pool a pixel-space mask (H, W) or (T, H, W) to latent spatial
    # size, broadcast to (1, t_latent, h_latent, w_latent) for 4D Wan2
    # latents. vae_stride=(s_t, s_h, s_w); last two mask dims are pixel H/W.
    arr = np.array(mask, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, None]
    elif arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"patch_downsample_mask: mask ndim {arr.ndim} not 2/3/4")
    _, _, h_px, w_px = arr.shape
    s_h, s_w = vae_stride[1], vae_stride[2]
    if h_px // s_h != h_latent or w_px // s_w != w_latent:
        logger.warning(
            "patch_downsample_mask: px %dx%d / stride %dx%d != latent %dx%d",
            h_px,
            w_px,
            s_h,
            s_w,
            h_latent,
            w_latent,
        )
    arr = arr[:, :, : h_latent * s_h, : w_latent * s_w]
    arr = arr.reshape(1, 1, h_latent, s_h, w_latent, s_w).mean(axis=(3, 5))
    out = mx.array(arr)
    out = mx.broadcast_to(out, (1, t_latent, h_latent, w_latent))
    return out
