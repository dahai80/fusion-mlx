import mlx.core as mx
import numpy as np


def preprocess_image(image_path: str, width: int, height: int) -> mx.array:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    # Resize so that the image covers the target size (LANCZOS)
    scale = max(width / img.width, height / img.height)
    img = img.resize(
        (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
    )

    # Center crop
    x1 = (img.width - width) // 2
    y1 = (img.height - height) // 2
    img = img.crop((x1, y1, x1 + width, y1 + height))

    # To tensor: [H, W, 3] float32 in [-1, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # [0,1] → [-1,1]
    return mx.array(arr[None, None])  # [1, 1, H, W, 3]


def build_i2v_mask(z_shape, patch_size):
    C, T, H, W = z_shape
    mask = mx.ones(z_shape)
    # Zero out the first temporal position
    mask = mx.concatenate([mx.zeros((C, 1, H, W)), mask[:, 1:]], axis=1)

    # Token-level mask for per-token timesteps: subsample to patch grid
    # mask shape [C, T, H, W] → take first channel, subsample by patch_size
    pt, ph, pw = patch_size
    mask_tokens = mask[0, ::pt, ::ph, ::pw]  # [T', H', W']
    mask_tokens = mask_tokens.reshape(1, -1)  # [1, L]
    return mask, mask_tokens
