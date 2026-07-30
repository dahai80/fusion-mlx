import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        out = nn.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class SimpleCameraAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        downscale_factor: int = 8,
        num_residual_blocks: int = 1,
    ):
        super().__init__()
        self.downscale_factor = downscale_factor
        unshuffled_dim = in_dim * downscale_factor * downscale_factor
        self.conv = nn.Conv2d(
            unshuffled_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.residual_blocks = [
            ResidualBlock(out_dim) for _ in range(num_residual_blocks)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C_cam, F, H, W]
        bs, c, f, h, w = x.shape
        # Merge frame dim into batch: [B*F, C_cam, H, W]
        x = x.transpose(0, 2, 1, 3, 4).reshape(bs * f, c, h, w)

        # Pixel unshuffle: [B*F, C_cam*ds^2, H/ds, W/ds]
        ds = self.downscale_factor
        x = x.reshape(bs * f, c, h // ds, ds, w // ds, ds)
        x = x.transpose(0, 1, 3, 5, 2, 4)
        x = x.reshape(bs * f, c * ds * ds, h // ds, w // ds)

        # MLX Conv2d uses NHWC layout — transpose NCHW -> NHWC
        x = x.transpose(0, 2, 3, 1)
        out = self.conv(x)
        out = out.transpose(0, 3, 1, 2)  # NHWC -> NCHW

        for block in self.residual_blocks:
            out = out.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            out = block(out)
            out = out.transpose(0, 3, 1, 2)  # NHWC -> NCHW

        # Restore: [B, F, out_dim, H', W'] -> [B, out_dim, F, H', W']
        out = out.reshape(bs, f, out.shape[1], out.shape[2], out.shape[3])
        out = out.transpose(0, 2, 1, 3, 4)

        logger.debug("CameraAdapter output shape: %s", out.shape)
        return out
