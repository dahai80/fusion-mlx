import logging

import mlx.core as mx
from mlx import nn

from fusion_mlx.image.sdxl.unet import Conv2d
from fusion_mlx.image.sr.config import RealESRGANConfig

logger = logging.getLogger(__name__)


def leaky_relu(x: mx.array, slope: float = 0.2) -> mx.array:
    return mx.where(x > 0, x, slope * x)


def pixel_shuffle(x: mx.array, upscale: int) -> mx.array:
    b, h, w, cr2 = x.shape
    c = cr2 // (upscale * upscale)
    x = x.reshape(b, h, w, c, upscale, upscale)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    x = x.reshape(b, h * upscale, w * upscale, c)
    return x


def nearest_upsample(x: mx.array, factor: int) -> mx.array:
    x = mx.repeat(x, factor, axis=1)
    x = mx.repeat(x, factor, axis=2)
    return x


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int):
        super().__init__()
        self.conv1 = Conv2d(num_feat, num_grow_ch)
        self.conv2 = Conv2d(num_feat + num_grow_ch, num_grow_ch)
        self.conv3 = Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch)
        self.conv4 = Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch)
        self.conv5 = Conv2d(num_feat + 4 * num_grow_ch, num_feat)

    def __call__(self, x: mx.array) -> mx.array:
        x1 = leaky_relu(self.conv1(x))
        x2 = leaky_relu(self.conv2(mx.concatenate([x, x1], axis=-1)))
        x3 = leaky_relu(self.conv3(mx.concatenate([x, x1, x2], axis=-1)))
        x4 = leaky_relu(self.conv4(mx.concatenate([x, x1, x2, x3], axis=-1)))
        x5 = self.conv5(mx.concatenate([x, x1, x2, x3, x4], axis=-1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int, res_scale: float):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.res_scale = res_scale

    def __call__(self, x: mx.array) -> mx.array:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.res_scale + x


class RRDBNet(nn.Module):
    def __init__(self, config: RealESRGANConfig):
        super().__init__()
        self.scale = config.scale
        nf = config.num_feat
        self.conv_first = Conv2d(config.num_in_ch, nf)
        self.body = [
            RRDB(nf, config.num_grow_ch, config.res_scale)
            for _ in range(config.num_block)
        ]
        self.conv_body = Conv2d(nf, nf)
        if config.scale == 4:
            self.conv_up1 = Conv2d(nf, nf)
            self.conv_up2 = Conv2d(nf, nf)
        elif config.scale == 2:
            self.conv_up1 = Conv2d(nf, nf)
        self.conv_hr = Conv2d(nf, nf)
        self.conv_last = Conv2d(nf, config.num_out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        feat = self.conv_first(x)
        body_feat = feat
        for blk in self.body:
            feat = blk(feat)
        feat = self.conv_body(feat)
        feat = body_feat + feat
        if self.scale == 4:
            feat = nearest_upsample(feat, 2)
            feat = leaky_relu(self.conv_up1(feat))
            feat = nearest_upsample(feat, 2)
            feat = leaky_relu(self.conv_up2(feat))
        elif self.scale == 2:
            feat = nearest_upsample(feat, 2)
            feat = leaky_relu(self.conv_up1(feat))
        out = self.conv_last(leaky_relu(self.conv_hr(feat)))
        return out
