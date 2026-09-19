# SPDX-License-Identifier: Apache-2.0
# Face-parsing BiSeNet MLX backend (#910, #915).
#
# Pure-MLX port of the BiSeNet face-parsing model used by MuseTalk's blend-mask
# preprocessing (musetalk/utils/face_parsing/model.py). 19-class per-pixel label
# map. MuseTalk variant: spatial path DELETED — feat_res8 (ResNet18 layer2,
# 128ch @1/8) replaces the spatial feature in FFM ("## here self.sp is deleted"
# in the source). Checkpoint 79999_iter.pth keys match this module tree exactly:
#
#   cp.resnet.{conv1,bn1,layer1..4}            — ResNet18 (torchvision weights)
#   cp.resnet.layer{2,3,4}.0.downsample.{0,1}  — Sequential conv+bn (keys .0/.1)
#   cp.arm16 / cp.arm32.{conv.{conv,bn}, conv_atten, bn_atten}
#   cp.conv_avg.{conv,bn}, cp.conv_head16/{conv,bn}, cp.conv_head32/{conv,bn}
#   ffm.{convblk.{conv,bn}, conv1, conv2}      — attention convs bias=False
#   conv_out / conv_out16 / conv_out32.{conv.{conv,bn}, conv_out}
#
# All convs bias=False (checkpoint carries no conv bias). Convert script
# transposes torch OIHW → MLX OHWI and drops num_batches_tracked.
#
# 19 classes: background/skin/l_brow/r_brow/l_eye/r_eye/eye_g/l_ear/r_ear/ear_r/
#   nose/mouth/u_lip/l_lip/neck/neck_l/cloth/hair/hat.

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD = (0.229, 0.224, 0.225)
_INPUT_SIZE = 512
_NUM_CLASSES = 19

_FACE_CLASSES = {1, 10, 12, 13}  # skin, nose, u_lip, l_lip — MuseTalk blend mask


class ConvBNReLU(nn.Module):
    """Conv2d (bias=False) + BatchNorm + ReLU — mirrors source ConvBNReLU keys."""

    def __init__(self, in_c, out_c, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c, ks, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm(out_c)

    def __call__(self, x):
        return nn.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """ResNet18 BasicBlock — downsample as [conv, bn] list (keys .0/.1)."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_c)
        self.downsample = None
        if in_c != out_c or stride != 1:
            self.downsample = [
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm(out_c),
            ]

    def __call__(self, x):
        h = nn.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        res = x
        if self.downsample is not None:
            res = self.downsample[1](self.downsample[0](x))
        return nn.relu(h + res)


class ResNet18Backbone(nn.Module):
    """ResNet18 — returns feat8 (128 @1/8), feat16 (256 @1/16), feat32 (512 @1/32)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = [BasicBlock(64, 64), BasicBlock(64, 64)]
        self.layer2 = [BasicBlock(64, 128, stride=2), BasicBlock(128, 128)]
        self.layer3 = [BasicBlock(128, 256, stride=2), BasicBlock(256, 256)]
        self.layer4 = [BasicBlock(256, 512, stride=2), BasicBlock(512, 512)]

    @staticmethod
    def _run(x, blocks):
        for blk in blocks:
            x = blk(x)
        return x

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self._run(x, self.layer1)
        feat8 = self._run(x, self.layer2)  # 1/8, 128
        feat16 = self._run(feat8, self.layer3)  # 1/16, 256
        feat32 = self._run(feat16, self.layer4)  # 1/32, 512
        return feat8, feat16, feat32


class ARM(nn.Module):
    """Attention Refinement Module — keys conv.{conv,bn} / conv_atten / bn_atten."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBNReLU(in_c, out_c, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_c, out_c, 1, bias=False)
        self.bn_atten = nn.BatchNorm(out_c)

    def __call__(self, x):
        feat = self.conv(x)
        atten = mx.mean(feat, axis=(1, 2), keepdims=True)
        atten = mx.sigmoid(self.bn_atten(self.conv_atten(atten)))
        return feat * atten


class ContextPath(nn.Module):
    """ResNet18 + ARM16/ARM32 + conv_avg/conv_head — returns (feat8, feat_cp8, feat_cp16)."""

    def __init__(self):
        super().__init__()
        self.resnet = ResNet18Backbone()
        self.arm16 = ARM(256, 128)
        self.arm32 = ARM(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def __call__(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        avg = mx.mean(feat32, axis=(1, 2), keepdims=True)
        avg = self.conv_avg(avg)
        avg_up = _nearest_up(avg, feat32.shape[1], feat32.shape[2])

        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = self.conv_head32(
            _nearest_up(feat32_sum, feat16.shape[1], feat16.shape[2])
        )

        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = self.conv_head16(
            _nearest_up(feat16_sum, feat8.shape[1], feat8.shape[2])
        )
        return feat8, feat16_up, feat32_up  # x8 (128), x8 (128), x16 (128)


class FFM(nn.Module):
    """Feature Fusion Module — convblk + channel attention (conv1/conv2 bias=False)."""

    def __init__(self, in_c, out_c=256):
        super().__init__()
        self.convblk = ConvBNReLU(in_c, out_c, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_c, out_c // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c // 4, out_c, 1, bias=False)

    def __call__(self, fsp, fcp):
        feat = self.convblk(mx.concatenate([fsp, fcp], axis=-1))
        atten = mx.mean(feat, axis=(1, 2), keepdims=True)
        atten = nn.relu(self.conv1(atten))
        atten = mx.sigmoid(self.conv2(atten))
        return feat * atten + feat


class BiSeNetOutput(nn.Module):
    """conv (ConvBNReLU 3×3) + 1×1 class conv — keys conv.{conv,bn} / conv_out."""

    def __init__(self, in_c, mid_c, n_classes=_NUM_CLASSES):
        super().__init__()
        self.conv = ConvBNReLU(in_c, mid_c, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_c, n_classes, 1, bias=False)

    def __call__(self, x):
        return self.conv_out(self.conv(x))


class BiSeNetMLX(nn.Module):
    """BiSeNet face-parsing (MuseTalk variant, no spatial path) — 19-class logits (#910)."""

    def __init__(self, n_classes=_NUM_CLASSES):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FFM(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def __call__(self, x):
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = feat_res8  # source replaces deleted spatial path with res3b1 feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)
        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)
        h, w = x.shape[1], x.shape[2]
        return (
            _bilinear_up(feat_out, h, w),
            _bilinear_up(feat_out16, h, w),
            _bilinear_up(feat_out32, h, w),
        )


def _nearest_up(x, out_h, out_w):
    """Nearest-neighbor upsample to (out_h, out_w) — torch F.interpolate(mode='nearest')."""
    h, w = x.shape[1], x.shape[2]
    if (h, w) == (out_h, out_w):
        return x
    iy = (mx.arange(out_h, dtype=mx.int32) * h) // out_h
    ix = (mx.arange(out_w, dtype=mx.int32) * w) // out_w
    return x[:, iy[:, None], ix[None, :], :]


def _bilinear_up(x, out_h, out_w):
    """Bilinear upsample, align_corners=True — torch F.interpolate(mode='bilinear')."""
    h, w = x.shape[1], x.shape[2]
    if (h, w) == (out_h, out_w):
        return x
    if out_h > 1:
        ys = mx.arange(out_h, dtype=mx.float32) * (h - 1) / (out_h - 1)
    else:
        ys = mx.zeros((out_h,), dtype=mx.float32)
    if out_w > 1:
        xs = mx.arange(out_w, dtype=mx.float32) * (w - 1) / (out_w - 1)
    else:
        xs = mx.zeros((out_w,), dtype=mx.float32)
    y0 = ys.astype(mx.int32)
    x0 = xs.astype(mx.int32)
    y1 = mx.minimum(y0 + 1, h - 1)
    x1 = mx.minimum(x0 + 1, w - 1)
    wy = (ys - y0.astype(mx.float32))[:, None, None]
    wx = (xs - x0.astype(mx.float32))[None, :, None]
    a = x[:, y0[:, None], x0[None, :], :]
    b = x[:, y0[:, None], x1[None, :], :]
    c = x[:, y1[:, None], x0[None, :], :]
    d = x[:, y1[:, None], x1[None, :], :]
    return a * (1 - wy) * (1 - wx) + b * wx * (1 - wy) + c * (1 - wy) * wy + d * wx * wy


def preprocess(frame_bgr, input_size=_INPUT_SIZE):
    """BGR uint8 (H,W,3) → normalized NHWC float32 (1,size,size,3) for BiSeNet."""
    import numpy as np

    if hasattr(frame_bgr, "shape") and frame_bgr.dtype == np.uint8:
        rgb = frame_bgr[..., ::-1].astype(np.float32) / 255.0
    else:
        rgb = np.asarray(frame_bgr, dtype=np.float32) / 255.0
    if rgb.shape[:2] != (input_size, input_size):
        from fusion_mlx.video.dwpose import _bilinear_resize

        rgb = _bilinear_resize(rgb, input_size, input_size)
    mean = np.array(_IMG_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(_IMG_STD, dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / std
    return mx.array(rgb)[None]


def decode_parsing(logits, face_classes=_FACE_CLASSES):
    """Logits (C,H,W) → label map (H,W) int + face mask (H,W) bool.

    face mask keeps the MuseTalk blend classes (skin/nose/u_lip/l_lip) so the
    caller can build the mouth-region paste alpha.
    """
    import numpy as np

    labels = np.argmax(np.array(logits), axis=0).astype(np.int32)
    face_mask = np.isin(labels, list(face_classes))
    return labels, face_mask


class FaceParsing:
    """High-level face-parsing wrapper (#910)."""

    def __init__(self, model: BiSeNetMLX):
        self.model = model
        self.model.eval()

    @classmethod
    def from_pretrained(cls, weights_dir: str | Path | None = None) -> FaceParsing:
        root = (
            Path(weights_dir)
            if weights_dir
            else Path.home() / ".fusion-mlx" / "models" / "face-parsing"
        )
        model = BiSeNetMLX()
        st_path = root / "79999_iter.safetensors"
        if st_path.exists():
            _load_safetensors(model, st_path)
            logger.info("[face_parsing] loaded weights from %s", st_path)
        else:
            logger.warning(
                "[face_parsing] no weights at %s — running untrained (shape only). "
                "Run scripts/convert_face_parsing.py to convert 79999_iter.pth.",
                st_path,
            )
        return cls(model)

    def parse(self, frame_bgr):
        """BGR uint8 (H,W,3) → (labels (H,W) int, face_mask (H,W) bool)."""
        x = preprocess(frame_bgr)
        logits = self.model(x)[0][0]  # (H, W, 19) — already upsampled to input size
        return decode_parsing(logits.transpose(2, 0, 1))

    def face_mask(self, frame_bgr):
        """BGR frame → boolean face-skin mask (H,W) for blend-paste alpha."""
        _, mask = self.parse(frame_bgr)
        return mask


def _load_safetensors(model: nn.Module, path: Path):
    """Load safetensors into model with strict key verification (fail loud on mismatch)."""
    try:
        from safetensors import safe_open

        weights = {}
        with safe_open(str(path), framework="numpy") as f:
            for key in f.keys():  # noqa: SIM118
                weights[key] = mx.array(f.get_tensor(key))
    except ImportError:
        weights = mx.load(str(path))

    from fusion_mlx.video.dwpose import _flatten_param_keys

    expected = set(_flatten_param_keys(model.parameters()))
    got = set(weights.keys())
    missing = expected - got
    extra = got - expected
    if missing or extra:
        raise ValueError(
            f"[face_parsing] weight key mismatch ({path}):\n"
            f"  missing ({len(missing)}): {sorted(missing)[:10]}\n"
            f"  extra   ({len(extra)}): {sorted(extra)[:10]}\n"
            f"Re-run scripts/convert_face_parsing.py against 79999_iter.pth."
        )
    model.load_weights(list(weights.items()))
