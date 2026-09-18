# SPDX-License-Identifier: Apache-2.0
# Face-parsing BiSeNet MLX backend (#910).
#
# Pure-MLX port of the BiSeNet face-parsing model (ResNet18 backbone + spatial/
# context paths + ARM + FFM) used by MuseTalk's blend-mask preprocessing.
# 19-class per-pixel label map; MuseTalk keeps face-skin/lower-mouth classes for
# the paste alpha mask. Checkpoint resnet18-5c106cde.pth (zllrunning/face-parsing.
# PyTorch) converted offline to safetensors (no torch at runtime — PRD gate).
#
# Architecture (zllrunning/face-parsing.PyTorch):
#   Spatial path: 3× (Conv+BN+ReLU, stride 2) → 1/8 features (256ch)
#   Context path: ResNet18 (layer1-4), take layer3 (1/16, 256ch) + layer4 (1/32, 512ch)
#     ARM on each + global avg pool branch → upsample
#   FFM: fuse spatial (256) + context (512) → 256 → head conv → 19 logits
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
    """Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm(out_c)
        self.act = nn.ReLU()

    def __call__(self, x):
        return self.act(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """ResNet18 BasicBlock: conv3+bn+relu+conv3+bn + identity residual."""

    def __init__(self, in_c, out_c, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm(out_c)
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride), nn.BatchNorm(out_c))
            if downsample
            else None
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = nn.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        identity = x if self.downsample is None else self.downsample(x)
        return nn.relu(h + identity)


class ResNet18Backbone(nn.Module):
    """ResNet18 (ImageNet) — returns layer3 (1/16) and layer4 (1/32) features."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, downsample=True), BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, downsample=True), BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2, downsample=True), BasicBlock(512, 512)
        )

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)  # 1/16, 256ch
        c4 = self.layer4(c3)  # 1/32, 512ch
        return c3, c4


class SpatialPath(nn.Module):
    """3× stride-2 Conv+BN+ReLU → 1/8 features (256ch)."""

    def __init__(self):
        super().__init__()
        self.b1 = ConvBNReLU(3, 64, kernel=7, stride=2, padding=3)
        self.b2 = ConvBNReLU(64, 128, kernel=3, stride=2, padding=1)
        self.b3 = ConvBNReLU(128, 256, kernel=3, stride=2, padding=1)

    def __call__(self, x):
        return self.b3(self.b2(self.b1(x)))


class ARM(nn.Module):
    """Attention Refinement Module: global avg pool → 1×1 conv → sigmoid → scale."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1)
        self.bn = nn.BatchNorm(out_c)

    def __call__(self, x):
        pooled = mx.mean(x, axis=(1, 2), keepdims=True)
        attn = mx.sigmoid(self.bn(self.conv(pooled)))
        return x * attn


class FFM(nn.Module):
    """Feature Fusion Module: concat spatial + context → conv → attention → 256."""

    def __init__(self, in_c, out_c=256):
        super().__init__()
        self.conv = ConvBNReLU(in_c, out_c, kernel=1, stride=1, padding=0)
        self.att_conv = nn.Conv2d(out_c, out_c, 1)

    def __call__(self, spatial, context):
        x = mx.concatenate([spatial, context], axis=-1)
        x = self.conv(x)
        pooled = mx.mean(x, axis=(1, 2), keepdims=True)
        attn = mx.sigmoid(self.att_conv(pooled))
        return x * attn


class BiSeNetMLX(nn.Module):
    """BiSeNet face-parsing (ResNet18) — 19-class per-pixel logits (#910)."""

    def __init__(self, num_classes=_NUM_CLASSES):
        super().__init__()
        self.spatial = SpatialPath()
        self.context_backbone = ResNet18Backbone()
        self.arm_c3 = ARM(256, 256)
        self.arm_c4 = ARM(512, 512)
        # Match c4 (512) → 256 so it can add to c3 (256).
        self.c4_reduce = ConvBNReLU(512, 256, kernel=1, stride=1, padding=0)
        self.global_pool_conv = ConvBNReLU(512, 256, kernel=1, stride=1, padding=0)
        # FFM fuses spatial(256) + context(256).
        self.ffm = FFM(256 + 256, 256)
        self.head = nn.Conv2d(256, num_classes, 1)

    def __call__(self, x):
        sp = self.spatial(x)  # 1/8, 256ch
        c3, c4 = self.context_backbone(x)
        c3 = self.arm_c3(c3)  # 1/16, 256
        c4 = self.arm_c4(c4)  # 1/32, 512
        c4_red = self.c4_reduce(c4)  # 1/32, 256
        # Global context: pool c4 → 256 → upsample.
        gp = mx.mean(c4, axis=(1, 2), keepdims=True)
        gp = self.global_pool_conv(gp)  # 1×1, 256
        # Combine at c3 scale (1/16): c4_red_up + c3 + gp_up.
        c4_up = _upsample(c4_red, c3.shape[1], c3.shape[2])
        gp_up = _upsample(gp, c3.shape[1], c3.shape[2])
        context = c4_up + c3 + gp_up  # 1/16, 256
        # Upsample context to spatial scale (1/8).
        context_up = _upsample(context, sp.shape[1], sp.shape[2])
        fused = self.ffm(sp, context_up)
        logits = self.head(fused)
        return _upsample(logits, x.shape[1], x.shape[2])


def _upsample(x, out_h, out_w):
    """Nearest-neighbor upsample to (out_h, out_w) — MLX has no interpolate w/ size."""
    in_h, in_w = x.shape[1], x.shape[2]
    if (in_h, in_w) == (out_h, out_w):
        return x
    sy = mx.arange(out_h, dtype=mx.float32) * (in_h / out_h)
    sx = mx.arange(out_w, dtype=mx.float32) * (in_w / out_w)
    iy = mx.minimum(sy.astype(mx.int32), in_h - 1)
    ix = mx.minimum(sx.astype(mx.int32), in_w - 1)
    return x[:, iy[:, None], ix[None, :], :]


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
        st_path = root / "resnet18-5c106cde.safetensors"
        if st_path.exists():
            _load_safetensors(model, st_path)
            logger.info("[face_parsing] loaded weights from %s", st_path)
        else:
            logger.warning(
                "[face_parsing] no weights at %s — running untrained (shape only). "
                "Run scripts/convert_face_parsing.py to convert resnet18-5c106cde.pth.",
                st_path,
            )
        return cls(model)

    def parse(self, frame_bgr):
        """BGR uint8 (H,W,3) → (labels (H,W) int, face_mask (H,W) bool)."""
        x = preprocess(frame_bgr)
        logits = self.model(x)[0]  # (C, H, W) NHWC → last axis is C
        return decode_parsing(logits.transpose(2, 0, 1))

    def face_mask(self, frame_bgr):
        """BGR frame → boolean face-skin mask (H,W) for blend-paste alpha."""
        _, mask = self.parse(frame_bgr)
        return mask


def _load_safetensors(model: nn.Module, path: Path):
    try:
        from safetensors.safe_open import safe_open

        weights = {}
        with safe_open(str(path), framework="numpy") as f:
            for key in f:
                weights[key] = mx.array(f.get_tensor(key))
        model.load_weights(list(weights.items()))
    except ImportError:
        weights = mx.load(str(path))
        model.load_weights(list(weights.items()))
