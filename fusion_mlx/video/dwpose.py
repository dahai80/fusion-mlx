# SPDX-License-Identifier: Apache-2.0
# DWPose / RTMPose-L MLX backend (#909).
#
# Pure-MLX port of the RTMPose-L whole-body pose model (CSPNeXt-L backbone +
# RTMCCHead SimCC coordinate-classification head) used by MuseTalk's face-landmark
# preprocessing. 133 COCO-WholeBody keypoints (17 body + 6 foot + 68 face [23:91]
# + 42 hand). Checkpoint dw-ll_ucoco_384.pth (yzd-v/DWPose on HF, ~388MB) is
# converted offline to safetensors (no torch at runtime — PRD gate).
#
# Architecture (from rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py):
#   Backbone: CSPNeXt-L (P5), single sequential chain (NOT HRNet multi-branch).
#     stem 3×Conv(3→32→32→64) + 4 stages(conv-stride2 + CSPLayer + SPPBottleneck@stage4)
#     out_indices=(4,) → [B,1024,9,12]  (H×W = 9×12 for 384×288 input)
#   Head: RTMCCHead — final_layer Conv(1024→133,k7) → flatten[108] → ScaleNorm+Linear(108→256)
#         → RTMCCBlock(GAU) → cls_x(256→576) + cls_y(256→768)
#   Decode: SimCC — argmax(simcc_x,576) + argmax(simcc_y,768) → locs/2.0 → pixel coords
#
# GAU (RTMCCBlock) config: use_rel_bias=False, pos_enc=False → pure tensor ops
# (ScaleNorm→Linear→SiLU→split→bmm→relu²→bmm→Linear+residual), no RoPE/bias.

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_IMG_MEAN = (123.675, 116.28, 103.53)
_IMG_STD = (58.395, 57.12, 57.375)
_INPUT_W = 288
_INPUT_H = 384
_SIMCC_SPLIT_RATIO = 2.0
_NUM_KEYPOINTS = 133


class DepthwiseSeparableConv(nn.Module):
    """Depthwise (groups=C) 5×5 + pointwise 1×1 conv (CSPNeXtBlock conv2)."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(in_c, in_c, kernel_size, padding=pad, groups=in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1)

    def __call__(self, x):
        return self.pw(self.dw(x))


class CSPNeXtBlock(nn.Module):
    """conv1(3×3) + DepthwiseSeparableConv(5×5+1×1) + identity residual."""

    def __init__(self, in_c: int, out_c: int, expand: float = 0.5):
        super().__init__()
        mid = int(in_c * expand)
        self.conv1 = nn.Conv2d(in_c, mid, 3, padding=1)
        self.conv2 = DepthwiseSeparableConv(mid, out_c)

    def __call__(self, x):
        h = self.conv2(nn.silu(self.conv1(x)))
        return x + h if x.shape[-1] == h.shape[-1] else h


class ChannelAttention(nn.Module):
    """AdaptiveAvgPool2d(1) → 1×1 conv → Hardsigmoid channel reweight."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

    def _hardsigmoid(self, x):
        # hardsigmoid = relu6(x+3)/6 — matches torch.nn.Hardsigmoid.
        return mx.minimum(mx.maximum(x + 3.0, 0.0), 6.0) / 6.0

    def __call__(self, x):
        pooled = mx.mean(x, axis=(1, 2), keepdims=True)
        return self._hardsigmoid(self.conv(pooled)) * x


class CSPLayer(nn.Module):
    """main_conv + N×CSPNeXtBlock + short_conv → concat → ChannelAttention → final_conv."""

    def __init__(self, in_c: int, out_c: int, num_blocks: int = 3, expand: float = 0.5):
        super().__init__()
        mid = int(out_c * expand)
        self.main_conv = nn.Conv2d(in_c, mid, 1)
        self.blocks = [CSPNeXtBlock(mid, mid, expand) for _ in range(num_blocks)]
        self.short_conv = nn.Conv2d(in_c, mid, 1)
        self.attn = ChannelAttention(mid * 2)
        self.final_conv = nn.Conv2d(mid * 2, out_c, 1)

    def __call__(self, x):
        main = self.main_conv(x)
        for blk in self.blocks:
            main = blk(main)
        short = self.short_conv(x)
        cat = mx.concatenate([short, main], axis=-1)
        cat = self.attn(cat)
        return self.final_conv(cat)


class SPPBottleneck(nn.Module):
    """conv(1×1) → parallel MaxPool(k=5,9,13) → concat → conv(1×1)."""

    def __init__(self, in_c: int, out_c: int, kernels=(5, 9, 13)):
        super().__init__()
        mid = in_c // 2
        self.conv1 = nn.Conv2d(in_c, mid, 1)
        self.pools = [nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernels]
        self.conv2 = nn.Conv2d(mid * (1 + len(kernels)), out_c, 1)

    def __call__(self, x):
        x = nn.silu(self.conv1(x))
        outs = [x] + [p(x) for p in self.pools]
        cat = mx.concatenate(outs, axis=-1)
        return nn.silu(self.conv2(cat))


_CSPNEXT_P5 = [
    (64, 128, 3, True, False),
    (128, 256, 6, True, False),
    (256, 512, 6, True, False),
    (512, 1024, 3, False, True),
]


class CSPNeXtBackbone(nn.Module):
    """CSPNeXt-L (P5) backbone — stem + 4 stages, returns stage4 [B,1024,9,12]."""

    def __init__(self, arch="P5", widen=1.0):
        super().__init__()
        arch_settings = _CSPNEXT_P5
        self.stem = nn.Sequential(
            nn.Conv2d(3, int(32 * widen), 3, stride=2, padding=1),
            nn.Conv2d(int(32 * widen), int(32 * widen), 3, padding=1),
            nn.Conv2d(int(32 * widen), int(64 * widen), 3, padding=1),
        )
        self.stages = []
        for in_c, out_c, n_blocks, add_identity, use_spp in arch_settings:
            in_c = int(in_c * widen)
            out_c = int(out_c * widen)
            layers = [nn.Conv2d(in_c, out_c, 3, stride=2, padding=1)]
            if use_spp:
                layers.append(SPPBottleneck(out_c, out_c))
            layers.append(CSPLayer(out_c, out_c, n_blocks))
            self.stages.append(nn.Sequential(*layers))

    def __call__(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x  # [B, 1024, 9, 12]


class ScaleNorm(nn.Module):
    """L2-norm × scale × g (RTMCCHead input norm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = mx.ones((1,))
        self.eps = eps

    def __call__(self, x):
        norm = mx.sqrt(
            mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps
        )
        return self.scale * (x / norm)


class RTMCCBlock(nn.Module):
    """Gated Attention Unit (use_rel_bias=False, pos_enc=False → pure tensor ops)."""

    def __init__(
        self,
        num_token: int,
        in_dim: int,
        out_dim: int,
        s: int = 128,
        expansion: int = 2,
    ):
        super().__init__()
        self.num_token = num_token
        self.s = s
        e = in_dim * expansion
        self.norm = ScaleNorm(in_dim)
        self.uv = nn.Linear(in_dim, 2 * e + s, bias=False)
        self.gamma = mx.ones((2, s))
        self.beta = mx.zeros((2, s))
        self.out_proj = nn.Linear(e, out_dim)

    def __call__(self, x):
        # x: (B, K, in_dim)
        x = self.norm(x)
        uv = self.uv(x)
        e = uv.shape[-1] - self.s
        half = e // 2
        u = uv[..., :half]
        v = uv[..., half : half * 2]
        base = uv[..., half * 2 :]
        base = base[:, :, None, :] * self.gamma + self.beta  # (B,K,2,s)
        q, k = base[..., 0, :], base[..., 1, :]  # each (B,K,s)
        qk = mx.matmul(q, mx.transpose(k, (0, 2, 1)))  # (B,K,K)
        kernel = mx.square(mx.maximum(qk / (self.s**0.5), 0.0))
        out = u * mx.matmul(kernel, v)  # (B,K,e/2)
        return (
            self.out_proj(out) + x
            if x.shape[-1] == out.shape[-1]
            else self.out_proj(out)
        )


class RTMCCHead(nn.Module):
    """SimCC coordinate-classification head: final_layer conv → flatten → GAU → cls_x/cls_y."""

    def __init__(
        self, in_channels=1024, out_channels=133, in_featuremap_size=(12, 9), s=128
    ):
        super().__init__()
        fh, fw = in_featuremap_size
        self.flat_dim = fh * fw
        self.final_layer = nn.Conv2d(in_channels, out_channels, 7, padding=3)
        self.mlp_scale = ScaleNorm(self.flat_dim)
        self.mlp_proj = nn.Linear(self.flat_dim, 256, bias=False)
        self.gau = RTMCCBlock(out_channels, 256, 256, s=s)
        self.cls_x = nn.Linear(256, int(_INPUT_W * _SIMCC_SPLIT_RATIO), bias=False)
        self.cls_y = nn.Linear(256, int(_INPUT_H * _SIMCC_SPLIT_RATIO), bias=False)

    def __call__(self, feats):
        # feats: (B, 1024, 9, 12) NHWC → (B, 133, 9, 12) → (B, 133, 108)
        x = self.final_layer(feats)
        b, fh, fw, c = x.shape
        x = x.reshape(b, fh * fw, c).transpose(0, 2, 1)  # (B, 133, 108)
        x = self.mlp_proj(self.mlp_scale(x))
        x = self.gau(x)
        simcc_x = self.cls_x(x)
        simcc_y = self.cls_y(x)
        return simcc_x, simcc_y


class DWPoseMLX(nn.Module):
    """RTMPose-L whole-body pose estimator (133 keypoints) — pure MLX (#909)."""

    def __init__(self):
        super().__init__()
        self.backbone = CSPNeXtBackbone()
        self.head = RTMCCHead()

    def __call__(self, x):
        feats = self.backbone(x)
        return self.head(feats)


def decode_simcc(simcc_x, simcc_y, split_ratio=_SIMCC_SPLIT_RATIO):
    """SimCC decode: argmax → locs/split_ratio → pixel coords + per-point score."""
    x_loc = mx.argmax(simcc_x, axis=-1)
    y_loc = mx.argmax(simcc_y, axis=-1)
    locs = mx.stack([x_loc, y_loc], axis=-1).astype(mx.float32) / split_ratio
    x_score = mx.max(simcc_x, axis=-1)
    y_score = mx.max(simcc_y, axis=-1)
    scores = mx.minimum(x_score, y_score)
    return locs, scores


def preprocess(frame_bgr, input_w=_INPUT_W, input_h=_INPUT_H):
    """BGR uint8 (H,W,3) → normalized NHWC float32 (1,input_h,input_w,3) for RTMPose.

    Caller is expected to pre-warp (affine) the face region to input_h×input_w;
    if the frame is already that size this normalizes in place. A simple bilinear
    resize is applied as a fallback when sizes differ (no affine warp available).
    """
    import numpy as np

    if hasattr(frame_bgr, "shape") and frame_bgr.dtype == np.uint8:
        rgb = frame_bgr[..., ::-1].astype(np.float32)
    else:
        rgb = np.asarray(frame_bgr, dtype=np.float32)
    if rgb.shape[:2] != (input_h, input_w):
        rgb = _bilinear_resize(rgb, input_h, input_w)
    mean = np.array(_IMG_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(_IMG_STD, dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / std
    return mx.array(rgb)[None]


def _bilinear_resize(img, out_h, out_w):
    import numpy as np

    h, w = img.shape[:2]
    if (h, w) == (out_h, out_w):
        return img
    ys = np.linspace(0, h - 1, out_h)
    xs = np.linspace(0, w - 1, out_w)
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    y1 = y0 + 1
    wy = (ys - y0)[:, None, None]  # (out_h, 1, 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2)
    x1 = x0 + 1
    wx = (xs - x0)[None, :, None]  # (1, out_w, 1)
    a = img[y0[:, None], x0[None, :]]
    b = img[y0[:, None], x1[None, :]]
    c = img[y1[:, None], x0[None, :]]
    d = img[y1[:, None], x1[None, :]]
    return (
        a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy
    ).astype(np.float32)


class DWPose:
    """High-level DWPose wrapper: load weights, detect keypoints (#909)."""

    def __init__(self, model: DWPoseMLX):
        self.model = model
        self.model.eval()

    @classmethod
    def from_pretrained(cls, weights_dir: str | Path | None = None) -> DWPose:
        root = (
            Path(weights_dir)
            if weights_dir
            else Path.home() / ".fusion-mlx" / "models" / "dwpose"
        )
        model = DWPoseMLX()
        st_path = root / "dw-ll_ucoco_384.safetensors"
        if st_path.exists():
            _load_safetensors(model, st_path)
            logger.info("[dwpose] loaded weights from %s", st_path)
        else:
            logger.warning(
                "[dwpose] no weights at %s — running untrained (forward shape only). "
                "Run scripts/convert_dwpose.py to convert dw-ll_ucoco_384.pth.",
                st_path,
            )
        return cls(model)

    def detect(self, frame_bgr):
        """BGR uint8 (H,W,3) → keypoints (133,2) float32 + scores (133,)."""
        import numpy as np

        x = preprocess(frame_bgr)
        simcc_x, simcc_y = self.model(x)
        locs, scores = decode_simcc(simcc_x[0], simcc_y[0])
        locs_np = np.array(locs)
        scores_np = np.array(scores)
        return locs_np, scores_np

    def face_landmarks(self, frame_bgr):
        """BGR frame → 68 face landmarks (68,2) — MuseTalk [23:91] subset."""
        locs, _ = self.detect(frame_bgr)
        return locs[23:91]


def _load_safetensors(model: nn.Module, path: Path):
    """Load a safetensors file into the model, flattening torch-style nested keys."""
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


def face_bbox_from_keypoints(kpts_133, upper_ratio=0.5):
    """Derive face crop bbox from 68 face landmarks [23:91] — MuseTalk style.

    Returns (x1, y1, x2, y2) in frame pixel coords. nose-bridge midpoint (lm[29])
    sets the upper boundary so the crop keeps the lower face (mouth).
    """
    import numpy as np

    face = kpts_133[23:91]
    half_face = face[29]
    min_x = float(face[:, 0].min())
    max_x = float(face[:, 0].max())
    upper = float(half_face[1])
    max_y = float(face[:, 1].max())
    x1, x2 = min_x, max_x
    y1 = upper + (max_y - upper) * (1.0 - upper_ratio)
    y2 = max_y
    return np.array([x1, y1, x2, y2], dtype=np.float32)
