# SPDX-License-Identifier: Apache-2.0
# DWPose / RTMPose-L MLX backend (#909).
#
# Pure-MLX port of the RTMPose-L whole-body pose model (CSPNeXt-L backbone +
# RTMCCHead SimCC coordinate-classification head) used by MuseTalk's face-landmark
# preprocessing. 133 COCO-WholeBody keypoints (17 body + 6 foot + 68 face [23:91]
# + 42 hand). Checkpoint dw-ll_ucoco_384.pth (yzd-v/DWPose on HF, ~388MB) is
# converted offline to safetensors (no torch at runtime — PRD gate).
#
# Architecture mirrors the mmpose source checkpoint key tree EXACTLY so weights
# load strict=True (no remap at runtime): backbone.stem.{0,1,2}.{conv,bn},
# backbone.stage{1..4}.{0..2}.{conv,bn | main_conv/short_conv/final_conv/blocks/attention},
# head.{final_layer, mlp.0/1, gau.{ln,uv,gamma,beta,o,res_scale}, cls_x, cls_y}.
#
# MLX Conv2d weight layout is OHWI (out, kh, kw, in) — convert script transposes
# torch OIHW (out, in, kh, kw) via permute(0,2,3,1). BatchNorm params
# (weight/bias/running_mean/running_var) map 1:1; num_batches_tracked dropped.

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


class ConvBNAct(nn.Module):
    """Conv2d (no bias) + BatchNorm + SiLU — mirrors mmpose ConvModule (conv/bn keys)."""

    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c,
            out_c,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm(out_c)

    def __call__(self, x):
        return nn.silu(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise (groups=C, k5) + pointwise (k1), each ConvBNAct — keys depthwise_conv/pointwise_conv."""

    def __init__(self, in_c, out_c, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise_conv = ConvBNAct(
            in_c, in_c, kernel_size, padding=pad, groups=in_c
        )
        self.pointwise_conv = ConvBNAct(in_c, out_c, 1)

    def __call__(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


DepthwiseSeparableConv = DepthwiseSeparableConvModule


class CSPNeXtBlock(nn.Module):
    """conv1 (3×3, in→in) + conv2 (DepthwiseSeparable, in→out) + identity residual."""

    def __init__(self, in_c, out_c, expand=0.5, add_identity=True):
        super().__init__()
        self.conv1 = ConvBNAct(in_c, in_c, 3, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(in_c, out_c)
        self.add_identity = add_identity

    def __call__(self, x):
        # ConvBNAct already applies SiLU — no extra activation between conv1/conv2
        h = self.conv2(self.conv1(x))
        # ONNX ground truth (dw-ll_ucoco_384): SPP stage's CSPLayer blocks have
        # NO residual add (17 Adds for 18 blocks — stage4 contributes none)
        if not self.add_identity or x.shape[-1] != h.shape[-1]:
            return h
        return x + h


class ChannelAttention(nn.Module):
    """AdaptiveAvgPool → 1×1 conv (fc) → Hardsigmoid channel reweight."""

    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1)

    def _hardsigmoid(self, x):
        return mx.minimum(mx.maximum(x + 3.0, 0.0), 6.0) / 6.0

    def __call__(self, x):
        pooled = mx.mean(x, axis=(1, 2), keepdims=True)
        return self._hardsigmoid(self.fc(pooled)) * x


class CSPLayer(nn.Module):
    """main_conv + N×CSPNeXtBlock + short_conv → concat → attention → final_conv."""

    def __init__(self, in_c, out_c, num_blocks=3, expand=0.5, add_identity=True):
        super().__init__()
        mid = int(out_c * expand)
        self.main_conv = ConvBNAct(in_c, mid, 1)
        self.blocks = [
            CSPNeXtBlock(mid, mid, expand, add_identity=add_identity)
            for _ in range(num_blocks)
        ]
        self.short_conv = ConvBNAct(in_c, mid, 1)
        self.attention = ChannelAttention(mid * 2)
        self.final_conv = ConvBNAct(mid * 2, out_c, 1)

    def __call__(self, x):
        main = self.main_conv(x)
        for blk in self.blocks:
            main = blk(main)
        short = self.short_conv(x)
        # mmpose: x_final = torch.cat((x_main, x_short), dim=1) — main first
        cat = mx.concatenate([main, short], axis=-1)
        cat = self.attention(cat)
        return self.final_conv(cat)


class SPPBottleneck(nn.Module):
    """conv1 (1×1) → parallel MaxPool(k=5,9,13) → concat → conv2 (1×1)."""

    def __init__(self, in_c, out_c, kernels=(5, 9, 13)):
        super().__init__()
        mid = in_c // 2
        self.conv1 = ConvBNAct(in_c, mid, 1)
        self.pools = [nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernels]
        self.conv2 = ConvBNAct(mid * (1 + len(kernels)), out_c, 1)

    def __call__(self, x):
        x = self.conv1(x)
        outs = [x] + [p(x) for p in self.pools]
        cat = mx.concatenate(outs, axis=-1)
        # conv2 is ConvBNAct (act included) — no extra silu (mmpose has none)
        return self.conv2(cat)


_CSPNEXT_P5 = [
    (64, 128, 3),
    (128, 256, 6),
    (256, 512, 6),
    (512, 1024, 3),
]


class CSPNeXtBackbone(nn.Module):
    """CSPNeXt-L (P5) — stem (3×ConvBNAct) + 4 stages, returns stage4 [B,1024,12,9]."""

    def __init__(self, arch="P5", widen=1.0):
        super().__init__()
        self.stem = [
            ConvBNAct(3, int(32 * widen), 3, stride=2, padding=1),
            ConvBNAct(int(32 * widen), int(32 * widen), 3, padding=1),
            ConvBNAct(int(32 * widen), int(64 * widen), 3, padding=1),
        ]
        self.stage1 = self._make_stage(64, 128, 3, widen, use_spp=False)
        self.stage2 = self._make_stage(128, 256, 6, widen, use_spp=False)
        self.stage3 = self._make_stage(256, 512, 6, widen, use_spp=False)
        self.stage4 = self._make_stage(512, 1024, 3, widen, use_spp=True)

    @staticmethod
    def _make_stage(in_c, out_c, n_blocks, widen, use_spp):
        in_c = int(in_c * widen)
        out_c = int(out_c * widen)
        layers = [ConvBNAct(in_c, out_c, 3, stride=2, padding=1)]
        if use_spp:
            layers.append(SPPBottleneck(out_c, out_c))
        layers.append(CSPLayer(out_c, out_c, n_blocks, add_identity=not use_spp))
        return layers

    def __call__(self, x):
        for blk in self.stem:
            x = blk(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for blk in stage:
                x = blk(x)
        return x  # [B, 1024, 12, 9]


class ScaleNorm(nn.Module):
    """L2-norm × g (scalar) — RTMCCHead mlp.0 / gau.ln (param name `g`)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = mx.ones((1,))
        self.eps = eps

    def __call__(self, x):
        norm = mx.sqrt(
            mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps
        )
        return self.g * (x / norm)


class ResScale(nn.Module):
    """Per-channel residual scale (param `scale`, shape (out_dim,)) — gau.res_scale."""

    def __init__(self, dim):
        super().__init__()
        self.scale = mx.ones((dim,))


class RTMCCBlock(nn.Module):
    """Gated Attention Unit (use_rel_bias=False, pos_enc=False → pure tensor ops)."""

    def __init__(self, num_token, in_dim, out_dim, s=128, expansion=2):
        super().__init__()
        self.num_token = num_token
        self.s = s
        e = in_dim * expansion
        self.ln = ScaleNorm(in_dim)
        self.uv = nn.Linear(in_dim, 2 * e + s, bias=False)
        self.gamma = mx.ones((2, s))
        self.beta = mx.zeros((2, s))
        self.o = nn.Linear(e, out_dim, bias=False)
        self.res_scale = ResScale(out_dim)

    def __call__(self, x):
        h = self.ln(x)
        uv = nn.silu(self.uv(h))
        e = (uv.shape[-1] - self.s) // 2  # expansion dim
        u = uv[..., :e]
        v = uv[..., e : e * 2]
        base = uv[..., e * 2 :]
        base = base[:, :, None, :] * self.gamma + self.beta  # (B,K,2,s)
        q, k = base[..., 0, :], base[..., 1, :]  # each (B,K,s)
        qk = mx.matmul(q, mx.transpose(k, (0, 2, 1)))  # (B,K,K)
        kernel = mx.square(mx.maximum(qk / (self.s**0.5), 0.0))
        out = u * mx.matmul(kernel, v)  # (B,K,e)
        out = self.o(out)
        # mmpose: res_scale scales the SHORTCUT, main branch unscaled
        return x * self.res_scale.scale + out


class RTMCCHead(nn.Module):
    """SimCC head: final_layer conv → flatten → mlp(ScaleNorm+Linear) → gau → cls_x/cls_y."""

    def __init__(
        self, in_channels=1024, out_channels=133, in_featuremap_size=(12, 9), s=128
    ):
        super().__init__()
        fh, fw = in_featuremap_size
        self.flat_dim = fh * fw
        self.final_layer = nn.Conv2d(in_channels, out_channels, 7, padding=3)
        self.mlp = [ScaleNorm(self.flat_dim), nn.Linear(self.flat_dim, 256, bias=False)]
        self.gau = RTMCCBlock(out_channels, 256, 256, s=s)
        self.cls_x = nn.Linear(256, int(_INPUT_W * _SIMCC_SPLIT_RATIO), bias=False)
        self.cls_y = nn.Linear(256, int(_INPUT_H * _SIMCC_SPLIT_RATIO), bias=False)

    def __call__(self, feats):
        x = self.final_layer(feats)  # (B, 12, 9, 133)
        b, fh, fw, c = x.shape
        x = x.reshape(b, fh * fw, c).transpose(0, 2, 1)  # (B, 133, 108)
        x = self.mlp[1](self.mlp[0](x))
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
    x_loc = mx.argmax(simcc_x, axis=-1)
    y_loc = mx.argmax(simcc_y, axis=-1)
    locs = mx.stack([x_loc, y_loc], axis=-1).astype(mx.float32) / split_ratio
    x_score = mx.max(simcc_x, axis=-1)
    y_score = mx.max(simcc_y, axis=-1)
    scores = mx.minimum(x_score, y_score)
    return locs, scores


def preprocess(frame_bgr, input_w=_INPUT_W, input_h=_INPUT_H):
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
    # half-pixel centers (cv2.INTER_LINEAR / align_corners=False semantics)
    ys = (np.arange(out_h) + 0.5) * (h / out_h) - 0.5
    xs = (np.arange(out_w) + 0.5) * (w / out_w) - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    y1 = y0 + 1
    wy = np.clip(ys - y0, 0.0, 1.0)[:, None, None]
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2)
    x1 = x0 + 1
    wx = np.clip(xs - x0, 0.0, 1.0)[None, :, None]
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
        import numpy as np

        frame_h, frame_w = frame_bgr.shape[:2]
        x = preprocess(frame_bgr)
        simcc_x, simcc_y = self.model(x)
        locs, scores = decode_simcc(simcc_x[0], simcc_y[0])
        locs = np.array(locs)
        # network input is (W=288, H=384); map back to original frame space (#916)
        locs[:, 0] *= frame_w / _INPUT_W
        locs[:, 1] *= frame_h / _INPUT_H
        return locs, np.array(scores)

    def face_landmarks(self, frame_bgr):
        locs, _ = self.detect(frame_bgr)
        return locs[23:91]


def _flatten_param_keys(d, prefix=""):
    """Recursively flatten an MLX param tree to dotted key paths."""
    keys = []
    if isinstance(d, dict):
        for k, v in d.items():
            keys.extend(_flatten_param_keys(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            keys.extend(_flatten_param_keys(v, f"{prefix}{i}."))
    else:
        keys.append(prefix[:-1])
    return keys


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

    expected = set(_flatten_param_keys(model.parameters()))
    got = set(weights.keys())
    missing = expected - got
    extra = got - expected
    if missing or extra:
        raise ValueError(
            f"[dwpose] weight key mismatch ({path}):\n"
            f"  missing ({len(missing)}): {sorted(missing)[:10]}\n"
            f"  extra   ({len(extra)}): {sorted(extra)[:10]}\n"
            f"Re-run scripts/convert_dwpose.py against the source .pth."
        )
    model.load_weights(list(weights.items()))


def face_bbox_from_keypoints(kpts_133, upper_ratio=0.5):
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
