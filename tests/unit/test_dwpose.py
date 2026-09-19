# SPDX-License-Identifier: Apache-2.0
"""#909: DWPose/RTMPose MLX backend tests (synthetic weights — shape/forward parity)."""

import mlx.core as mx
import numpy as np

from fusion_mlx.video.dwpose import (
    ChannelAttention,
    CSPLayer,
    CSPNeXtBackbone,
    CSPNeXtBlock,
    DepthwiseSeparableConv,
    DWPose,
    DWPoseMLX,
    RTMCCBlock,
    RTMCCHead,
    ScaleNorm,
    SPPBottleneck,
    decode_simcc,
    face_bbox_from_keypoints,
    preprocess,
)


def test_backbone_output_shape():
    bb = CSPNeXtBackbone()
    bb.eval()
    x = mx.random.uniform(0, 1, (1, 384, 288, 3))  # NHWC: H=384, W=288
    out = bb(x)
    # 384/32=12 (H), 288/32=9 (W) → (1, 12, 9, 1024) NHWC
    assert out.shape == (1, 12, 9, 1024)


def test_head_output_shape():
    head = RTMCCHead()
    head.eval()
    feats = mx.random.uniform(0, 1, (1, 12, 9, 1024))
    sx, sy = head(feats)
    assert sx.shape == (1, 133, 576)
    assert sy.shape == (1, 133, 768)


def test_full_model_forward():
    model = DWPoseMLX()
    model.eval()
    x = mx.random.uniform(0, 1, (1, 384, 288, 3))
    sx, sy = model(x)
    assert sx.shape == (1, 133, 576)
    assert sy.shape == (1, 133, 768)


def test_decode_simcc_shapes():
    sx = mx.random.uniform(-10, 10, (133, 576))
    sy = mx.random.uniform(-10, 10, (133, 768))
    locs, scores = decode_simcc(sx, sy)
    assert locs.shape == (133, 2)
    assert scores.shape == (133,)


def test_decode_simcc_argmax_correct():
    # Place a clear peak at index 100 on x, index 200 on y for keypoint 0.
    sx_np = np.full((133, 576), -1e4, dtype=np.float32)
    sy_np = np.full((133, 768), -1e4, dtype=np.float32)
    sx_np[0, 100] = 100.0
    sy_np[0, 200] = 100.0
    locs, _ = decode_simcc(mx.array(sx_np), mx.array(sy_np))
    # index 100 → 100/2.0 = 50.0; index 200 → 100.0
    assert locs[0, 0].item() == 50.0
    assert locs[0, 1].item() == 100.0


def test_depthwise_separable_conv():
    dsc = DepthwiseSeparableConv(8, 16, 5)
    dsc.eval()
    x = mx.random.uniform(0, 1, (1, 4, 4, 8))
    out = dsc(x)
    assert out.shape == (1, 4, 4, 16)


def test_cspnext_block_residual():
    blk = CSPNeXtBlock(16, 16)
    blk.eval()
    x = mx.random.uniform(0, 1, (1, 4, 4, 16))
    out = blk(x)
    assert out.shape == (1, 4, 4, 16)


def test_channel_attention_preserves_shape():
    ca = ChannelAttention(16)
    ca.eval()
    x = mx.random.uniform(0, 1, (1, 4, 4, 16))
    out = ca(x)
    assert out.shape == (1, 4, 4, 16)


def test_spp_bottleneck_preserves_spatial():
    spp = SPPBottleneck(16, 16)
    spp.eval()
    x = mx.random.uniform(0, 1, (1, 9, 12, 16))
    out = spp(x)
    assert out.shape[:3] == (1, 9, 12)


def test_csplayer_output():
    csp = CSPLayer(16, 32, num_blocks=2)
    csp.eval()
    x = mx.random.uniform(0, 1, (1, 4, 4, 16))
    out = csp(x)
    assert out.shape == (1, 4, 4, 32)


def test_rtmc_block_output_shape():
    gau = RTMCCBlock(num_token=133, in_dim=256, out_dim=256)
    gau.eval()
    x = mx.random.uniform(0, 1, (1, 133, 256))
    out = gau(x)
    assert out.shape == (1, 133, 256)


def test_scalenorm():
    sn = ScaleNorm(108)
    x = mx.random.uniform(0, 1, (1, 133, 108))
    out = sn(x)
    assert out.shape == (1, 133, 108)


def test_preprocess_output_shape():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    x = preprocess(frame)
    assert x.shape == (1, 384, 288, 3)


def test_preprocess_bgr_to_rgb():
    frame = np.zeros((384, 288, 3), dtype=np.uint8)
    frame[..., 0] = 255  # B=255, G=0, R=0 → after BGR→RGB, R=255
    x = preprocess(frame)
    mx.eval(x)
    # RGB: channel 2 (last) should be the high one (was B=255 → now R).
    assert x[0, 0, 0, 2].item() > x[0, 0, 0, 0].item()


def test_face_bbox_from_keypoints():
    kpts = np.zeros((133, 2), dtype=np.float32)
    kpts[23:91, 0] = np.linspace(10, 100, 68)  # x spread
    kpts[23:91, 1] = np.linspace(20, 80, 68)  # y spread
    bbox = face_bbox_from_keypoints(kpts)
    assert bbox.shape == (4,)
    assert bbox[0] <= bbox[2]
    assert bbox[1] <= bbox[3]


def test_dwpose_wrapper_untrained_forward():
    # DWPose.from_pretrained without weights should still forward (shape-only).
    dw = DWPose.from_pretrained(weights_dir="/nonexistent/path")
    frame = np.random.randint(0, 255, (384, 288, 3), dtype=np.uint8)
    locs, scores = dw.detect(frame)
    assert locs.shape == (133, 2)
    assert scores.shape == (133,)


def test_face_landmarks_subset():
    dw = DWPose.from_pretrained(weights_dir="/nonexistent/path")
    frame = np.random.randint(0, 255, (384, 288, 3), dtype=np.uint8)
    face = dw.face_landmarks(frame)
    assert face.shape == (68, 2)
