# SPDX-License-Identifier: Apache-2.0
"""#910: BiSeNet face-parsing MLX backend tests (synthetic weights — shape/forward)."""

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.video.face_parsing import (
    BiSeNetMLX,
    FaceParsing,
    ResNet18Backbone,
    SpatialPath,
    ARM,
    FFM,
    ConvBNReLU,
    BasicBlock,
    preprocess,
    decode_parsing,
)


def test_resnet18_backbone_shapes():
    bb = ResNet18Backbone()
    bb.eval()
    x = mx.random.uniform(0, 1, (1, 512, 512, 3))
    c3, c4 = bb(x)
    assert c3.shape[0] == 1
    assert c3.shape[-1] == 256  # 1/16, 256ch
    assert c4.shape[-1] == 512  # 1/32, 512ch


def test_spatial_path_output():
    sp = SpatialPath()
    sp.eval()
    x = mx.random.uniform(0, 1, (1, 512, 512, 3))
    out = sp(x)
    assert out.shape[-1] == 256
    # 1/8 spatial: 512/8 = 64
    assert out.shape[1] == 64


def test_arm_preserves_shape():
    arm = ARM(64, 64)
    arm.eval()
    x = mx.random.uniform(0, 1, (1, 8, 8, 64))
    out = arm(x)
    assert out.shape == (1, 8, 8, 64)


def test_ffm_output_shape():
    ffm = FFM(64 + 32, 128)
    ffm.eval()
    a = mx.random.uniform(0, 1, (1, 8, 8, 64))
    b = mx.random.uniform(0, 1, (1, 8, 8, 32))
    out = ffm(a, b)
    assert out.shape == (1, 8, 8, 128)


def test_basic_block_residual():
    blk = BasicBlock(32, 32)
    blk.eval()
    x = mx.random.uniform(0, 1, (1, 8, 8, 32))
    out = blk(x)
    assert out.shape == (1, 8, 8, 32)


def test_basic_block_downsample():
    blk = BasicBlock(32, 64, stride=2, downsample=True)
    blk.eval()
    x = mx.random.uniform(0, 1, (1, 8, 8, 32))
    out = blk(x)
    assert out.shape == (1, 4, 4, 64)


def test_conv_bn_relu():
    m = ConvBNReLU(8, 16, kernel=3, stride=1, padding=1)
    m.eval()
    x = mx.random.uniform(0, 1, (1, 4, 4, 8))
    out = m(x)
    assert out.shape == (1, 4, 4, 16)


def test_bisenet_full_forward():
    model = BiSeNetMLX()
    model.eval()
    x = mx.random.uniform(0, 1, (1, 512, 512, 3))
    logits = model(x)
    assert logits.shape == (1, 512, 512, 19)


def test_bisenet_forward_smaller_input():
    model = BiSeNetMLX()
    model.eval()
    x = mx.random.uniform(0, 1, (1, 256, 256, 3))
    logits = model(x)
    assert logits.shape == (1, 256, 256, 19)


def test_preprocess_output_shape():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    x = preprocess(frame)
    assert x.shape == (1, 512, 512, 3)


def test_decode_parsing_shapes():
    logits = np.random.randn(19, 64, 64).astype(np.float32)
    labels, mask = decode_parsing(logits)
    assert labels.shape == (64, 64)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.bool_


def test_decode_parsing_face_classes():
    # All-skin (class 1) → mask all True.
    logits = np.full((19, 8, 8), -1e4, dtype=np.float32)
    logits[1] = 100.0  # class 1 = skin
    labels, mask = decode_parsing(logits)
    assert labels.all() == 1
    assert mask.all()


def test_face_parsing_wrapper_untrained():
    fp = FaceParsing.from_pretrained(weights_dir="/nonexistent/path")
    frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    labels, mask = fp.parse(frame)
    # parse returns at model input size (512×512).
    assert labels.shape == (512, 512)
    assert mask.shape == (512, 512)


def test_face_mask_output():
    fp = FaceParsing.from_pretrained(weights_dir="/nonexistent/path")
    frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = fp.face_mask(frame)
    assert mask.shape == (512, 512)
    assert mask.dtype == np.bool_
