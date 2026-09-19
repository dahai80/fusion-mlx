# SPDX-License-Identifier: Apache-2.0
"""#910/#915: BiSeNet face-parsing MLX backend tests (synthetic weights — shape/forward)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from fusion_mlx.video.face_parsing import (
    ARM,
    FFM,
    BasicBlock,
    BiSeNetMLX,
    BiSeNetOutput,
    ContextPath,
    ConvBNReLU,
    FaceParsing,
    ResNet18Backbone,
    decode_parsing,
    preprocess,
)


def test_resnet18_backbone_shapes():
    bb = ResNet18Backbone()
    bb.eval()
    x = mx.random.uniform(0, 1, (1, 512, 512, 3))
    feat8, feat16, feat32 = bb(x)
    assert feat8.shape == (1, 64, 64, 128)  # 1/8, 128ch
    assert feat16.shape == (1, 32, 32, 256)  # 1/16, 256ch
    assert feat32.shape == (1, 16, 16, 512)  # 1/32, 512ch


def test_context_path_shapes():
    cp = ContextPath()
    cp.eval()
    x = mx.random.uniform(0, 1, (1, 512, 512, 3))
    feat8, feat_cp8, feat_cp16 = cp(x)
    assert feat8.shape == (1, 64, 64, 128)
    assert feat_cp8.shape == (1, 64, 64, 128)
    assert feat_cp16.shape == (1, 32, 32, 128)


def test_arm_preserves_shape():
    arm = ARM(64, 32)
    arm.eval()
    x = mx.random.uniform(0, 1, (1, 8, 8, 64))
    out = arm(x)
    assert out.shape == (1, 8, 8, 32)


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
    # Auto-downsample when channels/stride change — list [conv, bn] (keys .0/.1).
    blk = BasicBlock(32, 64, stride=2)
    blk.eval()
    x = mx.random.uniform(0, 1, (1, 8, 8, 32))
    out = blk(x)
    assert out.shape == (1, 4, 4, 64)
    flat = [k for k, _ in nn.utils.tree_flatten(blk.parameters())]
    assert "downsample.0.weight" in flat
    assert "downsample.1.weight" in flat


def test_conv_bn_relu():
    m = ConvBNReLU(8, 16, ks=3, stride=1, padding=1)
    m.eval()
    x = mx.random.uniform(0, 1, (1, 4, 4, 8))
    out = m(x)
    assert out.shape == (1, 4, 4, 16)


def test_conv_bn_relu_no_bias():
    m = ConvBNReLU(8, 16)
    flat = [k for k, _ in nn.utils.tree_flatten(m.parameters())]
    conv_bias = [k for k in flat if k.endswith("conv.bias")]
    assert not conv_bias


def test_bise_output_shape():
    head = BiSeNetOutput(256, 256, 19)
    head.eval()
    x = mx.random.uniform(0, 1, (1, 64, 64, 256))
    out = head(x)
    assert out.shape == (1, 64, 64, 19)


def test_bisenet_full_forward():
    model = BiSeNetMLX()
    model.eval()
    x = mx.random.uniform(0, 1, (1, 512, 512, 3))
    out, out16, out32 = model(x)
    assert out.shape == (1, 512, 512, 19)
    assert out16.shape == (1, 512, 512, 19)
    assert out32.shape == (1, 512, 512, 19)


def test_bisenet_forward_smaller_input():
    model = BiSeNetMLX()
    model.eval()
    x = mx.random.uniform(0, 1, (1, 256, 256, 3))
    out, _, _ = model(x)
    assert out.shape == (1, 256, 256, 19)


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


def test_load_safetensors_strict_mismatch():
    # Mismatched weights must fail loud (ValueError), not silently skip.
    import os
    import tempfile
    from pathlib import Path

    from safetensors.numpy import save_file

    from fusion_mlx.video.face_parsing import _load_safetensors

    model = BiSeNetMLX()
    path = Path(tempfile.mktemp(suffix=".safetensors"))
    try:
        save_file({"wrong.key": np.zeros((4,), dtype=np.float32)}, str(path))
        with pytest.raises(ValueError, match="key mismatch"):
            _load_safetensors(model, path)
    finally:
        os.unlink(path)
