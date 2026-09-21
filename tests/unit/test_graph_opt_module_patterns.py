# SPDX-License-Identifier: Apache-2.0
"""#932: module-level patterns — whole-block fusion of ResnetBlock2D-style
blocks whose ops live in a custom __call__ (invisible to window matching).

Parity gate: fused block output == original block output (weights shared),
cosine >= 0.98 per issue #932.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from fusion_mlx.graph_opt import (
    apply_patterns,
    fuse_resnet_block,
    register_module_pattern,
)
from fusion_mlx.graph_opt.passes import _MODULE_PATTERN_REGISTRY
from fusion_mlx.nn_ext import SafeGroupNorm
from fusion_mlx.video.musetalk_mlx.models.unet import ResnetBlock2D as UNetResnet
from fusion_mlx.video.musetalk_mlx.models.vae import ResnetBlock2D as VAEResnet


class BlockList(nn.Module):
    # Mimics DownBlock2D/UpBlock2D: resnets is a plain list of blocks.
    def __init__(self, blocks):
        super().__init__()
        self.resnets = blocks


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def test_module_pattern_registry_has_resnet_block():
    assert "resnet_block_gn_silu_conv" in _MODULE_PATTERN_REGISTRY


def test_vae_block_fusion_parity_and_weight_sharing():
    blk = VAEResnet(32, 64)
    x = mx.random.normal((1, 16, 16, 32))
    mx.eval(blk.parameters())
    ref = blk(x)
    mx.eval(ref)
    fused = fuse_resnet_block(blk)
    out = fused(x)
    mx.eval(out)
    assert _cosine(out, ref) >= 0.98
    # weights shared, not copied
    assert fused.gn1_conv1.conv is blk.conv1
    assert fused.gn2_conv2.conv is blk.conv2
    assert isinstance(fused.gn1_conv1.groupnorm, SafeGroupNorm)
    assert fused.gn1_conv1.groupnorm.eps == 1e-6
    assert fused.gn1_conv1.groupnorm.pytorch_compatible is True
    assert fused.conv_shortcut is blk.conv_shortcut


def test_unet_block_fusion_parity_with_temb():
    blk = UNetResnet(32, 64, time_dim=64)
    x = mx.random.normal((1, 16, 16, 32))
    temb = mx.random.normal((1, 64))
    mx.eval(blk.parameters())
    ref = blk(x, temb)
    mx.eval(ref)
    fused = fuse_resnet_block(blk)
    out = fused(x, temb)
    mx.eval(out)
    assert _cosine(out, ref) >= 0.98
    assert fused.time_emb_proj is blk.time_emb_proj


def test_apply_patterns_module_mode_rewrites_blocks():
    holder = BlockList([VAEResnet(32, 32), VAEResnet(32, 32)])
    n = apply_patterns(holder, enable_module_patterns=True)
    assert n == 2
    assert [type(b).__name__ for b in holder.resnets] == [
        "_FusedResnetBlock",
        "_FusedResnetBlock",
    ]


def test_apply_patterns_module_mode_off_by_default():
    holder = BlockList([VAEResnet(32, 32)])
    n = apply_patterns(holder)
    assert n == 0
    assert type(holder.resnets[0]).__name__ == "ResnetBlock2D"


def test_apply_patterns_module_mode_nested_list_descends():
    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = BlockList([UNetResnet(32, 32, time_dim=64)])

    o = Outer()
    n = apply_patterns(o, enable_module_patterns=True)
    assert n == 1
    fused = o.down.resnets[0]
    x = mx.random.normal((1, 16, 16, 32))
    temb = mx.random.normal((1, 64))
    mx.eval(fused.parameters())
    out = fused(x, temb)
    mx.eval(out)
    assert out.shape == (1, 16, 16, 32)


def test_fused_block_preserves_shortcut_only_when_needed():
    same_ch = VAEResnet(32, 32)
    fused_same = fuse_resnet_block(same_ch)
    assert fused_same.conv_shortcut is None
    out = fused_same(mx.zeros((1, 16, 16, 32)))
    mx.eval(out)
    assert out.shape == (1, 16, 16, 32)


def test_register_module_pattern_custom():
    calls = {"n": 0}

    def match(mod):
        return isinstance(mod, VAEResnet)

    def rewrite(mod):
        calls["n"] += 1
        return fuse_resnet_block(mod)

    register_module_pattern("test_custom_932", match, rewrite)
    # registry is first-match-wins; isolate the custom pattern from the builtin
    saved = {
        k: v for k, v in _MODULE_PATTERN_REGISTRY.items() if k != "test_custom_932"
    }
    try:
        for k in saved:
            _MODULE_PATTERN_REGISTRY.pop(k)
        holder = BlockList([VAEResnet(32, 32)])
        assert apply_patterns(holder, enable_module_patterns=True) == 1
        assert calls["n"] == 1
    finally:
        for k in saved:
            _MODULE_PATTERN_REGISTRY[k] = saved[k]
        _MODULE_PATTERN_REGISTRY.pop("test_custom_932", None)
