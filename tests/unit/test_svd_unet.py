# SPDX-License-Identifier: Apache-2.0
# Unit tests for SVD UNet Conv3d fix: verify NCDHW format works end-to-end.
# Tests exercise the custom Conv3d (matmul-based _conv3d_core) and UNet forward
# without real weights — small dims to keep fast.
# Importers: pytest standalone; Affected API: Conv3d, SVDTemporalUNet forward;
# User instruction: "落地SVD UNet, 真实的 SVD 权重可用时进行"

import pytest

mlx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import mlx.core as mx
import mlx.nn as nn


class TestConv3d:
    def test_conv3d_basic_shape(self):
        from fusion_mlx.video.svd.unet import Conv3d

        conv = Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        x = mx.random.normal(shape=(1, 8, 4, 16, 16))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 16, 4, 16, 16)

    def test_conv3d_stride2_shape(self):
        from fusion_mlx.video.svd.unet import Conv3d

        conv = Conv3d(16, 16, kernel_size=3, stride=2, padding=1)
        x = mx.random.normal(shape=(1, 16, 4, 16, 16))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 16, 2, 8, 8)

    def test_conv3d_1x1_kernel(self):
        from fusion_mlx.video.svd.unet import Conv3d

        conv = Conv3d(8, 16, kernel_size=1, stride=1, padding=0)
        x = mx.random.normal(shape=(1, 8, 4, 16, 16))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 16, 4, 16, 16)

    def test_conv3d_weight_format_ncdhw(self):
        from fusion_mlx.video.svd.unet import Conv3d

        conv = Conv3d(4, 8, kernel_size=3, stride=1, padding=1)
        # Weight shape must be (OC, IC, KT, KH, KW) — PyTorch convention
        assert conv.weight.shape == (8, 4, 3, 3, 3)

    def test_conv3d_preserves_ncdhw(self):
        from fusion_mlx.video.svd.unet import Conv3d

        conv = Conv3d(4, 4, kernel_size=3, stride=1, padding=1)
        x = mx.random.normal(shape=(2, 4, 3, 8, 8))
        out = conv(x)
        mx.eval(out)
        # Output must still be NCDHW: (B, C, T, H, W)
        assert out.shape == (2, 4, 3, 8, 8)


class TestTemporalConv3d:
    def test_temporal_conv_shape(self):
        from fusion_mlx.video.svd.unet import TemporalConv3d

        tc = TemporalConv3d(32, 32, kernel_size=3)
        x = mx.random.normal(shape=(1, 32, 4, 8, 8))
        out = tc(x)
        mx.eval(out)
        assert out.shape == (1, 32, 4, 8, 8)


class TestResnetBlock:
    def test_resnet_same_channels(self):
        from fusion_mlx.video.svd.unet import ResnetBlock

        block = ResnetBlock(32)
        x = mx.random.normal(shape=(1, 32, 4, 8, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 32, 4, 8, 8)

    def test_resnet_channel_change(self):
        from fusion_mlx.video.svd.unet import ResnetBlock

        block = ResnetBlock(32, 64)
        x = mx.random.normal(shape=(1, 32, 4, 8, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 64, 4, 8, 8)

    def test_resnet_with_temb(self):
        from fusion_mlx.video.svd.unet import ResnetBlock

        block = ResnetBlock(32)
        x = mx.random.normal(shape=(1, 32, 4, 8, 8))
        temb = mx.random.normal(shape=(1, 32, 1, 1, 1))
        out = block(x, temb=temb)
        mx.eval(out)
        assert out.shape == (1, 32, 4, 8, 8)


class TestDownBlock:
    def test_downsample_shape(self):
        from fusion_mlx.video.svd.unet import DownBlock

        block = DownBlock(32, 32, num_layers=1, downsample=True)
        x = mx.random.normal(shape=(1, 32, 4, 16, 16))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 32, 2, 8, 8)

    def test_no_downsample_shape(self):
        from fusion_mlx.video.svd.unet import DownBlock

        block = DownBlock(32, 64, num_layers=1, downsample=False)
        x = mx.random.normal(shape=(1, 32, 4, 8, 8))
        out = block(x)
        mx.eval(out)
        # SVDUNetBlock(32, out_dim=64) projects to 64 channels
        assert out.shape == (1, 64, 4, 8, 8)


class TestUpBlock:
    def test_upsample_shape(self):
        from fusion_mlx.video.svd.unet import UpBlock

        block = UpBlock(64, 32, num_layers=1, upsample=True)
        x = mx.random.normal(shape=(1, 64, 2, 8, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 32, 4, 16, 16)

    def test_no_upsample_shape(self):
        from fusion_mlx.video.svd.unet import UpBlock

        block = UpBlock(64, 32, num_layers=1, upsample=False)
        x = mx.random.normal(shape=(1, 64, 4, 8, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 32, 4, 8, 8)


class TestSVDTemporalUNet:
    def _make_small_unet(self):
        from fusion_mlx.video.svd.unet import SVDTemporalUNet

        return SVDTemporalUNet(
            in_channels=8,
            out_channels=4,
            context_dim=64,
            dims=(32, 64, 64, 64),
            num_heads=4,
        )

    def test_forward_no_timestep(self):
        unet = self._make_small_unet()
        # T=8: down 8->4->2->1 (3 stride-2 downs), up 1->2->4->8 (3 ups)
        x = mx.random.normal(shape=(1, 8, 8, 32, 32))
        out = unet(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 32, 32)

    def test_forward_with_timestep(self):
        unet = self._make_small_unet()
        x = mx.random.normal(shape=(1, 8, 8, 32, 32))
        t = mx.array([500.0])
        out = unet(x, timestep=t)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 32, 32)

    def test_forward_with_context(self):
        unet = self._make_small_unet()
        x = mx.random.normal(shape=(1, 8, 8, 32, 32))
        t = mx.array([500.0])
        ctx = mx.random.normal(shape=(1, 6, 64))
        out = unet(x, timestep=t, context=ctx)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 32, 32)

    def test_forward_cfg_batch(self):
        unet = self._make_small_unet()
        x = mx.random.normal(shape=(2, 8, 8, 32, 32))
        t = mx.array([500.0, 500.0])
        ctx = mx.random.normal(shape=(2, 6, 64))
        out = unet(x, timestep=t, context=ctx)
        mx.eval(out)
        assert out.shape == (2, 4, 8, 32, 32)

    def test_conv_in_out_weight_format(self):
        unet = self._make_small_unet()
        # conv_in weight: (OC, IC, KT, KH, KW) — PyTorch convention
        assert unet.conv_in.weight.shape == (32, 8, 3, 3, 3)
        assert unet.conv_out.weight.shape == (4, 32, 3, 3, 3)

    def test_no_nn_conv3d_remaining(self):
        import fusion_mlx.video.svd.unet as unet_mod

        for name in dir(unet_mod):
            obj = getattr(unet_mod, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, nn.Conv3d)
                and obj is not nn.Conv3d
            ):
                pytest.fail(f"nn.Conv3d subclass still in module: {name}")
