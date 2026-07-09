# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from fusion_mlx.video.ltx_video_legacy.vae import (
    OURS_VAE_CONFIG,
    CausalConv3d,
    Decoder,
    DepthToSpaceUpsample,
    LayerNorm,
    LTVideoVAE,
    PixelNorm,
    ResnetBlock3D,
    VAEConfig,
    _flatten_keys,
    _pixel_shuffle_nd,
)

# ---------------------------------------------------------------------------
# numpy oracle references (torch-free). Weights are authored in PyTorch layout
# (out,in,kD,kH,kW) and transposed to MLX layout at load time, exactly like the
# real from_pretrained path, so the port is checked against an independent impl.
# ---------------------------------------------------------------------------

RTOL = 1e-3
ATOL = 1e-4
RTOL_DEC = 5e-3
ATOL_DEC = 1e-3


def np_pixel_norm(x, eps=1e-8):
    xf = x.astype(np.float32)
    return (xf / np.sqrt(np.mean(xf * xf, axis=1, keepdims=True) + eps)).astype(x.dtype)


def np_layernorm(x, w=None, b=None, eps=1e-6):
    xt = np.transpose(x, (0, 2, 3, 4, 1)).astype(np.float32)
    mu = np.mean(xt, axis=-1, keepdims=True)
    var = np.mean((xt - mu) ** 2, axis=-1, keepdims=True)
    out = (xt - mu) / np.sqrt(var + eps)
    if w is not None:
        out = out * w.astype(np.float32)
    if b is not None:
        out = out + b.astype(np.float32)
    return np.transpose(out, (0, 4, 1, 2, 3)).astype(x.dtype)


def np_silu(x):
    xf = x.astype(np.float32)
    return (xf / (1.0 + np.exp(-xf))).astype(x.dtype)


def np_pixel_shuffle_nd(x, p1, p2, p3):
    B, Cp, D, H, W = x.shape
    Cc = Cp // (p1 * p2 * p3)
    x = x.reshape(B, Cc, p1, p2, p3, D, H, W)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(B, Cc, D * p1, H * p2, W * p3)


def np_conv3d(x, w_torch, b, k, causal):
    # x (B,C,D,H,W); w_torch (out,in,k,k,k); b (out,) or None
    if causal:
        pad = np.repeat(x[:, :, :1, :, :], k - 1, axis=2)
        x = np.concatenate([pad, x], axis=2)
    else:
        n = (k - 1) // 2
        fp = np.repeat(x[:, :, :1, :, :], n, axis=2)
        lp = np.repeat(x[:, :, -1:, :, :], n, axis=2)
        x = np.concatenate([fp, x, lp], axis=2)
    sp = k // 2
    x = np.pad(x, ((0, 0), (0, 0), (0, 0), (sp, sp), (sp, sp)), mode="constant")
    win = np.lib.stride_tricks.sliding_window_view(x, (k, k, k), axis=(2, 3, 4))
    out = np.einsum("bcdhwijk,ocijk->bodhw", win, w_torch, optimize=True)
    if b is not None:
        out = out + b.reshape(1, -1, 1, 1, 1)
    return out


def _load_torch(module, np_weights):
    pairs = []
    for k, v in np_weights.items():
        va = mx.array(v)
        if va.ndim == 5:
            va = va.transpose(0, 2, 3, 4, 1)
        pairs.append((k, va.astype(mx.float32)))
    module.load_weights(pairs, strict=False)
    mx.eval(module.parameters())
    return module


# ---------------------------------------------------------------------------
# component oracle tests
# ---------------------------------------------------------------------------


class TestPixelNorm:
    def test_matches_numpy(self):
        x = np.random.randn(2, 4, 3, 5, 5).astype(np.float32)
        m = PixelNorm()
        out = m(mx.array(x))
        np.testing.assert_allclose(
            np.array(out), np_pixel_norm(x), rtol=RTOL, atol=ATOL
        )


class TestLayerNorm:
    def test_matches_numpy(self):
        x = np.random.randn(1, 6, 2, 4, 4).astype(np.float32)
        w = np.random.randn(6).astype(np.float32)
        b = np.random.randn(6).astype(np.float32)
        m = LayerNorm(6)
        _load_torch(m, {"norm.weight": w, "norm.bias": b})
        out = m(mx.array(x))
        np.testing.assert_allclose(
            np.array(out), np_layernorm(x, w, b), rtol=RTOL, atol=ATOL
        )


class TestPixelShuffle:
    def test_matches_numpy(self):
        x = np.random.randn(1, 24, 2, 4, 4).astype(np.float32)  # 24 = 3*2*2*2
        out = _pixel_shuffle_nd(mx.array(x), 2, 2, 2)
        np.testing.assert_allclose(
            np.array(out), np_pixel_shuffle_nd(x, 2, 2, 2), rtol=RTOL, atol=ATOL
        )
        assert out.shape == (1, 3, 4, 8, 8)


class TestCausalConv3d:
    def _run(self, causal):
        k = 3
        x = np.random.randn(2, 2, 3, 5, 5).astype(np.float32)
        w = np.random.randn(4, 2, k, k, k).astype(np.float32) * 0.1
        b = np.random.randn(4).astype(np.float32) * 0.01
        m = CausalConv3d(2, 4, kernel_size=k)
        _load_torch(m, {"conv.weight": w, "conv.bias": b})
        out = m(mx.array(x), causal=causal)
        exp = np_conv3d(x, w, b, k, causal)
        np.testing.assert_allclose(np.array(out), exp, rtol=RTOL, atol=ATOL)
        assert out.shape == (2, 4, 3, 5, 5)

    def test_causal(self):
        self._run(True)

    def test_non_causal(self):
        self._run(False)

    def test_kernel1_no_temporal_pad(self):
        x = np.random.randn(1, 2, 2, 3, 3).astype(np.float32)
        w = np.random.randn(3, 2, 1, 1, 1).astype(np.float32) * 0.1
        b = np.random.randn(3).astype(np.float32) * 0.01
        m = CausalConv3d(2, 3, kernel_size=1)
        _load_torch(m, {"conv.weight": w, "conv.bias": b})
        out = m(mx.array(x), causal=True)
        # k=1 -> no pad, pure 1x1x1 conv over channels
        exp = np.einsum("bcdhw,oc->bodhw", x, w[:, :, 0, 0, 0]) + b.reshape(
            1, -1, 1, 1, 1
        )
        np.testing.assert_allclose(np.array(out), exp, rtol=RTOL, atol=ATOL)


def np_resnet_block(x, w, causal, in_ch, out_ch, k=3):
    h = np_pixel_norm(x)
    h = np_silu(h)
    h = np_conv3d(h, w["conv1.conv.weight"], w["conv1.conv.bias"], k, causal)
    h = np_pixel_norm(h)
    h = np_silu(h)
    h = np_conv3d(h, w["conv2.conv.weight"], w["conv2.conv.bias"], k, causal)
    if in_ch != out_ch:
        s = np_layernorm(x, w["norm3.norm.weight"], w["norm3.norm.bias"])
        s = np_conv3d(s, w["conv_shortcut.weight"], w["conv_shortcut.bias"], 1, False)
    else:
        s = x
    return s + h


class TestResnetBlock:
    def test_no_shortcut(self):
        x = np.random.randn(1, 4, 3, 5, 5).astype(np.float32)
        m = ResnetBlock3D(4, 4)
        w = {
            "conv1.conv.weight": np.random.randn(4, 4, 3, 3, 3).astype(np.float32)
            * 0.1,
            "conv1.conv.bias": np.zeros((4,), np.float32),
            "conv2.conv.weight": np.random.randn(4, 4, 3, 3, 3).astype(np.float32)
            * 0.1,
            "conv2.conv.bias": np.zeros((4,), np.float32),
        }
        _load_torch(m, w)
        out = m(mx.array(x), causal=False)
        exp = np_resnet_block(x, w, False, 4, 4)
        np.testing.assert_allclose(np.array(out), exp, rtol=RTOL, atol=ATOL)

    def test_with_shortcut(self):
        x = np.random.randn(1, 2, 3, 5, 5).astype(np.float32)
        m = ResnetBlock3D(2, 4)
        w = {
            "conv1.conv.weight": np.random.randn(4, 2, 3, 3, 3).astype(np.float32)
            * 0.1,
            "conv1.conv.bias": np.zeros((4,), np.float32),
            "conv2.conv.weight": np.random.randn(4, 4, 3, 3, 3).astype(np.float32)
            * 0.1,
            "conv2.conv.bias": np.zeros((4,), np.float32),
            "conv_shortcut.weight": np.random.randn(4, 2, 1, 1, 1).astype(np.float32)
            * 0.1,
            "conv_shortcut.bias": np.zeros((4,), np.float32),
            "norm3.norm.weight": np.ones((2,), np.float32),
            "norm3.norm.bias": np.zeros((2,), np.float32),
        }
        _load_torch(m, w)
        out = m(mx.array(x), causal=False)
        exp = np_resnet_block(x, w, False, 2, 4)
        np.testing.assert_allclose(np.array(out), exp, rtol=RTOL, atol=ATOL)


def np_depth_to_space(x, w, causal, stride=(2, 2, 2)):
    k = 3
    p1, p2, p3 = stride
    y = np_conv3d(x, w["conv.conv.weight"], w["conv.conv.bias"], k, causal)
    y = np_pixel_shuffle_nd(y, p1, p2, p3)
    if stride[0] == 2:
        y = y[:, :, 1:, :, :]
    return y


class TestDepthToSpace:
    def test_upsample(self):
        x = np.random.randn(1, 2, 2, 4, 4).astype(np.float32)
        m = DepthToSpaceUpsample(2, stride=(2, 2, 2))
        out_ch = 8 * 2  # 2*2*2 * 2 // 1
        w = {
            "conv.conv.weight": np.random.randn(out_ch, 2, 3, 3, 3).astype(np.float32)
            * 0.1,
            "conv.conv.bias": np.zeros((out_ch,), np.float32),
        }
        _load_torch(m, w)
        out = m(mx.array(x), causal=False)
        exp = np_depth_to_space(x, w, False)
        np.testing.assert_allclose(np.array(out), exp, rtol=RTOL, atol=ATOL)
        assert out.shape == (1, 2, 3, 8, 8)


# ---------------------------------------------------------------------------
# full Decoder numpy oracle (small config) - the integration correctness check
# ---------------------------------------------------------------------------

SMALL_BLOCKS = [["res_x", 1], ["compress_all", 1], ["res_x_y", 1], ["res_x", 1]]


def _small_decoder_weights():
    base = 4
    # output_channel growth over reversed blocks: res_x(4) res_x_y(8) compress(8) res_x(8) -> final 8
    # conv_in: 4(latent) -> 8
    w = {}
    w["conv_in.conv.weight"] = np.random.randn(8, 4, 3, 3, 3).astype(np.float32) * 0.1
    w["conv_in.conv.bias"] = np.zeros((8,), np.float32)
    # up_blocks.0 = res_x UNetMidBlock3D(in=8), 1 layer
    w["up_blocks.0.res_blocks.0.conv1.conv.weight"] = (
        np.random.randn(8, 8, 3, 3, 3).astype(np.float32) * 0.1
    )
    w["up_blocks.0.res_blocks.0.conv1.conv.bias"] = np.zeros((8,), np.float32)
    w["up_blocks.0.res_blocks.0.conv2.conv.weight"] = (
        np.random.randn(8, 8, 3, 3, 3).astype(np.float32) * 0.1
    )
    w["up_blocks.0.res_blocks.0.conv2.conv.bias"] = np.zeros((8,), np.float32)
    # up_blocks.1 = res_x_y ResnetBlock3D(8->4)
    w["up_blocks.1.conv1.conv.weight"] = (
        np.random.randn(4, 8, 3, 3, 3).astype(np.float32) * 0.1
    )
    w["up_blocks.1.conv1.conv.bias"] = np.zeros((4,), np.float32)
    w["up_blocks.1.conv2.conv.weight"] = (
        np.random.randn(4, 4, 3, 3, 3).astype(np.float32) * 0.1
    )
    w["up_blocks.1.conv2.conv.bias"] = np.zeros((4,), np.float32)
    w["up_blocks.1.conv_shortcut.weight"] = (
        np.random.randn(4, 8, 1, 1, 1).astype(np.float32) * 0.1
    )
    w["up_blocks.1.conv_shortcut.bias"] = np.zeros((4,), np.float32)
    w["up_blocks.1.norm3.norm.weight"] = np.ones((8,), np.float32)
    w["up_blocks.1.norm3.norm.bias"] = np.zeros((8,), np.float32)
    # up_blocks.2 = compress_all DepthToSpace(in=4), out=8*4=32
    w["up_blocks.2.conv.conv.weight"] = (
        np.random.randn(32, 4, 3, 3, 3).astype(np.float32) * 0.1
    )
    w["up_blocks.2.conv.conv.bias"] = np.zeros((32,), np.float32)
    # up_blocks.3 = res_x UNetMidBlock3D(in=4)
    w["up_blocks.3.res_blocks.0.conv1.conv.weight"] = (
        np.random.randn(4, 4, 3, 3, 3).astype(np.float32) * 0.1
    )
    w["up_blocks.3.res_blocks.0.conv1.conv.bias"] = np.zeros((4,), np.float32)
    w["up_blocks.3.res_blocks.0.conv2.conv.weight"] = (
        np.random.randn(4, 4, 3, 3, 3).astype(np.float32) * 0.1
    )
    w["up_blocks.3.res_blocks.0.conv2.conv.bias"] = np.zeros((4,), np.float32)
    # conv_out: 4 -> out_channels*patch^2 = 3*4 = 12
    w["conv_out.conv.weight"] = np.random.randn(12, 4, 3, 3, 3).astype(np.float32) * 0.1
    w["conv_out.conv.bias"] = np.zeros((12,), np.float32)
    return w


def np_decoder(x, w, blocks, patch_size, causal):
    k = 3
    s = np_conv3d(x, w["conv_in.conv.weight"], w["conv_in.conv.bias"], k, causal)
    output_channel = 8
    for i, (name, _) in enumerate(list(reversed(blocks))):
        if name == "res_x":
            for j in range(1):
                h = np_resnet_block(
                    s,
                    {
                        "conv1.conv.weight": w[
                            f"up_blocks.{i}.res_blocks.{j}.conv1.conv.weight"
                        ],
                        "conv1.conv.bias": w[
                            f"up_blocks.{i}.res_blocks.{j}.conv1.conv.bias"
                        ],
                        "conv2.conv.weight": w[
                            f"up_blocks.{i}.res_blocks.{j}.conv2.conv.weight"
                        ],
                        "conv2.conv.bias": w[
                            f"up_blocks.{i}.res_blocks.{j}.conv2.conv.bias"
                        ],
                    },
                    causal,
                    output_channel,
                    output_channel,
                )
                s = h
        elif name == "res_x_y":
            new_out = output_channel // 2
            s = np_resnet_block(
                s,
                {
                    "conv1.conv.weight": w[f"up_blocks.{i}.conv1.conv.weight"],
                    "conv1.conv.bias": w[f"up_blocks.{i}.conv1.conv.bias"],
                    "conv2.conv.weight": w[f"up_blocks.{i}.conv2.conv.weight"],
                    "conv2.conv.bias": w[f"up_blocks.{i}.conv2.conv.bias"],
                    "conv_shortcut.weight": w[f"up_blocks.{i}.conv_shortcut.weight"],
                    "conv_shortcut.bias": w[f"up_blocks.{i}.conv_shortcut.bias"],
                    "norm3.norm.weight": w[f"up_blocks.{i}.norm3.norm.weight"],
                    "norm3.norm.bias": w[f"up_blocks.{i}.norm3.norm.bias"],
                },
                causal,
                output_channel,
                new_out,
            )
            output_channel = new_out
        elif name == "compress_all":
            s = np_depth_to_space(
                s,
                {
                    "conv.conv.weight": w[f"up_blocks.{i}.conv.conv.weight"],
                    "conv.conv.bias": w[f"up_blocks.{i}.conv.conv.bias"],
                },
                causal,
            )
    s = np_pixel_norm(s)
    s = np_silu(s)
    s = np_conv3d(s, w["conv_out.conv.weight"], w["conv_out.conv.bias"], k, causal)
    s = np_pixel_shuffle_nd(s, 1, patch_size, patch_size)
    return s


class TestDecoderOracle:
    def test_full_forward_matches_numpy(self):
        np.random.seed(0)
        cfg = VAEConfig(
            latent_channels=4,
            out_channels=3,
            base_channels=4,
            patch_size=2,
            blocks=SMALL_BLOCKS,
            causal_decoder=False,
        )
        dec = Decoder(cfg)
        w = _small_decoder_weights()
        _load_torch(dec, w)
        x = np.random.randn(1, 4, 2, 4, 4).astype(np.float32) * 0.5
        out = dec(mx.array(x), target_shape=(1, 3, 2, 4, 4))
        exp = np_decoder(x, w, SMALL_BLOCKS, 2, False)
        np.testing.assert_allclose(np.array(out), exp, rtol=RTOL_DEC, atol=ATOL_DEC)
        assert out.shape == (1, 3, 3, 16, 16)


# ---------------------------------------------------------------------------
# from_pretrained with a mock checkpoint
# ---------------------------------------------------------------------------


def _write_safetensors(path, weights):
    from safetensors.numpy import save_file

    save_file({k: v for k, v in weights.items()}, str(path))


def _build_mock_ours_checkpoint(tmp_path: Path, cfg: VAEConfig):
    np.random.seed(1)
    base = cfg.base_channels
    blocks = cfg.blocks
    output_channel = base
    for name, _ in list(reversed(blocks)):
        if name == "res_x_y":
            output_channel = output_channel * 2
    w = {}
    w["decoder.conv_in.conv.weight"] = (
        np.random.randn(output_channel, cfg.latent_channels, 3, 3, 3).astype(np.float32)
        * 0.05
    )
    w["decoder.conv_in.conv.bias"] = np.zeros((output_channel,), np.float32)
    cur = output_channel
    for i, (name, _) in enumerate(list(reversed(blocks))):
        inp = cur
        if name == "res_x":
            for j in range(1):
                w[f"decoder.up_blocks.{i}.res_blocks.{j}.conv1.conv.weight"] = (
                    np.random.randn(inp, inp, 3, 3, 3).astype(np.float32) * 0.05
                )
                w[f"decoder.up_blocks.{i}.res_blocks.{j}.conv1.conv.bias"] = np.zeros(
                    (inp,), np.float32
                )
                w[f"decoder.up_blocks.{i}.res_blocks.{j}.conv2.conv.weight"] = (
                    np.random.randn(inp, inp, 3, 3, 3).astype(np.float32) * 0.05
                )
                w[f"decoder.up_blocks.{i}.res_blocks.{j}.conv2.conv.bias"] = np.zeros(
                    (inp,), np.float32
                )
        elif name == "res_x_y":
            cur = cur // 2
            w[f"decoder.up_blocks.{i}.conv1.conv.weight"] = (
                np.random.randn(cur, inp, 3, 3, 3).astype(np.float32) * 0.05
            )
            w[f"decoder.up_blocks.{i}.conv1.conv.bias"] = np.zeros((cur,), np.float32)
            w[f"decoder.up_blocks.{i}.conv2.conv.weight"] = (
                np.random.randn(cur, cur, 3, 3, 3).astype(np.float32) * 0.05
            )
            w[f"decoder.up_blocks.{i}.conv2.conv.bias"] = np.zeros((cur,), np.float32)
            w[f"decoder.up_blocks.{i}.conv_shortcut.weight"] = (
                np.random.randn(cur, inp, 1, 1, 1).astype(np.float32) * 0.05
            )
            w[f"decoder.up_blocks.{i}.conv_shortcut.bias"] = np.zeros(
                (cur,), np.float32
            )
            w[f"decoder.up_blocks.{i}.norm3.norm.weight"] = np.ones((inp,), np.float32)
            w[f"decoder.up_blocks.{i}.norm3.norm.bias"] = np.zeros((inp,), np.float32)
        elif name == "compress_all":
            out_ch = 8 * inp
            w[f"decoder.up_blocks.{i}.conv.conv.weight"] = (
                np.random.randn(out_ch, inp, 3, 3, 3).astype(np.float32) * 0.05
            )
            w[f"decoder.up_blocks.{i}.conv.conv.bias"] = np.zeros((out_ch,), np.float32)
    w["decoder.conv_out.conv.weight"] = (
        np.random.randn(cfg.out_channels * cfg.patch_size**2, cur, 3, 3, 3).astype(
            np.float32
        )
        * 0.05
    )
    w["decoder.conv_out.conv.bias"] = np.zeros(
        (cfg.out_channels * cfg.patch_size**2,), np.float32
    )
    w["per_channel_statistics.mean-of-means"] = np.zeros(
        (cfg.latent_channels,), np.float32
    )
    w["per_channel_statistics.std-of-means"] = np.ones(
        (cfg.latent_channels,), np.float32
    )
    cfg_dict = {
        "_class_name": "CausalVideoAutoencoder",
        "dims": 3,
        "in_channels": cfg.in_channels,
        "out_channels": cfg.out_channels,
        "latent_channels": cfg.latent_channels,
        "blocks": cfg.blocks,
        "scaling_factor": cfg.scaling_factor,
        "norm_layer": cfg.norm_layer,
        "patch_size": cfg.patch_size,
        "latent_log_var": cfg.latent_log_var,
        "use_quant_conv": cfg.use_quant_conv,
        "causal_decoder": cfg.causal_decoder,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict))
    _write_safetensors(tmp_path / "diffusion_pytorch_model.safetensors", w)
    return w


class TestFromPretrained:
    def test_loads_ours_format(self, tmp_path):
        cfg = VAEConfig(
            latent_channels=4,
            out_channels=3,
            base_channels=4,
            patch_size=2,
            blocks=SMALL_BLOCKS,
            causal_decoder=False,
        )
        _build_mock_ours_checkpoint(tmp_path, cfg)
        m = LTVideoVAE.from_pretrained(tmp_path, dtype=mx.float32)
        z = mx.array(np.random.randn(1, 4, 2, 4, 4).astype(np.float32) * 0.5)
        out = m.decode(z, target_shape=(1, 3, 2, 4, 4))
        mx.eval(out)
        assert out.shape == (1, 3, 3, 16, 16)
        assert bool(mx.all(mx.isfinite(out)))
        assert m.mean_of_means is not None
        assert m.std_of_means is not None


# ---------------------------------------------------------------------------
# diffusers -> OURS key rename (single forward pass, dict-ordered, no cascade)
# ---------------------------------------------------------------------------

from fusion_mlx.video.ltx_video_legacy.vae import _map_vae_decoder_weights

# (diffusers key, expected OURS key after strip "decoder.")
# NB: diffusers up_blocks.{2,3}.conv_in is itself an LTXVideoResnetBlock3d, so
# its sub-keys are conv1.conv / conv2.conv / conv_shortcut.conv / norm3.
RENAME_CASES = [
    (
        "decoder.mid_block.resnets.0.conv1.conv.weight",
        "up_blocks.0.res_blocks.0.conv1.conv.weight",
    ),
    (
        "decoder.up_blocks.0.resnets.0.conv2.conv.weight",
        "up_blocks.1.res_blocks.0.conv2.conv.weight",
    ),
    (
        "decoder.up_blocks.1.upsamplers.0.conv.conv.weight",
        "up_blocks.2.conv.conv.weight",
    ),
    (
        "decoder.up_blocks.1.resnets.0.conv1.conv.weight",
        "up_blocks.3.res_blocks.0.conv1.conv.weight",
    ),
    (
        "decoder.up_blocks.1.resnets.0.conv_shortcut.conv.weight",
        "up_blocks.3.res_blocks.0.conv_shortcut.weight",
    ),
    (
        "decoder.up_blocks.1.resnets.0.norm3.weight",
        "up_blocks.3.res_blocks.0.norm3.norm.weight",
    ),
    ("decoder.up_blocks.2.conv_in.conv1.conv.weight", "up_blocks.4.conv1.conv.weight"),
    (
        "decoder.up_blocks.2.conv_in.conv_shortcut.conv.weight",
        "up_blocks.4.conv_shortcut.weight",
    ),
    ("decoder.up_blocks.2.conv_in.norm3.weight", "up_blocks.4.norm3.norm.weight"),
    (
        "decoder.up_blocks.2.upsamplers.0.conv.conv.weight",
        "up_blocks.5.conv.conv.weight",
    ),
    (
        "decoder.up_blocks.2.resnets.0.conv1.conv.weight",
        "up_blocks.6.res_blocks.0.conv1.conv.weight",
    ),
    ("decoder.up_blocks.3.conv_in.conv1.conv.weight", "up_blocks.7.conv1.conv.weight"),
    (
        "decoder.up_blocks.3.upsamplers.0.conv.conv.weight",
        "up_blocks.8.conv.conv.weight",
    ),
    (
        "decoder.up_blocks.3.resnets.0.conv1.conv.weight",
        "up_blocks.9.res_blocks.0.conv1.conv.weight",
    ),
    ("decoder.conv_in.conv.weight", "conv_in.conv.weight"),
    ("decoder.conv_out.conv.weight", "conv_out.conv.weight"),
]


class TestDiffusersRename:
    def test_keys_renamed_correctly(self):
        raw = {}
        for diff_key, _ in RENAME_CASES:
            # 5D so the transpose path is exercised; values irrelevant for keys.
            raw[diff_key] = np.zeros((2, 2, 3, 3, 3), np.float32)
        mapped = _map_vae_decoder_weights({k: mx.array(v) for k, v in raw.items()})
        for diff_key, expected_ours in RENAME_CASES:
            assert (
                expected_ours in mapped
            ), f"{diff_key} -> {expected_ours} missing (got {sorted(mapped.keys())})"
        # no encoder/non-decoder keys survive
        assert all(not k.startswith("encoder.") for k in mapped)

    def test_5d_weights_transposed_to_mlx_layout(self):
        # torch (out,in,kD,kH,kW) -> MLX (out,kD,kH,kW,in)
        raw = {
            "decoder.conv_out.conv.weight": np.arange(
                2 * 3 * 1 * 1 * 1, dtype=np.float32
            ).reshape(2, 3, 1, 1, 1)
        }
        mapped = _map_vae_decoder_weights({k: mx.array(v) for k, v in raw.items()})
        w = np.array(mapped["conv_out.conv.weight"])
        assert w.shape == (2, 1, 1, 1, 3)
        # torch[1,0,0,0,0]=3 (arange 0..5 reshaped (2,3,...) -> [1,0]=3)
        assert w[1, 0, 0, 0, 0] == 3
        # round-trip: MLX (out,kD,kH,kW,in).transpose(0,4,1,2,3) == torch
        rt = w.transpose(0, 4, 1, 2, 3)
        np.testing.assert_array_equal(rt, raw["decoder.conv_out.conv.weight"])


# ---------------------------------------------------------------------------
# full OURS config: construct + param count + tiny forward
# ---------------------------------------------------------------------------


class TestFullOURSConfig:
    def test_constructs_and_param_count(self):
        cfg = VAEConfig.from_dict(OURS_VAE_CONFIG)
        vae = LTVideoVAE(cfg)
        keys = _flatten_keys(vae.decoder.parameters())

        def _count(tree):
            if isinstance(tree, dict):
                return sum(_count(v) for v in tree.values())
            if isinstance(tree, list):
                return sum(_count(v) for v in tree)
            return int(tree.size)

        n_params = _count(vae.decoder.parameters())
        # LTX-Video 0.9.x VAE decoder: 512-channel peak, ~238M params
        # (3D 3x3x3 convs at 512ch dominate; consistent with ~500MB bf16 on disk)
        assert n_params > 200_000_000, f"param count too low: {n_params/1e6:.1f}M"
        assert n_params < 280_000_000, f"param count too high: {n_params/1e6:.1f}M"
        # every decoder key is a conv/norm weight we expect
        assert "conv_in.conv.weight" in keys
        assert "conv_out.conv.weight" in keys
        assert any(k.startswith("up_blocks.0.res_blocks") for k in keys)
        assert any(k.startswith("up_blocks.9.res_blocks") for k in keys)

    def test_tiny_forward_shape(self):
        cfg = VAEConfig.from_dict(OURS_VAE_CONFIG)
        vae = LTVideoVAE(cfg)
        z = mx.zeros((1, 128, 1, 4, 4))
        out = vae.decode(z, target_shape=(1, 3, 1, 4, 4))
        mx.eval(out)
        # 4x4 latent, patch 4, 3x compress(2): spatial 4 -> 8 -> 16 -> 32 -> *4 = 128
        assert out.shape == (1, 3, 1, 128, 128)
        assert bool(mx.all(mx.isfinite(out)))
