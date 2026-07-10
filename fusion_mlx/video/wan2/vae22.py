import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

CACHE_T = 2

# Per-channel normalization for z_dim=48 latent space
VAE22_MEAN = mx.array(
    [
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.1570,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.1230,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.0520,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    ]
)

VAE22_STD = mx.array(
    [
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.4990,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.0600,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    ]
)


class CausalConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        # Causal temporal padding: matches the reference CausalConv3d(nn.Conv3d)
        # which converts symmetric padding to causal: 2*padding[0] on the left.
        # For most convs (kernel=3, padding=1): 2*1 = 2 (same as kernel-1).
        # For downsample time_conv (kernel=3, padding=0): 2*0 = 0 (NO padding).
        self._causal_pad_t = 2 * padding[0]
        self._pad_h = padding[1]
        self._pad_w = padding[2]

        # Weight: [O, D, H, W, I] for MLX
        self.weight = mx.zeros(
            (out_channels, kernel_size[0], kernel_size[1], kernel_size[2], in_channels)
        )
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x, cache_x=None):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape
        kd, kh, kw = self.kernel_size

        # For 1x1x1 kernel or kernel_d==1, use direct conv
        if kd == 1 and kh == 1 and kw == 1:
            # Simple pointwise: reshape to [B*T, 1, 1, C] → conv2d
            x_flat = x.reshape(B * T, H, W, C)
            w2d = self.weight[:, 0, :, :, :]  # [O, kH, kW, I]
            y = mx.conv_general(x_flat, w2d) + self.bias
            return y.reshape(B, T, y.shape[1], y.shape[2], -1)

        # Causal temporal padding: prepend cached frames if available,
        # then zero-pad any remaining positions.
        pad_needed = self._causal_pad_t
        if cache_x is not None and pad_needed > 0:
            x = mx.concatenate([cache_x, x], axis=1)
            pad_needed -= cache_x.shape[1]

        if pad_needed > 0:
            pad_t = mx.zeros((B, pad_needed, H, W, C), dtype=x.dtype)
            x = mx.concatenate([pad_t, x], axis=1)

        # Spatial padding
        if self._pad_h > 0 or self._pad_w > 0:
            x = mx.pad(
                x,
                [
                    (0, 0),
                    (0, 0),
                    (self._pad_h, self._pad_h),
                    (self._pad_w, self._pad_w),
                    (0, 0),
                ],
            )

        T_padded = x.shape[1]
        H_padded, W_padded = x.shape[2], x.shape[3]
        T_out = (T_padded - kd) // self.stride[0] + 1

        # Decompose 3D conv into sum of 2D convolutions over temporal kernel
        # weight shape: [O, kd, kh, kw, I] → split into kd 2D kernels [O, kh, kw, I]
        outputs = []
        for t in range(T_out):
            t_start = t * self.stride[0]
            # Sum 2D convs for each temporal kernel position
            accum = None
            for d in range(kd):
                frame = x[:, t_start + d]  # [B, H_padded, W_padded, C]
                w2d = self.weight[:, d, :, :, :]  # [O, kh, kw, I]
                conv_out = mx.conv_general(
                    frame, w2d, stride=(self.stride[1], self.stride[2])
                )
                accum = conv_out if accum is None else accum + conv_out
            outputs.append(accum + self.bias)

        return mx.stack(outputs, axis=1)  # [B, T_out, H_out, W_out, O]


class RMS_norm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        # Weight stored as (dim,) — PyTorch stores (dim, 1, 1, 1) but we squeeze
        self.gamma = mx.ones((dim,))

    def __call__(self, x):
        # x: [..., C] (channels-last)
        # PyTorch uses F.normalize (L2 norm), not RMS: x / max(||x||_2, eps)
        l2_sq = mx.sum(x * x, axis=-1, keepdims=True)
        return (
            x * mx.rsqrt(mx.maximum(l2_sq, mx.array(1e-24))) * self.scale * self.gamma
        )


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Sequential residual path: [norm, silu, conv3d, norm, silu, dropout, conv3d]
        # We store as named layers matching PyTorch's indices
        self.residual = ResidualBlockLayers(in_dim, out_dim)
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else None

    def __call__(self, x, feat_cache=None, feat_idx=None):
        h = self.shortcut(x) if self.shortcut is not None else x
        return self.residual(x, feat_cache, feat_idx) + h


class ResidualBlockLayers(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Indices match PyTorch nn.Sequential indices for weight key compat
        # Index 0: RMS_norm
        self.layer_0 = RMS_norm(in_dim)
        # Index 2: CausalConv3d
        self.layer_2 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        # Index 3: RMS_norm
        self.layer_3 = RMS_norm(out_dim)
        # Index 6: CausalConv3d
        self.layer_6 = CausalConv3d(out_dim, out_dim, 3, padding=1)

    def _conv_with_cache(self, conv, x, feat_cache, feat_idx):
        idx = feat_idx[0]
        # Save last CACHE_T frames before conv (for next chunk's context)
        cache_x = x[:, -CACHE_T:]
        if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
            cache_x = mx.concatenate([feat_cache[idx][:, -1:], cache_x], axis=1)
        out = conv(x, cache_x=feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
        return out

    def __call__(self, x, feat_cache=None, feat_idx=None):
        x = self.layer_0(x)
        x = nn.silu(x)
        if feat_cache is not None:
            x = self._conv_with_cache(self.layer_2, x, feat_cache, feat_idx)
        else:
            x = self.layer_2(x)
        mx.eval(x)  # Eval between convolutions to limit graph size
        x = self.layer_3(x)
        x = nn.silu(x)
        if feat_cache is not None:
            x = self._conv_with_cache(self.layer_6, x, feat_cache, feat_idx)
        else:
            x = self.layer_6(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        # Conv2d as linear per spatial position — weight [O, H, W, I] for MLX
        # to_qkv: dim -> 3*dim, proj: dim -> dim (1x1 conv2d)
        self.to_qkv_weight = mx.zeros((3 * dim, 1, 1, dim))
        self.to_qkv_bias = mx.zeros((3 * dim,))
        self.proj_weight = mx.zeros((dim, 1, 1, dim))
        self.proj_bias = mx.zeros((dim,))

    def __call__(self, x):
        # x: [B, T, H, W, C]
        identity = x
        B, T, H, W, C = x.shape

        # Apply per frame: merge B and T
        x = x.reshape(B * T, H, W, C)
        x = self.norm(x)

        # QKV via 1x1 conv2d (equivalent to linear on last dim)
        qkv = (
            mx.conv_general(x, self.to_qkv_weight) + self.to_qkv_bias
        )  # [BT, H, W, 3C]
        qkv = qkv.reshape(B * T, H * W, 3 * C)
        q, k, v = mx.split(qkv, 3, axis=-1)  # each [BT, HW, C]

        # Single-head attention
        q = q[:, None, :, :]  # [BT, 1, HW, C]
        k = k[:, None, :, :]
        v = v[:, None, :, :]

        scale = C**-0.5
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale
        )  # [BT, 1, HW, C]
        out = out.squeeze(1).reshape(B * T, H, W, C)

        # Project output
        out = mx.conv_general(out, self.proj_weight) + self.proj_bias  # [BT, H, W, C]
        out = out.reshape(B, T, H, W, C)
        return out + identity


class DupUp3D(nn.Module):

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x, first_chunk=False):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape

        # Repeat channels
        x = mx.repeat(x, self.repeats, axis=-1)  # [B, T, H, W, C*repeats]

        # Reshape to [B, T, H, W, out_C, factor_t, factor_s, factor_s]
        x = x.reshape(
            B, T, H, W, self.out_channels, self.factor_t, self.factor_s, self.factor_s
        )

        # Permute to interleave: [B, T, factor_t, H, factor_s, W, factor_s, out_C]
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)

        # Reshape to final: [B, T*factor_t, H*factor_s, W*factor_s, out_C]
        x = x.reshape(
            B,
            T * self.factor_t,
            H * self.factor_s,
            W * self.factor_s,
            self.out_channels,
        )

        if first_chunk:
            x = x[:, self.factor_t - 1 :, :, :, :]
        return x


class AvgDown3D(nn.Module):

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape

        # Pad temporal if not divisible by factor_t
        pad_t = (self.factor_t - T % self.factor_t) % self.factor_t
        if pad_t > 0:
            x = mx.pad(x, [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)])
            T = T + pad_t

        ft, fs = self.factor_t, self.factor_s
        # Reshape to split spatial/temporal dims
        x = x.reshape(B, T // ft, ft, H // fs, fs, W // fs, fs, C)
        # Move factors next to channels
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)  # [B, T', H', W', C, ft, fs, fs]
        # Expand channels
        x = x.reshape(B, T // ft, H // fs, W // fs, C * self.factor)
        # Group and average
        x = x.reshape(B, T // ft, H // fs, W // fs, self.out_channels, self.group_size)
        x = x.mean(axis=-1)
        return x


class Resample(nn.Module):

    def __init__(self, dim, mode):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            # resample.0 = Upsample (no params), resample.1 = Conv2d
            self.resample_weight = mx.zeros((dim, 3, 3, dim))  # Conv2d [O, H, W, I]
            self.resample_bias = mx.zeros((dim,))
        elif mode == "upsample3d":
            self.resample_weight = mx.zeros((dim, 3, 3, dim))
            self.resample_bias = mx.zeros((dim,))
            # time_conv: CausalConv3d(dim, dim*2, (3,1,1), padding=(1,0,0))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            # resample.0 = ZeroPad2d (no params), resample.1 = Conv2d(stride=2)
            self.resample_weight = mx.zeros((dim, 3, 3, dim))
            self.resample_bias = mx.zeros((dim,))
        elif mode == "downsample3d":
            self.resample_weight = mx.zeros((dim, 3, 3, dim))
            self.resample_bias = mx.zeros((dim,))
            # time_conv: CausalConv3d(dim, dim, (3,1,1), stride=(2,1,1))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _upsample2x(self, x):
        N, H, W, C = x.shape
        # Repeat along H and W axes separately
        x = mx.repeat(x, repeats=2, axis=1)  # [N, 2H, W, C]
        x = mx.repeat(x, repeats=2, axis=2)  # [N, 2H, 2W, C]
        return x

    def _conv2d(self, x):
        x = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
        return mx.conv_general(x, self.resample_weight) + self.resample_bias

    def _downsample_conv2d(self, x):
        # ZeroPad2d((0,1,0,1)): pad right=1, bottom=1
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return (
            mx.conv_general(x, self.resample_weight, stride=(2, 2)) + self.resample_bias
        )

    def __call__(self, x, first_chunk=False, feat_cache=None, feat_idx=None):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape

        # --- Temporal upsample (before spatial, matching reference) ---
        if self.mode == "upsample3d":
            if first_chunk and T > 1:
                first_frame = x[:, 0:1]
                rest = x[:, 1:]
                tc_out = self.time_conv(rest)
                tc_out = tc_out.reshape(B, T - 1, H, W, 2, C)
                stream0 = tc_out[:, :, :, :, 0, :]
                stream1 = tc_out[:, :, :, :, 1, :]
                interleaved = mx.stack([stream0, stream1], axis=2)
                interleaved = interleaved.reshape(B, (T - 1) * 2, H, W, C)
                x = mx.concatenate([first_frame, interleaved], axis=1)
            else:
                tc_out = self.time_conv(x)
                tc_out = tc_out.reshape(B, T, H, W, 2, C)
                stream0 = tc_out[:, :, :, :, 0, :]
                stream1 = tc_out[:, :, :, :, 1, :]
                x = mx.stack([stream0, stream1], axis=2)
                x = x.reshape(B, T * 2, H, W, C)
            mx.eval(x)
            T = x.shape[1]

        # --- Spatial operation (all modes, matching reference line 152-155) ---
        if self.mode in ("upsample2d", "upsample3d"):
            chunk_size = 8
            chunks = []
            for t_start in range(0, T, chunk_size):
                t_end = min(t_start + chunk_size, T)
                x_chunk = x[:, t_start:t_end].reshape(-1, H, W, C)
                x_chunk = self._upsample2x(x_chunk)
                x_chunk = self._conv2d(x_chunk)
                mx.eval(x_chunk)
                chunks.append(x_chunk)
            x = mx.concatenate(chunks, axis=0)
            H2, W2 = x.shape[1], x.shape[2]
            x = x.reshape(B, T, H2, W2, C)
        elif self.mode in ("downsample2d", "downsample3d"):
            x_flat = x.reshape(B * T, H, W, C)
            x_flat = self._downsample_conv2d(x_flat)
            mx.eval(x_flat)
            H2, W2 = x_flat.shape[1], x_flat.shape[2]
            x = x_flat.reshape(B, T, H2, W2, C)

        # --- Temporal downsample (after spatial, matching reference line 157-168) ---
        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    # First chunk: store spatially-downsampled result, skip time_conv
                    feat_cache[idx] = x
                    feat_idx[0] += 1
                else:
                    # Subsequent chunks: prepend cached last frame, apply time_conv
                    save_x = x[:, -1:]
                    x = self.time_conv(
                        mx.concatenate([feat_cache[idx][:, -1:], x], axis=1)
                    )
                    feat_cache[idx] = save_x
                    feat_idx[0] += 1
            elif T > 1:
                x = self.time_conv(x)
            mx.eval(x)

        return x


class Up_ResidualBlock(nn.Module):

    def __init__(
        self, in_dim, out_dim, num_res_blocks, temperal_upsample=False, up_flag=False
    ):
        super().__init__()
        self.up_flag = up_flag

        # DupUp3D shortcut (no learnable params)
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path: ResidualBlocks + optional Resample
        blocks = []
        dim_in = in_dim
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(dim_in, out_dim))
            dim_in = out_dim

        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            blocks.append(Resample(out_dim, mode=mode))

        self.upsamples = blocks

    def __call__(self, x, first_chunk=False):
        x_main = x
        for module in self.upsamples:
            if isinstance(module, Resample):
                x_main = module(x_main, first_chunk)
            else:
                x_main = module(x_main)
            mx.eval(x_main)  # Limit graph size per sub-block

        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            mx.eval(x_shortcut)
            return x_main + x_shortcut
        return x_main


class Down_ResidualBlock(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        num_res_blocks,
        temperal_downsample=False,
        down_flag=False,
    ):
        super().__init__()
        self.down_flag = down_flag

        # AvgDown3D shortcut (no learnable params, always present)
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path: ResidualBlocks + optional Resample
        blocks = []
        dim_in = in_dim
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(dim_in, out_dim))
            dim_in = out_dim

        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            blocks.append(Resample(out_dim, mode=mode))

        self.downsamples = blocks

    def __call__(self, x, feat_cache=None, feat_idx=None):
        x_shortcut = self.avg_shortcut(x)
        mx.eval(x_shortcut)

        for module in self.downsamples:
            if feat_cache is not None:
                if isinstance(module, ResidualBlock):
                    x = module(x, feat_cache, feat_idx)
                elif isinstance(module, Resample):
                    x = module(x, feat_cache=feat_cache, feat_idx=feat_idx)
                else:
                    x = module(x)
            else:
                x = module(x)
            mx.eval(x)

        return x + x_shortcut


class Decoder3d(nn.Module):

    def __init__(
        self,
        dim=256,
        z_dim=48,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        temperal_upsample=(True, True, False),
    ):
        super().__init__()
        # Compute layer dimensions
        dims = [dim * dim_mult[-1]] + [dim * m for m in reversed(dim_mult)]

        # Initial conv
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # Middle blocks
        self.middle = [
            ResidualBlock(dims[0], dims[0]),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0]),
        ]

        # Upsample blocks
        self.upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = temperal_upsample[i] if i < len(temperal_upsample) else False
            self.upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks + 1,
                    temperal_upsample=t_up,
                    up_flag=(i != len(dim_mult) - 1),
                )
            )

        # Output head: [RMS_norm, SiLU, CausalConv3d]
        self.head = Head22(dims[-1])

    def __call__(self, x, first_chunk=False):
        # x: [B, T, H, W, C=z_dim]
        x = self.conv1(x)

        for layer in self.middle:
            x = layer(x)
        mx.eval(x)  # Evaluate to limit graph size

        for i, layer in enumerate(self.upsamples):
            x = layer(x, first_chunk)
            mx.eval(x)  # Evaluate after each upsample block

        x = self.head(x)
        return x


class Encoder3d(nn.Module):

    def __init__(
        self,
        dim=160,
        z_dim=96,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        temperal_downsample=(False, True, True),
    ):
        super().__init__()
        # Channel dimensions: [160, 160, 320, 640, 640]
        dims = [dim * m for m in [1] + list(dim_mult)]

        # Initial conv: patchified input (12 ch) → first dim
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # Downsample blocks
        self.downsamples = []
        for i in range(len(dim_mult)):
            in_d, out_d = dims[i], dims[i + 1]
            t_down = temperal_downsample[i] if i < len(temperal_downsample) else False
            self.downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_d,
                    out_dim=out_d,
                    num_res_blocks=num_res_blocks,
                    temperal_downsample=t_down,
                    down_flag=(i < len(dim_mult) - 1),
                )
            )

        # Middle blocks (same as decoder)
        out_dim = dims[-1]
        self.middle = [
            ResidualBlock(out_dim, out_dim),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim),
        ]

        # Output head: RMS_norm → SiLU → CausalConv3d → z_dim channels
        self.head = Head22(out_dim, out_channels=z_dim)

    def __call__(self, x, feat_cache=None, feat_idx=None):
        # x: [B, T, H, W, 12] (patchified)
        if feat_cache is not None:
            return self._forward_cached(x, feat_cache, feat_idx)

        # No cache: internally chunk as 1+4+4+... (matches reference behavior)
        num_convs = _count_conv3d(self)
        internal_cache = [None] * num_convs
        T = x.shape[1]
        starts = [0] + list(range(1, T, 4))
        ends = starts[1:] + [T]
        outputs = []
        for s, e in zip(starts, ends):
            if s >= e:
                continue
            feat_idx_local = [0]
            out = self._forward_cached(x[:, s:e], internal_cache, feat_idx_local)
            outputs.append(out)
            mx.eval(internal_cache)
        if len(outputs) == 1:
            return outputs[0]
        return mx.concatenate(outputs, axis=1)

    def _forward_cached(self, x, feat_cache, feat_idx):
        idx = feat_idx[0]
        cache_x = x[:, -CACHE_T:]
        if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
            cache_x = mx.concatenate([feat_cache[idx][:, -1:], cache_x], axis=1)
        x = self.conv1(x, cache_x=feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        mx.eval(x)

        x = self.head(x, feat_cache, feat_idx)
        return x


class Head22(nn.Module):

    def __init__(self, dim, out_channels=12):
        super().__init__()
        # Index 0: RMS_norm
        self.layer_0 = RMS_norm(dim)
        # Index 2: CausalConv3d
        self.layer_2 = CausalConv3d(dim, out_channels, 3, padding=1)

    def __call__(self, x, feat_cache=None, feat_idx=None):
        x = self.layer_0(x)
        x = nn.silu(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, -1:], cache_x], axis=1)
            x = self.layer_2(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.layer_2(x)
        return x


def _count_conv3d(module):
    count = 0
    if isinstance(module, CausalConv3d):
        count += 1
    for child in module.children().values():
        if isinstance(child, list):
            for item in child:
                count += _count_conv3d(item)
        elif isinstance(child, nn.Module):
            count += _count_conv3d(child)
    return count


class Wan22VAEEncoder(nn.Module):

    def __init__(self, z_dim=48, dim=160):
        super().__init__()
        self.z_dim = z_dim
        # conv1: top-level 1x1x1 conv after encoder (z_dim*2 → z_dim*2)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.encoder = Encoder3d(
            dim=dim,
            z_dim=z_dim * 2,  # Encoder outputs z_dim*2, split into mu + log_var
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            temperal_downsample=(False, True, True),
        )

    def encode(self, img):
        x = _patchify(img, patch_size=2)
        T = x.shape[1]

        # Initialize temporal cache (one slot per CausalConv3d in encoder)
        num_convs = _count_conv3d(self.encoder)
        feat_cache = [None] * num_convs

        # Chunked encoding: first chunk = 1 frame, rest = 4 frames each
        num_chunks = 1 + (T - 1) // 4
        out = None
        for i in range(num_chunks):
            feat_idx = [0]  # Reset layer index each chunk (but keep cache)
            if i == 0:
                chunk = x[:, :1]
            else:
                chunk = x[:, 1 + 4 * (i - 1) : 1 + 4 * i]
            chunk_out = self.encoder(chunk, feat_cache=feat_cache, feat_idx=feat_idx)
            if out is None:
                out = chunk_out
            else:
                out = mx.concatenate([out, chunk_out], axis=1)
            mx.eval(out)

        # conv1 (pointwise) + split into mu, log_var
        out = self.conv1(out)
        mu = out[:, :, :, :, : self.z_dim]

        # Normalize
        mu = normalize_latents(mu)
        return mu

    def __call__(self, img):
        return self.encode(img)


class Wan22VAEDecoder(nn.Module):

    def __init__(self, z_dim=48, dim=160, dec_dim=256):
        super().__init__()
        self.z_dim = z_dim
        # conv2: 1x1x1 conv before decoder
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim=dec_dim,
            z_dim=z_dim,
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            temperal_upsample=(True, True, False),
        )

    def __call__(self, z):
        x = self.conv2(z)

        # All-at-once decode with first_chunk=True to trim extra temporal
        # frames from causal padding (matches PyTorch's chunked behavior)
        out = self.decoder(x, first_chunk=True)

        # Unpatchify: 12 channels → 3 RGB (spatial 2×2)
        out = _unpatchify(out, patch_size=2)

        return mx.clip(out, -1.0, 1.0)

    def decode_tiled(self, z, tiling_config=None):
        from .tiling import TilingConfig, decode_with_tiling

        if tiling_config is None:
            tiling_config = TilingConfig.default()

        # Check if tiling is actually needed
        b, t, h_px, w_px, c = z.shape
        # Latent dimensions (before conv2/decoder upsampling)
        h_lat, w_lat = h_px, w_px
        needs_tiling = False
        if tiling_config.spatial_config is not None:
            s_tile = tiling_config.spatial_config.tile_size_in_pixels // 16
            if h_lat > s_tile or w_lat > s_tile:
                needs_tiling = True
        if tiling_config.temporal_config is not None:
            t_tile = tiling_config.temporal_config.tile_size_in_frames // 4
            if t > t_tile:
                needs_tiling = True

        if not needs_tiling:
            return self(z)

        # Transpose to channels-first for decode_with_tiling: [B,T,H,W,C] → [B,C,T,H,W]
        z_cf = z.transpose(0, 4, 1, 2, 3)

        # Tile decoder: receives (B,C,T,H,W) channels-first, returns (B,3,T',H',W')
        def tile_decode(tile_latents, **kwargs):
            tile_cl = tile_latents.transpose(0, 2, 3, 4, 1)  # → [B,T,H,W,C]
            x = self.conv2(tile_cl)
            out = self.decoder(x, first_chunk=True)
            out = _unpatchify(out, patch_size=2)
            out = mx.clip(out, -1.0, 1.0)
            return out.transpose(0, 4, 1, 2, 3)  # → [B,3,T',H',W']

        result_cf = decode_with_tiling(
            decoder_fn=tile_decode,
            latents=z_cf,
            tiling_config=tiling_config,
            spatial_scale=16,  # 8× conv upsample + 2× unpatchify
            temporal_scale=4,  # two 2× temporal upsamples (first_chunk=True → causal)
            causal_temporal=True,
        )

        # Back to channels-last: [B,3,T',H',W'] → [B,T',H',W',3]
        return result_cf.transpose(0, 2, 3, 4, 1)


def denormalize_latents(z, mean=None, std=None):
    if mean is None:
        mean = VAE22_MEAN
    if std is None:
        std = VAE22_STD
    inv_scale = std  # scale was 1/std, so divide by scale = multiply by std
    return z * inv_scale.reshape(1, 1, 1, 1, -1) + mean.reshape(1, 1, 1, 1, -1)


def normalize_latents(z, mean=None, std=None):
    if mean is None:
        mean = VAE22_MEAN
    if std is None:
        std = VAE22_STD
    return (z - mean.reshape(1, 1, 1, 1, -1)) / std.reshape(1, 1, 1, 1, -1)


def _unpatchify(x, patch_size=2):
    if patch_size == 1:
        return x
    B, T, H, W, Cpacked = x.shape
    C = Cpacked // (patch_size * patch_size)
    # Reshape: [B, T, H, W, r, q, C] then rearrange to [B, T, H*q, W*r, C]
    # PyTorch patchify: "b c f (h q) (w r) -> b (c r q) f h w" — so c is packed as (c, r, q)
    # Unpatchify reverses: [B, T, H, W, (C, r, q)] -> [B, T, H, q, W, r, C]
    x = x.reshape(B, T, H, W, C, patch_size, patch_size)
    # Rearrange: put q next to H, r next to W
    x = x.transpose(0, 1, 2, 6, 3, 5, 4)  # [B, T, H, q, W, r, C]
    x = x.reshape(B, T, H * patch_size, W * patch_size, C)
    return x


def _patchify(x, patch_size=2):
    if patch_size == 1:
        return x
    B, T, Hfull, Wfull, C = x.shape
    H = Hfull // patch_size
    W = Wfull // patch_size
    # [B, T, H, q, W, r, C]
    x = x.reshape(B, T, H, patch_size, W, patch_size, C)
    # Rearrange to pack q,r into channels: [B, T, H, W, C, r, q]
    x = x.transpose(0, 1, 2, 4, 6, 5, 3)  # [B, T, H, W, C, r, q]
    x = x.reshape(B, T, H, W, C * patch_size * patch_size)
    return x


def sanitize_wan22_vae_weights(weights: dict, include_encoder: bool = False) -> dict:
    sanitized = {}
    consumed = set()

    for key, value in weights.items():
        # Skip encoder and conv1 unless requested
        if not include_encoder:
            if key.startswith("encoder.") or key.startswith("conv1."):
                consumed.add(key)
                continue

        new_key = key

        # Map nn.Sequential indexed layers to our named attributes
        # ResidualBlockLayers: indices 0, 2, 3, 6 → _layer_0, _layer_2, _layer_3, _layer_6
        # Head22: indices 0, 2 → _layer_0, _layer_2
        for idx in ["0", "2", "3", "6"]:
            # Match patterns like "residual.0.gamma" → "residual.layer_0.gamma"
            # or "head.0.gamma" → "head.layer_0.gamma"
            old_pattern = f".residual.{idx}."
            new_pattern = f".residual.layer_{idx}."
            new_key = new_key.replace(old_pattern, new_pattern)

        # Head layer mapping: head.0.gamma → head.layer_0.gamma, head.2.weight → head.layer_2.weight
        for idx in ["0", "2"]:
            old_pattern = f".head.{idx}."
            new_pattern = f".head.layer_{idx}."
            new_key = new_key.replace(old_pattern, new_pattern)

        # Map Resample Conv2d: resample.1.weight → resample_weight, resample.1.bias → resample_bias
        if ".resample.1.weight" in new_key:
            new_key = new_key.replace(".resample.1.weight", ".resample_weight")
        elif ".resample.1.bias" in new_key:
            new_key = new_key.replace(".resample.1.bias", ".resample_bias")

        # Map AttentionBlock Conv2d weights
        if ".to_qkv.weight" in new_key:
            new_key = new_key.replace(".to_qkv.weight", ".to_qkv_weight")
        elif ".to_qkv.bias" in new_key:
            new_key = new_key.replace(".to_qkv.bias", ".to_qkv_bias")
        elif ".proj.weight" in new_key and "time_projection" not in new_key:
            new_key = new_key.replace(".proj.weight", ".proj_weight")
        elif ".proj.bias" in new_key and "time_projection" not in new_key:
            new_key = new_key.replace(".proj.bias", ".proj_bias")

        # Transpose conv weights to channels-last
        is_weight = new_key.endswith(".weight") or new_key.endswith("_weight")
        if is_weight:
            if value.ndim == 5:
                # Conv3d: [O, I, D, H, W] → [O, D, H, W, I]
                value = np.transpose(np.array(value), (0, 2, 3, 4, 1))
                value = mx.array(value)
            elif value.ndim == 4:
                # Conv2d: [O, I, H, W] → [O, H, W, I]
                value = np.transpose(np.array(value), (0, 2, 3, 1))
                value = mx.array(value)

        # Squeeze RMS_norm gamma: (dim, 1, 1, 1) or (dim, 1, 1) → (dim,)
        if "gamma" in new_key:
            value = mx.array(np.array(value).squeeze())

        sanitized[new_key] = value
        consumed.add(key)

    unconsumed = set(weights.keys()) - consumed
    if unconsumed:
        logger.warning("Unconsumed Wan2.2 VAE weight keys: %s", sorted(unconsumed))

    return sanitized
