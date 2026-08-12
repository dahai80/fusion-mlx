import mlx.core as mx
import mlx.nn as nn

CACHE_T = 2

# Per-channel normalization statistics for z_dim=16
VAE_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
VAE_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        # Causal time padding: upstream CausalConv3d sets _padding = 2*padding[0]
        # (left-only causal pad, no future context). This equals k-stride for the
        # common stride=1 case but is 0 for the downsample3d time_conv which uses
        # padding=(0,0,0) — the causal context there comes from cache_x, not zero
        # padding. Using k-stride instead of 2*padding[0] over-pads the stride=2
        # path and corrupts both frame count and values (issue #458).
        self._causal_pad_t = 2 * padding[0]
        self._pad_h = padding[1]
        self._pad_w = padding[2]

        # MLX Conv3d: weight shape [O, D, H, W, I]
        # MLX 0.32 mx.zeros/mx.ones 对 5D tuple 报 std::bad_cast, broadcast_to 不接 mx.array
        # 绕开: numpy 构零张量再 mx.array (load_weights 后会被真实权重覆盖)
        import numpy as np

        w_shape = (
            out_channels,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            in_channels,
        )
        self.weight = mx.array(np.zeros(w_shape, dtype=np.float32))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array, cache_x: mx.array = None) -> mx.array:
        b, c, t, h, w = x.shape

        causal_pad = self._causal_pad_t
        if cache_x is not None and causal_pad > 0:
            x = mx.concatenate([cache_x, x], axis=2)
            causal_pad = max(0, causal_pad - cache_x.shape[2])

        if causal_pad > 0:
            pad_t = mx.zeros((b, c, causal_pad, h, w), dtype=x.dtype)
            x = mx.concatenate([pad_t, x], axis=2)

        if self._pad_h > 0 or self._pad_w > 0:
            x = mx.pad(
                x,
                [
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (self._pad_h, self._pad_h),
                    (self._pad_w, self._pad_w),
                ],
            )

        x = x.transpose(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        out = self._conv3d(x)
        return out.transpose(0, 4, 1, 2, 3)  # [B, O, T', H', W']

    def _conv3d(self, x: mx.array) -> mx.array:
        b, t, h, w, c_in = x.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        t_out = (t - kt) // st + 1

        # Loaded 5D weights are PyTorch-layout [O, I, D, H, W] (the VAE
        # loader transposes only 4D Conv2d weights, not 5D CausalConv3d).
        # Build a 2D conv weight [O, kh, kw, kt*c_in] whose last axis is
        # ordered (di, ci) — di outer, ci inner — to match the window
        # flatten below (time slices concatenated per-channel).
        # [O, I, D, H, W] -> [O, H, W, D, I] -> [O, kh, kw, kt*c_in]
        # Last axis ordered (di, ci) — di outer, ci inner — to match the
        # window flatten below (transpose(0,2,3,1,4) gives [B,H,W,kt,C] ->
        # [B,H,W,kt*C] with di outer, ci inner).
        # Axes: [O,ci,di,kh,kw] -> [O,kh,kw,di,ci] = (0,3,4,2,1).
        w_2d = self.weight.transpose(0, 3, 4, 2, 1).reshape(
            self.weight.shape[0], kh, kw, kt * c_in
        )
        outputs = []
        for t_i in range(t_out):
            t_start = t_i * st
            window = x[:, t_start : t_start + kt]
            # [B, kt, H, W, C] -> [B, H, W, kt*C] (di outer, ci inner)
            window = window.transpose(0, 2, 3, 1, 4).reshape(b, h, w, kt * c_in)
            out_2d = mx.conv2d(window, w_2d, stride=(sh, sw)) + self.bias
            outputs.append(out_2d)
        return mx.stack(outputs, axis=1)


class RMS_norm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True):
        super().__init__()
        self.channel_first = channel_first
        self.scale = dim**0.5
        if channel_first:
            broadcastable = (1, 1) if images else (1, 1, 1)
            self.gamma = mx.ones((dim, *broadcastable))
        else:
            self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm_dim = 1 if self.channel_first else -1
        # L2 normalize along channel dim (matches F.normalize)
        norm = mx.sqrt(
            mx.clip(
                mx.sum(x * x, axis=norm_dim, keepdims=True), a_min=1e-12, a_max=None
            )
        )
        return (x / norm) * self.scale * self.gamma


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.residual = [
            RMS_norm(in_dim, images=False),  # [0]
            None,  # [1] SiLU
            CausalConv3d(in_dim, out_dim, 3, padding=1),  # [2]
            RMS_norm(out_dim, images=False),  # [3]
            None,  # [4] SiLU
            None,  # [5] Dropout
            CausalConv3d(out_dim, out_dim, 3, padding=1),  # [6]
        ]
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else None

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None, final=False) -> mx.array:
        h = x if self.shortcut is None else self.shortcut(x)

        if feat_cache is not None:
            # First conv: norm -> silu -> [cache] -> conv
            x = nn.silu(self.residual[0](x))
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.residual[2](x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1

            # Second conv: norm -> silu -> [cache] -> conv
            x = nn.silu(self.residual[3](x))
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.residual[6](x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = nn.silu(self.residual[0](x))
            x = self.residual[2](x)
            x = nn.silu(self.residual[3](x))
            x = self.residual[6](x)

        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMS_norm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        b, c, t, h, w = x.shape

        # [B,C,T,H,W] -> [B,T,C,H,W] -> [BT,C,H,W] -> norm -> [BT,H,W,C]
        x = x.transpose(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        x = x.transpose(0, 2, 3, 1)  # [BT, H, W, C]

        qkv = self.to_qkv(x)  # [BT, H, W, 3C]
        qkv = qkv.reshape(b * t, h * w, 3, c).transpose(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q[:, None, :, :]  # [BT, 1, HW, C]
        k = k[:, None, :, :]
        v = v[:, None, :, :]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)
        out = out.squeeze(1).reshape(b * t, h, w, c)  # [BT, H, W, C]

        out = self.proj(out)  # [BT, H, W, C]
        out = out.reshape(b, t, h, w, c).transpose(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        return out + identity


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        assert mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d")
        self.mode = mode
        self.dim = dim

        if mode.startswith("upsample"):
            # resample.0 = Upsample (no params), resample.1 = Conv2d
            self.resample = [None, nn.Conv2d(dim, dim // 2, 3, padding=1)]
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
                )
        else:
            # resample.0 = ZeroPad2d (no params), resample.1 = Conv2d(stride=2)
            self.resample = [None, nn.Conv2d(dim, dim, 3, stride=2)]
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
                )

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None, final=False) -> mx.array:
        b, c, t, h, w = x.shape

        if self.mode == "upsample3d":
            # Upstream Resample.forward upsample3d (vae.py:104-123): the
            # temporal upsample (time_conv + interleave to t*2) happens ONLY
            # in the cache 'else' branch (a subsequent chunk). The first
            # cached call sets a 'Rep' marker and does NO temporal upsample
            # (spatial only). With feat_cache=None there is NEVER a temporal
            # upsample — upsample3d degrades to spatial-only. Temporal
            # frames are produced across latent chunks via the cache, not
            # within a single non-cached pass (issue #458).
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :]
                    if feat_cache[idx] == "Rep":
                        x_t = self.time_conv(x)
                    else:
                        x_t = self.time_conv(
                            mx.concatenate([feat_cache[idx], x], axis=2)
                        )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x_t = x_t.reshape(b, 2, c, t, h, w)
                    x = mx.stack([x_t[:, 0], x_t[:, 1]], axis=3).reshape(
                        b, c, t * 2, h, w
                    )
                    t = t * 2
            # feat_cache is None: spatial-only (no temporal upsample).

        if self.mode.startswith("upsample"):
            # Per-frame spatial upsample: nearest 2x + Conv2d
            x = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)  # [BT, H, W, C]
            x = mx.repeat(x, 2, axis=1)
            x = mx.repeat(x, 2, axis=2)
            x = self.resample[1](x)  # Conv2d [BT, 2H, 2W, C//2]
            c_out = x.shape[-1]
            return x.reshape(b, t, h * 2, w * 2, c_out).transpose(0, 4, 1, 2, 3)
        else:
            # Per-frame spatial downsample: ZeroPad(0,1,0,1) + Conv2d(stride=2)
            x = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)  # [BT, H, W, C]
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])  # ZeroPad2d(0,1,0,1)
            x = self.resample[1](x)  # Conv2d stride=2
            c_out = x.shape[-1]
            h_out, w_out = x.shape[1], x.shape[2]
            x = x.reshape(b, t, h_out, w_out, c_out).transpose(0, 4, 1, 2, 3)

            if self.mode == "downsample3d":
                if feat_cache is not None:
                    # Upstream Resample.forward downsample3d (vae.py:129-150):
                    #   slot idx     = last-frame cache (time_conv context)
                    #   slot idx+1   = deferred whole-frame buffer
                    #   feat_idx += 2 (two slots per downsample3d stage).
                    # First chunk: store x, RETURN x unchanged (no time_conv).
                    # Subsequent chunks: prepend cached last frame, run
                    # time_conv, splice any deferred frames. A 1-frame
                    # non-final result is deferred and None returned.
                    idx = feat_idx[0]
                    if feat_cache[idx] is None:
                        feat_cache[idx] = x
                    else:
                        cached = feat_cache[idx][:, :, -1:, :, :]
                        cache_x = x[:, :, -1:, :, :]
                        x = self.time_conv(mx.concatenate([cached, x], axis=2))
                        feat_cache[idx] = cache_x

                        deferred_x = feat_cache[idx + 1]
                        if deferred_x is not None:
                            x = mx.concatenate([deferred_x, x], axis=2)
                            feat_cache[idx + 1] = None

                        if x.shape[2] == 1 and not final:
                            feat_cache[idx + 1] = x
                            x = None
                    feat_idx[0] += 2
                else:
                    x = self.time_conv(x)
            return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list = None,
        num_res_blocks: int = 2,
        temporal_upsample: list = None,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temporal_upsample is None:
            temporal_upsample = [True, True, False]

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # Middle: [ResBlock, AttentionBlock, ResBlock]
        self.middle = [
            ResidualBlock(dims[0], dims[0]),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0]),
        ]

        # Flat upsample list matching original nn.Sequential indexing
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temporal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
        self.upsamples = upsamples

        # Output head: [RMS_norm, SiLU (no params), CausalConv3d]
        self.head = [
            RMS_norm(dims[-1], images=False),  # [0]
            None,  # [1] SiLU
            CausalConv3d(dims[-1], 3, 3, padding=1),  # [2]
        ]

    def run_up(self, layer_idx, x_ref, feat_cache, feat_idx, out_chunks):
        # Upstream Decoder3d.run_up (vae.py:363-398): recursive split into
        # 2-frame sub-chunks at each upsample3d. The causal structure REQUIRES
        # this split — a flat sequential pass upsamples all frames together
        # and produces wrong frame counts/values (issue #458).
        x = x_ref[0]
        x_ref[0] = None
        if layer_idx >= len(self.upsamples):
            x = nn.silu(self.head[0](x))
            if feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    cache_x = mx.concatenate(
                        [feat_cache[idx][:, :, -1:], cache_x], axis=2
                    )
                x = self.head[2](x, cache_x=feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = self.head[2](x)
            out_chunks.append(x)
            return

        layer = self.upsamples[layer_idx]
        if feat_cache is not None and isinstance(
            layer, (ResidualBlock, Resample)
        ):
            x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            x = layer(x)

        if (
            isinstance(layer, Resample)
            and layer.mode == "upsample3d"
            and x.shape[2] > 2
        ):
            for frame_idx in range(0, x.shape[2], 2):
                self.run_up(
                    layer_idx + 1,
                    [x[:, :, frame_idx : frame_idx + 2, :, :]],
                    feat_cache,
                    list(feat_idx),
                    out_chunks,
                )
            return

        self.run_up(layer_idx + 1, [x], feat_cache, feat_idx, out_chunks)

    def __call__(
        self, x: mx.array, feat_cache=None, feat_idx=None
    ) -> list:
        # Returns a LIST of output chunks (upstream forward returns out_chunks).
        # WanVAE.decode concatenates the lists from each latent chunk.
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate(
                    [feat_cache[idx][:, :, -1:], cache_x], axis=2
                )
            x = self.conv1(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if feat_cache is not None and isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        out_chunks = []
        self.run_up(0, [x], feat_cache, feat_idx, out_chunks)
        return out_chunks


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list = None,
        num_res_blocks: int = 2,
        temporal_downsample: list = None,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]

        dims = [dim * u for u in [1] + dim_mult]

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # Flat downsample list matching original nn.Sequential indexing
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temporal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
        self.downsamples = downsamples

        # Middle: [ResBlock, AttentionBlock, ResBlock]
        self.middle = [
            ResidualBlock(dims[-1], dims[-1]),
            AttentionBlock(dims[-1]),
            ResidualBlock(dims[-1], dims[-1]),
        ]

        # Output head: [RMS_norm, SiLU (no params), CausalConv3d]
        self.head = [
            RMS_norm(dims[-1], images=False),
            None,  # SiLU
            CausalConv3d(dims[-1], z_dim, 3, padding=1),
        ]

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None, final=False) -> mx.array:
        if feat_cache is not None:
            # conv1 with caching
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.conv1(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feat_cache is not None and isinstance(layer, (ResidualBlock, Resample)):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx, final=final)
                if x is None:
                    return None
            else:
                x = layer(x)

        for layer in self.middle:
            if feat_cache is not None and isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx, final=final)
            else:
                x = layer(x)

        if feat_cache is not None:
            # Head: norm -> silu -> [cache] -> conv
            x = nn.silu(self.head[0](x))
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.head[2](x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = nn.silu(self.head[0](x))
            x = self.head[2](x)

        return x


class WanVAE(nn.Module):
    def __init__(self, z_dim: int = 16, encoder: bool = False):
        super().__init__()
        self.z_dim = z_dim
        self.mean = mx.array(VAE_MEAN)
        self.std = mx.array(VAE_STD)
        self.inv_std = 1.0 / self.std

        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim=96, z_dim=z_dim)

        if encoder:
            self.encoder = Encoder3d(dim=96, z_dim=z_dim * 2)
            self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)

    def encode(self, x: mx.array) -> mx.array:
        # Streaming causal encode matching upstream WanVAE.encode: chunk the
        # time axis as 1, then 2,2,2... (not 1,4,4). The 1+2N pattern is what
        # the downsample3d deferred-frame cache expects; 1+4N corrupts the
        # temporal context and produces a latent uncorrelated to upstream
        # (issue #458). A chunk may return None when a downsample3d stage
        # defers a single frame to the next chunk.
        num_slots = self._count_encoder_cache_slots()
        feat_cache = [None] * num_slots

        t = x.shape[2]
        t = 1 + ((t - 1) // 2) * 2  # round down to 1+2N like upstream
        iter_ = 1 + (t - 1) // 2

        out = None
        for i in range(iter_):
            feat_idx = [0]
            if i == 0:
                chunk = x[:, :, :1]
                final = iter_ == 1
            else:
                chunk = x[:, :, 1 + 2 * (i - 1) : 1 + 2 * i]
                final = i == iter_ - 1

            chunk_out = self.encoder(
                chunk, feat_cache=feat_cache, feat_idx=feat_idx, final=final
            )

            if chunk_out is None:
                continue
            if out is None:
                out = chunk_out
            else:
                out = mx.concatenate([out, chunk_out], axis=2)

        mu, _ = mx.split(self.conv1(out), 2, axis=1)

        # Normalize: (mu - mean) * inv_std
        mean = self.mean.reshape(1, -1, 1, 1, 1)
        inv_std = self.inv_std.reshape(1, -1, 1, 1, 1)
        return (mu - mean) * inv_std

    def _count_encoder_cache_slots(self) -> int:
        # Match upstream count_cache_layers: every CausalConv3d (5D weight)
        # plus every downsample3d Resample (which reserves a deferred-frame
        # slot in addition to the cache slot its time_conv CausalConv3d
        # already contributes).
        import mlx.nn as nn

        cc = sum(
            1
            for k, v in nn.utils.tree_flatten(self.encoder.parameters())
            if hasattr(v, "ndim") and v.ndim == 5
        )
        ds3d = sum(
            1
            for layer in self.encoder.downsamples
            if isinstance(layer, Resample) and layer.mode == "downsample3d"
        )
        return cc + ds3d

    def decode(self, z: mx.array) -> mx.array:
        # Upstream WanVAE.decode (vae.py:491-511): chunk latents as 1,2,2...
        # (iter_ = 1 + z.shape[2]//2), decode each chunk through the recursive
        # decoder (which returns a LIST of output sub-chunks), extend the
        # output list across chunks, then concatenate along time. A flat
        # single-pass decode produces the wrong frame count (8 vs 5) because
        # the causal upsample3d recursion must split per 2-frame group
        # (issue #458).
        mean = self.mean.reshape(1, -1, 1, 1, 1)
        inv_std = self.inv_std.reshape(1, -1, 1, 1, 1)
        z = z / inv_std + mean

        x = self.conv2(z)
        iter_ = 1 + z.shape[2] // 2
        feat_map = None
        if iter_ > 1:
            feat_map = [None] * self._count_decoder_cache_slots()

        out_chunks = None
        for i in range(iter_):
            feat_idx = [0]
            if i == 0:
                chunk = x[:, :, :1]
            else:
                chunk = x[:, :, 1 + 2 * (i - 1) : 1 + 2 * i]
            chunk_out = self.decoder(chunk, feat_cache=feat_map, feat_idx=feat_idx)
            if out_chunks is None:
                out_chunks = chunk_out
            else:
                out_chunks.extend(chunk_out)

        out = mx.concatenate(out_chunks, axis=2)
        # Upstream WanVAE.decode does NOT clip — clamping to [-1,1] is the
        # caller's responsibility (done at save time, e.g. (x*255+127.5)).
        # Clipping here distorts outputs whose head exceeds the range.
        return out

    def _count_decoder_cache_slots(self) -> int:
        # Match upstream count_cache_layers: every CausalConv3d (5D weight).
        # Decoder has no downsample3d, so no extra deferred slots.
        import mlx.nn as nn

        return sum(
            1
            for k, v in nn.utils.tree_flatten(self.decoder.parameters())
            if hasattr(v, "ndim") and v.ndim == 5
        )

    def decode_tiled(self, z: mx.array, tiling_config=None) -> mx.array:
        from .tiling import TilingConfig, decode_with_tiling

        if tiling_config is None:
            tiling_config = TilingConfig.default()

        # Check if tiling is actually needed
        _, _, f, h, w = z.shape
        needs_tiling = False
        if tiling_config.spatial_config is not None:
            s_tile = tiling_config.spatial_config.tile_size_in_pixels // 8
            if h > s_tile or w > s_tile:
                needs_tiling = True
        if tiling_config.temporal_config is not None:
            t_tile = tiling_config.temporal_config.tile_size_in_frames // 4
            if f > t_tile:
                needs_tiling = True

        if not needs_tiling:
            return self.decode(z)

        # Denormalize once (small tensor), then tile the denormalized latents
        mean = self.mean.reshape(1, -1, 1, 1, 1)
        inv_std = self.inv_std.reshape(1, -1, 1, 1, 1)
        z_denorm = z / inv_std + mean

        def tile_decode(tile_latents, **kwargs):
            x = self.conv2(tile_latents)
            iter_ = 1 + tile_latents.shape[2] // 2
            feat_map = None
            if iter_ > 1:
                feat_map = [None] * self._count_decoder_cache_slots()
            out_chunks = None
            for i in range(iter_):
                feat_idx = [0]
                if i == 0:
                    chunk = x[:, :, :1]
                else:
                    chunk = x[:, :, 1 + 2 * (i - 1) : 1 + 2 * i]
                chunk_out = self.decoder(
                    chunk, feat_cache=feat_map, feat_idx=feat_idx
                )
                if out_chunks is None:
                    out_chunks = chunk_out
                else:
                    out_chunks.extend(chunk_out)
            out = mx.concatenate(out_chunks, axis=2)
            return out

        return decode_with_tiling(
            decoder_fn=tile_decode,
            latents=z_denorm,
            tiling_config=tiling_config,
            spatial_scale=8,  # 3× spatial 2× upsamples = 8×
            temporal_scale=4,  # 2× temporal upsamples × 2 = 4×
            causal_temporal=False,  # Wan2.1 uses non-causal temporal (T → 4T)
        )
