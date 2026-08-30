# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 VisualVAE 纯 MLX 移植：3D 因果 CNN 编码器 + ViT3D 解码器。
# 源码：/Users/dahai/minimax/MiniMax-H3/FL2VA/video_vae/{klvae,vae_cnn,vae_vit,
# base_module,attention,func,norm,conv,normalize,vae_module}.py
#
# 编码器：EncoderFCN3D（因果 3D 卷积 + GroupNorm，use_t_isolated_gn）
# 解码器：ViT3DDecoder（patch_size=16 / patch_size_t=4，3D RoPE，register tokens）
# 权重布局对齐 safetensors key tree，from_pretrained 仅做 prefix 映射。
#
# GOTCHA（移植自 Cosmos/Wan2 经验）：
#   - nn.GroupNorm channel-last：5D [B,C,T,H,W] 需转 [B*T,H*W,C] 再归一
#   - Conv3d 权重 PyTorch [O,I,D,H,W]，MLX 卷积需 channel-last 重排
#   - use_t_isolated_gn：时间维独立归一（merge time to batch 再 GroupNorm）
#   - causal time padding：左侧补零（2*padding），不取未来帧
import glob
import logging
import math
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import H3VAEConfig

logger = logging.getLogger(__name__)

# ImageNet 归一化常量（源自 normalize.py pixel_norm_type="imagenet")
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


def _silu(x):
    return x * mx.sigmoid(x)


def _gelu(x):
    return x * (mx.sigmoid(x) + 1.0) * 0.5  # tanh 近似 GELU


def _normalize_pixel(x):
    mean = mx.array(NORM_MEAN, dtype=x.dtype).reshape(1, 3, 1, 1, 1)
    std = mx.array(NORM_STD, dtype=x.dtype).reshape(1, 3, 1, 1, 1)
    return (x - mean) / std


def _denormalize_pixel(x):
    mean = mx.array(NORM_MEAN, dtype=x.dtype).reshape(1, 3, 1, 1, 1)
    std = mx.array(NORM_STD, dtype=x.dtype).reshape(1, 3, 1, 1, 1)
    return x * std + mean


# ============================================================================
# 3D 因果卷积（移植自 wan2/vae.py CausalConv3d + klvae conv.py BaseConv3d）
# ============================================================================


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        causal=True,
        pad_mode="constant",
        pad_mode_t="constant",
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
        self.padding = padding
        self.causal = causal
        self.pad_mode = pad_mode
        self.pad_mode_t = pad_mode_t
        # 因果时间 padding：左侧补 2*padding[0]，右侧不补（不取未来帧）
        self._causal_pad_t = 2 * padding[0] if causal else padding[0]
        self._pad_h = padding[1]
        self._pad_w = padding[2]

        w_shape = (
            out_channels,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            in_channels,
        )
        self.weight = mx.array(np.zeros(w_shape, dtype=np.float32))
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x):
        b, c, t, h, w = x.shape

        # 时间维 padding
        if t > 1:
            if self.causal:
                pad_t = mx.zeros((b, c, self._causal_pad_t, h, w), dtype=x.dtype)
                x = mx.concatenate([pad_t, x], axis=2)
            else:
                p = self._causal_pad_t
                if p > 0:
                    pad_pre = mx.zeros((b, c, p, h, w), dtype=x.dtype)
                    pad_post = mx.zeros((b, c, p, h, w), dtype=x.dtype)
                    x = mx.concatenate([pad_pre, x, pad_post], axis=2)
        else:
            # 单帧：直接复制扩展到 kernel 时间长度
            kt = self.kernel_size[0]
            if kt > 1:
                x = mx.repeat(x, kt, axis=2)

        # 空间维 padding
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

    def _conv3d(self, x):
        b, t, h, w, c_in = x.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        t_out = (t - kt) // st + 1

        # 权重 [O,kt,kh,kw,c_in] -> [O,kh,kw,kt,c_in] -> reshape [O,kh,kw,kt*c_in]
        # window 为 [B,H,W,kt,c_in]（见下方 transpose(0,2,3,1,4)），权重须同序：
        # kt 外、c_in 内。原 transpose(0,3,4,2,1) 错排通道，致编码器 forward 全错。
        w_2d = self.weight.transpose(0, 2, 3, 1, 4).reshape(
            self.weight.shape[0], kh, kw, kt * c_in
        )
        outputs = []
        for t_i in range(t_out):
            t_start = t_i * st
            window = x[:, t_start : t_start + kt]
            window = window.transpose(0, 2, 3, 1, 4).reshape(b, h, w, kt * c_in)
            out_2d = mx.conv2d(window, w_2d, stride=(sh, sw))
            if self.bias is not None:
                out_2d = out_2d + self.bias
            outputs.append(out_2d)
        return mx.stack(outputs, axis=1)


class PointwiseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, causal=True):
        super().__init__()
        # 1x1x1 conv = 纯线性：权重 [O, I]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = mx.array(np.zeros((out_channels, in_channels), dtype=np.float32))
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x):
        b, c, t, h, w = x.shape
        # channel-last -> [B*T*H*W, C] @ [C, O] -> [B*T*H*W, O]
        x = x.transpose(0, 2, 3, 4, 1).reshape(b * t * h * w, c)
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        y = y.reshape(b, t, h, w, self.out_channels).transpose(0, 4, 1, 2, 3)
        return y


# ============================================================================
# GroupNorm（移植自 cosmos/vae.py _CausalNorm + klvae norm.py）
# use_t_isolated_gn=True：时间维独立归一（merge time to batch）
# ============================================================================


class GroupNorm3D(nn.Module):
    def __init__(self, channels, num_groups=32, eps=1e-6, use_t_isolated_gn=False):
        super().__init__()
        self.num_groups = num_groups
        self.channels = channels
        self.eps = eps
        self.use_t_isolated_gn = use_t_isolated_gn
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x):
        b, c, t, h, w = x.shape
        if self.use_t_isolated_gn and t > 1:
            # 时间维独立：merge time to batch
            x = x.transpose(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            out = self._group_norm_4d(x)
            out = out.reshape(b, t, c, h, w).transpose(0, 2, 1, 3, 4)
            return out
        return self._group_norm_5d(x)

    def _group_norm_5d(self, x):
        b, c, t, h, w = x.shape
        x = x.transpose(0, 2, 3, 4, 1).reshape(b * t * h * w, c)
        out = self._group_norm(x)
        return out.reshape(b, t, h, w, c).transpose(0, 4, 1, 2, 3)

    def _group_norm_4d(self, x):
        b, c, h, w = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(b * h * w, c)
        out = self._group_norm(x)
        return out.reshape(b, h, w, c).transpose(0, 3, 1, 2)

    def _group_norm(self, x):
        # x: [N, C]
        n = x.shape[0]
        xg = x.reshape(n, self.num_groups, self.channels // self.num_groups)
        mean = xg.mean(axis=-1, keepdims=True)
        var = xg.var(axis=-1, keepdims=True)
        xg = (xg - mean) / mx.sqrt(var + self.eps)
        xg = xg.reshape(n, self.channels)
        return xg * self.weight + self.bias


# ============================================================================
# 编码器：ResnetBlock3D + Downsample3D + EncoderFCN3D
# ============================================================================


class ResnetBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, use_t_isolated_gn=False, causal=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = GroupNorm3D(in_ch, use_t_isolated_gn=use_t_isolated_gn)
        self.conv1 = CausalConv3d(
            in_ch, out_ch, kernel_size=3, padding=1, causal=causal
        )
        self.norm2 = GroupNorm3D(out_ch, use_t_isolated_gn=use_t_isolated_gn)
        self.conv2 = CausalConv3d(
            out_ch, out_ch, kernel_size=3, padding=1, causal=causal
        )
        if in_ch != out_ch:
            self.nin_shortcut = PointwiseConv3d(in_ch, out_ch, causal=causal)
        else:
            self.nin_shortcut = None

    def __call__(self, x):
        h = _silu(self.norm1(x))
        h = self.conv1(h)
        h = _silu(self.norm2(h))
        h = self.conv2(h)
        residual = x
        if self.nin_shortcut is not None:
            residual = self.nin_shortcut(x)
        return residual + h


class Downsample3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_stride=1, space_stride=2, causal=True):
        super().__init__()
        self.time_stride = time_stride
        self.space_stride = space_stride
        self.conv = CausalConv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=(time_stride, space_stride, space_stride),
            padding=(1, 0, 0),
            causal=causal,
        )

    def __call__(self, x):
        if self.space_stride == 2:
            x = mx.pad(
                x,
                [(0, 0), (0, 0), (0, 0), (0, 1), (0, 1)],
            )
        return self.conv(x)


class EncoderFCN3D(nn.Module):
    def __init__(
        self,
        ch,
        ch_mult,
        space_down,
        time_down,
        num_res_blocks,
        in_channels,
        z_channels,
        use_t_isolated_gn=False,
        causal=True,
    ):
        super().__init__()
        self.num_levels = len(ch_mult)
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * self.num_levels
        self.num_res_blocks = num_res_blocks

        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        block_out = block_mid

        self.conv_in = CausalConv3d(
            in_channels, block_in[0], kernel_size=3, padding=1, causal=causal
        )

        self.down = []
        for i_level in range(self.num_levels):
            block_list = []
            for i in range(self.num_res_blocks[i_level]):
                in_c = block_in[i_level] if i == 0 else block_mid[i_level]
                block_list.append(
                    ResnetBlock3D(
                        in_c,
                        block_mid[i_level],
                        use_t_isolated_gn=use_t_isolated_gn,
                        causal=causal,
                    )
                )
            downsample = None
            if space_down[i_level] * time_down[i_level] > 1:
                downsample = Downsample3D(
                    block_mid[i_level],
                    block_out[i_level],
                    time_stride=time_down[i_level],
                    space_stride=space_down[i_level],
                    causal=causal,
                )
            elif block_out[i_level] != block_mid[i_level]:
                downsample = PointwiseConv3d(
                    block_mid[i_level], block_out[i_level], causal=causal
                )
            self.down.append({"block": block_list, "downsample": downsample})

        self.norm_out = GroupNorm3D(block_out[-1], use_t_isolated_gn=use_t_isolated_gn)
        self.conv_out = CausalConv3d(
            block_out[-1],
            z_channels * 2,  # double_z=True
            kernel_size=3,
            padding=1,
            causal=causal,
        )

    def __call__(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for block in self.down[i_level]["block"]:
                h = block(h)
            ds = self.down[i_level]["downsample"]
            if ds is not None:
                h = ds(h)
        h = _silu(self.norm_out(h))
        h = self.conv_out(h)
        return h


# ============================================================================
# 3D RoPE + ViT3D 解码器
# ============================================================================


def create_token_ids(patch_dims, dtype):
    # length_normalized：coords = arange(0.5, n)/n*2-1
    coords_list = []
    for dim_size in patch_dims:
        coords = (mx.arange(dim_size, dtype=dtype) + 0.5) / dim_size
        coords = coords * 2.0 - 1.0
        coords_list.append(coords)
    # meshgrid ij -> stack -> flatten
    grid = mx.meshgrid(*coords_list, indexing="ij")
    coords = mx.stack(grid, axis=-1)
    coords = coords.reshape(-1, len(patch_dims))
    return mx.expand_dims(coords, axis=0)  # [1, N, D]


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=100.0, n_dim=3, use_angle=True):
        super().__init__()
        self.dim = dim
        self.n_dim = n_dim
        self.angle_scale = 2.0 * math.pi if use_angle else 1.0
        # inv_freq = 1 / base^(arange(0, 1, 2*n_dim/dim))
        num_steps = int(math.ceil(dim / (2 * n_dim)))
        steps = mx.arange(num_steps, dtype=mx.float32) * (2 * n_dim / dim)
        self.inv_freq = 1.0 / mx.power(float(rotary_base), steps)

    def __call__(self, img_ids):
        # img_ids: [B, N, D]
        b, n, d = img_ids.shape
        angles = (
            self.angle_scale
            * img_ids[:, :, :, None]
            * self.inv_freq[None, None, None, :]
        )  # [B, N, D, num_steps]
        angles = angles.reshape(b, n, -1)  # [B, N, D*num_steps] = [B, N, dim/2]
        angles = mx.concatenate([angles, angles], axis=-1)  # tile(2) -> [B, N, dim]
        angles = mx.expand_dims(angles, axis=2)  # [B, N, 1, dim]
        cos = mx.cos(angles)
        sin = mx.sin(angles)
        return cos.astype(img_ids.dtype), sin.astype(img_ids.dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(t, rotary_pos_emb):
    cos, sin = rotary_pos_emb
    cos = cos.astype(t.dtype)
    sin = sin.astype(t.dtype)
    rot_dim = cos.shape[-1]
    t_dim = t.shape[-1]
    if rot_dim < t_dim:
        t_rot = t[..., :rot_dim]
        t_pass = t[..., rot_dim:]
        t_rot = t_rot * cos + _rotate_half(t_rot) * sin
        return mx.concatenate([t_rot, t_pass], axis=-1)
    return t * cos + _rotate_half(t) * sin


class FeedForward(nn.Module):
    def __init__(self, dim, use_gated=True, bias=True, mult=4):
        super().__init__()
        self.use_gated = use_gated
        # ViT 解码器 FF：inner_dim = dim*mult（mult=4），gated 时 w1 输出 2*inner_dim。
        # 真实 ckpt dim=2048 → inner_dim=8192，w1=(16384,2048) w2=(2048,8192)。
        inner_dim = dim * mult
        self.inner_dim = inner_dim
        if use_gated:
            self.w1 = nn.Linear(dim, inner_dim * 2, bias=bias)
        else:
            self.w1 = nn.Linear(dim, inner_dim, bias=bias)
        self.w2 = nn.Linear(inner_dim, dim, bias=bias)

    def __call__(self, x):
        h = self.w1(x)
        if self.use_gated:
            gate, hidden = mx.split(h, 2, axis=-1)
            h = _silu(gate) * hidden
        else:
            h = _silu(h)
        return self.w2(h)


class Attention(nn.Module):
    def __init__(
        self,
        heads,
        dim_head,
        embed_dim,
        bias=True,
        eps=1e-5,
        qk_norm_type="rms_norm",
        qk_norm_affine=False,
    ):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.embed_dim = embed_dim
        self.to_qkv = nn.Linear(embed_dim, self.inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(self.inner_dim, embed_dim, bias=bias)
        # 官方 source/config.json: qk_norm_type="rms_norm", qk_norm_affine=false
        # 对 q,k 在 RoPE 之前做无权重 RMSNorm（仅缩放，无可学习参数）。
        self.qk_norm_type = qk_norm_type
        self.qk_norm_affine = qk_norm_affine
        self.qk_norm_eps = eps
        if qk_norm_affine:
            self.qk_norm_weight = mx.ones((dim_head,))

    def _qk_norm(self, x):
        if self.qk_norm_type != "rms_norm":
            return x
        rms = mx.sqrt((x * x).mean(axis=-1, keepdims=True) + self.qk_norm_eps)
        out = x / rms
        if self.qk_norm_affine:
            return out * self.qk_norm_weight
        return out

    def __call__(self, hidden_states, rotary_pos_emb=None):
        b, n, _ = hidden_states.shape
        qkv = self.to_qkv(hidden_states)
        qkv = qkv.reshape(b, n, -1, 3 * self.dim_head)
        q, k, v = mx.split(qkv, 3, axis=-1)
        # [b, n, heads, dim_head]
        q = q.reshape(b, n, self.heads, self.dim_head)
        k = k.reshape(b, n, self.heads, self.dim_head)
        v = v.reshape(b, n, self.heads, self.dim_head)
        # qk_norm（rms_norm, 无 affine）在 RoPE 之前，对齐官方 attention.py
        q = self._qk_norm(q)
        k = self._qk_norm(k)
        if rotary_pos_emb is not None:
            # cos/sin: [b, n, 1, dim]，在 [b, n, heads, dim] 上广播
            q = apply_rotary_pos_emb(q, rotary_pos_emb)
            k = apply_rotary_pos_emb(k, rotary_pos_emb)
        # -> [b, heads, n, dim_head] for sdpa
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.dim_head)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, self.inner_dim)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        heads,
        dim_head,
        embed_dim=None,
        norm_type="rms_norm",
        norm_affine=True,
        ffn_use_gated=True,
        bias=True,
        eps=1e-5,
        use_scale=True,
        qk_norm_type="rms_norm",
        qk_norm_affine=False,
    ):
        super().__init__()
        dim = embed_dim if embed_dim is not None else dim_head * heads
        self.use_scale = use_scale
        if norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(dim, eps=eps)
            self.norm2 = nn.LayerNorm(dim, eps=eps)
        else:
            self.norm1 = nn.RMSNorm(dim, eps=eps)
            self.norm2 = nn.RMSNorm(dim, eps=eps)
        self.attn = Attention(
            heads,
            dim_head,
            dim,
            bias=bias,
            eps=eps,
            qk_norm_type=qk_norm_type,
            qk_norm_affine=qk_norm_affine,
        )
        if use_scale:
            self.scale1 = mx.zeros((dim,))
            self.scale2 = mx.zeros((dim,))
        else:
            self.scale1 = None
            self.scale2 = None
        self.ff = FeedForward(dim, use_gated=ffn_use_gated, bias=bias)

    def __call__(self, hidden_states, rotary_pos_emb=None):
        norm_h = self.norm1(hidden_states).astype(hidden_states.dtype)
        attn_out = self.attn(norm_h, rotary_pos_emb)
        if self.scale1 is not None:
            hidden_states = hidden_states + attn_out * self.scale1
        else:
            hidden_states = hidden_states + attn_out
        norm_h = self.norm2(hidden_states).astype(hidden_states.dtype)
        ff_out = self.ff(norm_h)
        if self.scale2 is not None:
            hidden_states = hidden_states + ff_out * self.scale2
        else:
            hidden_states = hidden_states + ff_out
        return hidden_states


def _pack_tensors_3d(tensors, patch_size, patch_size_t):
    b, c, t, h, w = tensors.shape
    tensors = tensors.reshape(
        b,
        c,
        t // patch_size_t,
        patch_size_t,
        h // patch_size,
        patch_size,
        w // patch_size,
        patch_size,
    )
    tensors = tensors.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return tensors.reshape(
        b,
        (t // patch_size_t) * (h // patch_size) * (w // patch_size),
        c * patch_size_t * patch_size * patch_size,
    )


def _unpack_tensors_3d(tensors, patch_size, patch_size_t, t, h, w):
    b, num_patches, channels = tensors.shape
    num_c = channels // (patch_size_t * patch_size * patch_size)
    tensors = tensors.reshape(
        b,
        t // patch_size_t,
        h // patch_size,
        w // patch_size,
        num_c,
        patch_size_t,
        patch_size,
        patch_size,
    )
    tensors = tensors.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return tensors.reshape(b, num_c, t, h, w)


class ViT3DDecoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        patch_size_t=4,
        t_causal=False,
        in_channels=24,
        out_channels=3,
        num_layers=36,
        heads=32,
        dim_head=64,
        norm_type="rms_norm",
        norm_affine=True,
        ffn_use_gated=True,
        rope_theta=100.0,
        rope_dim_ratio=0.75,
        bias=True,
        eps=1e-5,
        num_register_tokens=4,
        qk_norm_type="rms_norm",
        qk_norm_affine=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.t_causal = t_causal
        self.num_register_tokens = num_register_tokens

        dim = heads * dim_head
        self.dim = dim
        rope_apply_dim = int(dim_head * rope_dim_ratio)
        self.pos_embed = RotaryEmbeddingND(
            rope_apply_dim, rope_theta, n_dim=3, use_angle=True
        )
        self.x_embedder = nn.Linear(in_channels, dim, bias=bias)
        if num_register_tokens > 0:
            self.register_tokens = mx.zeros((1, num_register_tokens, dim)) * 0.02
        else:
            self.register_tokens = None
        self.transformer_blocks = [
            TransformerBlock(
                heads=heads,
                dim_head=dim_head,
                embed_dim=dim,
                norm_type=norm_type,
                norm_affine=norm_affine,
                ffn_use_gated=ffn_use_gated,
                bias=bias,
                eps=eps,
                qk_norm_type=qk_norm_type,
                qk_norm_affine=qk_norm_affine,
            )
            for _ in range(num_layers)
        ]
        self.norm_out = nn.LayerNorm(dim, eps=eps)
        patch_dim = out_channels * patch_size_t * patch_size * patch_size
        self.proj_out = nn.Linear(dim, patch_dim, bias=bias)

    def __call__(self, x):
        b, c, lt, lh, lw = x.shape
        num_suffix = 1 + self.num_register_tokens

        hidden_states = _pack_tensors_3d(x, 1, 1)
        hidden_states = self.x_embedder(hidden_states)
        num_patches = hidden_states.shape[1]

        tokens = [hidden_states]
        if self.register_tokens is not None:
            reg = mx.broadcast_to(
                self.register_tokens, (b, self.num_register_tokens, self.dim)
            )
            tokens.append(reg)
        # cls token（全零占位，has_cls_token=False 但 forward 追加零 cls）
        cls_token = mx.zeros((b, 1, self.dim), dtype=hidden_states.dtype)
        tokens.append(cls_token)
        hidden_states = mx.concatenate(tokens, axis=1)

        latent_size = (lt, lh, lw)
        img_ids = create_token_ids(latent_size, x.dtype)
        img_ids = mx.broadcast_to(img_ids, (b, img_ids.shape[1], img_ids.shape[2]))
        suffix_ids = mx.zeros((b, num_suffix, 3), dtype=img_ids.dtype)
        img_ids = mx.concatenate([img_ids, suffix_ids], axis=1)

        rotary_pos_emb = self.pos_embed(img_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_pos_emb)

        hidden_states = self.norm_out(hidden_states)
        output = self.proj_out(hidden_states)
        output = output[:, :num_patches, :]

        video_t = lt * self.patch_size_t
        video_h = lh * self.patch_size
        video_w = lw * self.patch_size
        output = _unpack_tensors_3d(
            output, self.patch_size, self.patch_size_t, video_t, video_h, video_w
        )
        return output


# ============================================================================
# 对角高斯分布
# ============================================================================


class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        p = parameters.astype(mx.float32)
        self.mean, self.logvar = mx.split(p, 2, axis=1)
        self.logvar = mx.clip(self.logvar, -30.0, 20.0)
        self.std = mx.exp(0.5 * self.logvar)

    def sample(self):
        noise = mx.random.normal(self.mean.shape)
        return (self.mean + self.std * noise).astype(self.mean.dtype)


# ============================================================================
# AutoencoderKLLegacy（顶层封装）
# ============================================================================


class MiniMaxH3VideoVAE(nn.Module):
    def __init__(self, config: H3VAEConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or H3VAEConfig()
        self.config = cfg
        self.vae_ratio = cfg.vae_ratio
        self.vae_ratio_t = cfg.vae_ratio_t
        self.causal_encoder = cfg.causal_encoder
        self.causal_decoder = cfg.causal_decoder
        self.embed_dim = cfg.embed_dim
        self.z_channels = cfg.z_channels
        # 解码器空间分块（源自官方 klvae.tiled_decode + vae_processor）。
        # ViT3D 解码器在大空间 token 数下越界分布（1344×768 decoded DC=-1.15
        # vs 768×448 +0.81），官方 config vae_decoder_tiling=1 强制分块。
        # tile_size/overlap 为像素空间，latent = pixel//vae_ratio。
        self.decoder_tile_size = int(kwargs.get("decoder_tile_size", cfg.vae_tile_size))
        self.decoder_tile_overlap_min = int(
            kwargs.get("decoder_tile_overlap_min", cfg.vae_tile_overlap_min)
        )

        self.encoder = EncoderFCN3D(
            ch=cfg.ch,
            ch_mult=cfg.ch_mult,
            space_down=cfg.space_down,
            time_down=cfg.time_down,
            num_res_blocks=cfg.num_res_blocks,
            in_channels=cfg.in_channels,
            z_channels=cfg.z_channels,
            use_t_isolated_gn=cfg.use_t_isolated_gn,
            causal=cfg.causal_encoder,
        )
        # quant_conv / post_quant_conv: Conv3d 1x1x1
        self.quant_conv = PointwiseConv3d(cfg.z_channels * 2, 2 * cfg.embed_dim)
        self.post_quant_conv = PointwiseConv3d(cfg.embed_dim, cfg.z_channels)

        self.decoder = ViT3DDecoder(
            patch_size=cfg.vae_ratio,
            patch_size_t=cfg.vae_ratio_t,
            t_causal=cfg.causal_decoder,
            in_channels=cfg.z_channels,
            out_channels=cfg.in_channels,
            num_layers=cfg.vit_num_layers,
            heads=cfg.vit_heads,
            dim_head=cfg.vit_dim_head,
            norm_type=cfg.vit_norm_type,
            ffn_use_gated=cfg.vit_ffn_use_gated,
            rope_theta=cfg.vit_rope_theta,
            rope_dim_ratio=cfg.vit_rope_dim_ratio,
            qk_norm_type=cfg.vit_qk_norm_type,
            qk_norm_affine=cfg.vit_qk_norm_affine,
        )

    def encode(self, x):
        logger.info("h3 vae encode: input shape=%s", x.shape)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        logger.info("h3 vae encode: moments shape=%s", moments.shape)
        return moments

    def decode(self, z):
        logger.info("h3 vae decode: latent shape=%s", z.shape)
        # 空间分块：latent 空间 h/w 超过单 tile（像素 tile_size//vae_ratio）时分块
        # 解码再 blend，保持 ViT3D 解码器 token 数在训练分布内。
        latent_h, latent_w = z.shape[-2], z.shape[-1]
        latent_tile = self.decoder_tile_size // self.vae_ratio
        if latent_h > latent_tile or latent_w > latent_tile:
            dec = self.tiled_decode(z)
        else:
            z2 = self.post_quant_conv(z)
            dec = self.decoder(z2)
        logger.info("h3 vae decode: output shape=%s", dec.shape)
        return dec

    def _split_tiles(self, input_len):
        # 官方 klvae.split_tiles(is_decoder=True) 的纯 MLX 移植。
        # input_len 为像素空间长度，返回 (start_idx, tile_len, overlap) 像素空间。
        import math

        tile_size = self.decoder_tile_size
        overlap_min = self.decoder_tile_overlap_min
        if tile_size >= input_len:
            return [0], [input_len], []
        n = math.ceil(input_len / tile_size)
        while True:
            overlaps = [overlap_min] * (n - 1)
            remaining = tile_size * n - sum(overlaps) - input_len
            if remaining < 0:
                n += 1
            else:
                break
        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (n - 1)] += self.vae_ratio
        start_idx = [0]
        for i in range(n - 1):
            start_idx.append(start_idx[-1] + tile_size - overlaps[i])
        tile_len = [tile_size] * n
        return start_idx, tile_len, overlaps

    @staticmethod
    def _blend(a, b, blend_extent, dim):
        # 官方 klvae.blend：a 尾部与 b 头部线性交叉融合 blend_extent 个单元。
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = mx.arange(blend_extent, dtype=a.dtype)
        weight_a = 1.0 - positions / blend_extent
        weight_b = positions / blend_extent
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = weight_a.reshape(shape)
        weight_b = weight_b.reshape(shape)
        sl_a = [slice(None)] * a.ndim
        sl_a[dim] = slice(-blend_extent, None)
        a_overlap = a[tuple(sl_a)]
        sl_b = [slice(None)] * b.ndim
        sl_b[dim] = slice(0, blend_extent)
        b_overlap = b[tuple(sl_b)]
        blended = a_overlap * weight_a + b_overlap * weight_b
        if blend_extent < b.shape[dim]:
            sl_rest = [slice(None)] * b.ndim
            sl_rest[dim] = slice(blend_extent, None)
            b_rest = b[tuple(sl_rest)]
            return mx.concatenate([blended, b_rest], axis=dim)
        return blended

    def tiled_decode(self, z):
        # 官方 klvae.tiled_decode（单进程 sp_size=1）纯 MLX 移植。
        # z: (b,c,t,h,w) latent。按像素空间 h/w 分块，每块独立 decode 再 blend。
        height = z.shape[-2] * self.vae_ratio
        width = z.shape[-1] * self.vae_ratio
        y_idx, y_len, y_overlap = self._split_tiles(height)
        x_idx, x_len, x_overlap = self._split_tiles(width)
        i_max, j_max = len(y_idx), len(x_idx)

        rows = [[None] * j_max for _ in range(i_max)]
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            i_pos_l, i_len_l = i_pos // self.vae_ratio, i_len // self.vae_ratio
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                j_pos_l, j_len_l = j_pos // self.vae_ratio, j_len // self.vae_ratio
                tile = z[..., i_pos_l : i_pos_l + i_len_l, j_pos_l : j_pos_l + j_len_l]
                z2 = self.post_quant_conv(tile)
                rows[i][j] = self.decoder(z2)
                mx.eval(rows[i][j])

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., : -y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -x_overlap[j]]
                result_row.append(tile)
            result_rows.append(mx.concatenate(result_row, axis=-1))
        dec = mx.concatenate(result_rows, axis=-2)
        return dec

    def encode_base(self, x, process_image=False):
        if x.ndim == 4:
            x = mx.expand_dims(x, axis=2)
        moments = self.encode(x)
        z = DiagonalGaussianDistribution(moments).sample()
        return z

    @classmethod
    def from_pretrained(cls, model_path, config=None, **kwargs):
        cfg = config or H3VAEConfig()
        vae = cls(config=cfg)
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not safetensor_files:
            logger.warning("h3 vae: no safetensors at %s, random init", model_path)
            return vae
        from mlx.utils import tree_flatten, tree_unflatten

        all_params = {}
        for sf in safetensor_files:
            all_params.update(mx.load(sf))
        mapped = _remap_vae_weights(all_params)
        flat = tree_flatten(vae.parameters())
        flat_keys = {k for k, _ in flat}
        loaded = {}
        matched = 0
        for k, v in flat:
            if k in mapped:
                mv = mapped[k]
                if mv.dtype != mx.float16:
                    mv = mv.astype(mx.float16)
                if list(mv.shape) != list(v.shape):
                    logger.warning(
                        "h3 vae: shape mismatch %s ckpt=%s module=%s",
                        k,
                        tuple(mv.shape),
                        tuple(v.shape),
                    )
                    loaded[k] = v
                else:
                    loaded[k] = mv
                    matched += 1
            else:
                loaded[k] = v
        unmatched_ckpt = [ck for ck in mapped if ck not in flat_keys]
        vae.update(tree_unflatten(loaded))
        logger.info(
            "h3 vae from_pretrained: matched %d/%d, %d ckpt unmatched",
            matched,
            len(flat),
            len(unmatched_ckpt),
        )
        if unmatched_ckpt:
            logger.warning("h3 vae: unmatched ckpt keys: %s", unmatched_ckpt[:20])
        return vae


def _remap_vae_weights(params):
    # 源 checkpoint key tree（diffusers AutoencoderKLLegacy）：
    #   encoder.conv_in.{weight,bias}           [O,I,3,3,3]
    #   encoder.down.{i}.block.{j}.{norm1,norm2,conv1,conv2,nin_shortcut}.*
    #   encoder.down.{i}.downsample.conv.*       [O,I,3,3,3]
    #   encoder.norm_out.*  encoder.conv_out.*
    #   quant_conv.{weight,bias}     [2*embed, z*2, 1,1,1]   <- nn.Conv3d(...,1)
    #   post_quant_conv.{weight,bias} [z, embed, 1,1,1]
    #   decoder.x_embedder.* / decoder.transformer_blocks.{i}.* / decoder.norm_out.* / decoder.proj_out.*
    #   decoder.register_tokens / decoder.pos_embed.inv_freq
    #
    # MLX PointwiseConv3d 用 2D 权重 [O,I]，需把 1x1x1 的 5D 卷积权重 squeeze。
    # CausalConv3d 用 5D [O,D,H,W,I]，PyTorch 源是 [O,I,D,H,W] —— 重排在加载时
    # 按 shape 匹配跳过，但实际需转置 [O,I,D,H,W]->[O,D,H,W,I]。这里统一处理。
    out = {}
    for k, v in params.items():
        if k.endswith(".weight") and v.ndim == 5:
            # 1x1x1 pointwise conv -> 2D [O,I]
            if list(v.shape[2:]) == [1, 1, 1]:
                # 仅 quant_conv / post_quant_conv / nin_shortcut 是 PointwiseConv3d
                if "quant_conv" in k or "post_quant_conv" in k or "nin_shortcut" in k:
                    out[k] = v.reshape(v.shape[0], v.shape[1])
                    continue
            # CausalConv3d: PyTorch [O,I,D,H,W] -> MLX [O,D,H,W,I]
            out[k] = v.transpose(0, 2, 3, 4, 1)
        else:
            out[k] = v
    return out
