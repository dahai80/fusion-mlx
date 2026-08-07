import logging
import math

import mlx.core as mx
from mlx.core.fast import scaled_dot_product_attention

from mlx import nn

logger = logging.getLogger(__name__)


def _nchw_to_nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def _nhwc_to_nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


class WuerstchenLayerNorm(nn.Module):
    # SDCascadeLayerNorm: LayerNorm over channel axis (last axis in NHWC),
    # elementwise_affine=False, eps=1e-6. No learnable weight/bias (matches
    # diffusers elementwise_affine=False -> no params). Operates on NHWC.

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        return x


class GlobalResponseNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.zeros((1, 1, 1, dim))
        self.beta = mx.zeros((1, 1, 1, dim))

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        agg = mx.sqrt(mx.sum(x * x, axis=(1, 2), keepdims=True) + 1e-6)
        stand = agg / (mx.mean(agg, axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * stand) + self.beta + x


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = k // 2
        scale = 1.0 / (in_ch * k * k) ** 0.5
        self.weight = mx.random.normal((out_ch, k, k, 1)) * scale
        self.bias = mx.zeros((out_ch,))
        self.padding = padding
        self.groups = in_ch

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.groups:
            raise ValueError(
                f"depthwise groups mismatch: in_ch={self.groups} x_ch={x.shape[-1]}"
            )
        w = mx.broadcast_to(
            self.weight,
            (
                self.weight.shape[0],
                self.weight.shape[1],
                self.weight.shape[2],
                self.groups,
            ),
        )
        if self.padding > 0:
            x = mx.pad(
                x,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
            )
        y = mx.conv2d(x, w, stride=1)
        y = y + self.bias
        return y


class _DropoutSlot(nn.Module):
    def __init__(self):
        super().__init__()
        self._drop = nn.Dropout(0.0)

    def __call__(self, x: mx.array, dropout: float = 0.0) -> mx.array:
        if dropout > 0:
            self._drop.p = dropout
            return self._drop(x)
        return x


def _channelwise(c_in: int, c_out: int, dropout: float = 0.0) -> list:
    # [Linear, GELU, GlobalResponseNorm, Linear] matching diffusers key
    # layout channelwise.0 / channelwise.2 / channelwise.4. A Dropout
    # placeholder sits at index 3 (no params) to match the diffusers
    # Sequential(Linear, GELU, GRN, Dropout, Linear) key indexing.
    return [
        nn.Linear(c_in, c_out * 4),
        nn.GELU(),
        GlobalResponseNorm(c_out * 4),
        _DropoutSlot(),
        nn.Linear(c_out * 4, c_out),
    ]


def _run_channelwise(layers: list, x: mx.array, dropout: float = 0.0) -> mx.array:
    x = layers[0](x)
    x = layers[1](x)
    x = layers[2](x)
    x = layers[3](x, dropout)
    x = layers[4](x)
    return x


class TimestepBlock(nn.Module):
    # mapper + mapper_{cond} for each cond name. t chunked into
    # len(conds)+1 along axis=1; x*(1+a)+b with a,b shape (b,c,1,1).

    def __init__(self, c: int, c_timestep: int, conds: tuple = ()):
        super().__init__()
        self.conds = list(conds)
        self.mapper = nn.Linear(c_timestep, c * 2)
        for cname in self.conds:
            setattr(self, f"mapper_{cname}", nn.Linear(c_timestep, c * 2))

    def __call__(self, x: mx.array, t: mx.array) -> mx.array:
        n = len(self.conds) + 1
        chunks = mx.split(t, n, axis=1)
        a, b = mx.split(self.mapper(chunks[0]), 2, axis=1)
        a = a[:, None, None, :]
        b = b[:, None, None, :]
        for i, cname in enumerate(self.conds):
            ac, bc = mx.split(
                getattr(self, f"mapper_{cname}")(chunks[i + 1]), 2, axis=1
            )
            a = a + ac[:, None, None, :]
            b = b + bc[:, None, None, :]
        return x * (1 + a) + b


class ResBlock(nn.Module):
    # depthwise (groups=c) -> norm (affine=False) -> concat x_skip ->
    # channelwise (c+c_skip -> c*4 -> c with GRN) -> + residual.
    # Keys: depthwise, channelwise.0/2/4.

    def __init__(
        self, c: int, c_skip: int = 0, kernel_size: int = 3, dropout: float = 0.0
    ):
        super().__init__()
        self.c_skip = c_skip
        self.depthwise = DepthwiseConv2d(c, c, kernel_size)
        self.norm = WuerstchenLayerNorm(c)
        self.channelwise = _channelwise(c + c_skip, c, dropout)
        self.dropout = dropout

    def __call__(self, x: mx.array, x_skip: mx.array | None = None) -> mx.array:
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = mx.concatenate([x, x_skip], axis=-1)
        h = _run_channelwise(self.channelwise, x, self.dropout)
        return h + x_res


class _Attention(nn.Module):
    # diffusers Attention module subset: to_q/to_k/to_v/to_out.0 (Linear).
    # Used inside AttnBlock so checkpoint keys are attention.to_q etc.

    def __init__(self, dims: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dims // heads
        self.to_q = nn.Linear(dims, dims)
        self.to_k = nn.Linear(dims, dims)
        self.to_v = nn.Linear(dims, dims)
        self.to_out = [nn.Linear(dims, dims)]
        self.dropout = dropout
        self._drop = nn.Dropout(0.0)

    def __call__(self, x: mx.array, kv: mx.array) -> mx.array:
        b, s, _ = x.shape
        q = self.to_q(x).reshape(b, s, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        sk = kv.shape[1]
        k = (
            self.to_k(kv)
            .reshape(b, sk, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.to_v(kv)
            .reshape(b, sk, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=mx.float32))
        out = scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.heads * self.head_dim)
        out = self.to_out[0](out)
        if self.dropout > 0:
            self._drop.p = self.dropout
            out = self._drop(out)
        return out


class AttnBlock(nn.Module):
    # norm (affine=False) + attention(self-attn with kv concat) + kv_mapper.
    # Keys: attention.to_q/k/v/out.0, kv_mapper.1.

    def __init__(
        self,
        c: int,
        c_cond: int,
        nhead: int,
        self_attn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.norm = WuerstchenLayerNorm(c)
        self.attention = _Attention(c, nhead, dropout)
        self.kv_mapper = [nn.SiLU(), nn.Linear(c_cond, c)]

    def __call__(self, x: mx.array, kv: mx.array) -> mx.array:
        kv = self.kv_mapper[0](kv)
        kv = self.kv_mapper[1](kv)
        norm_x = self.norm(x)
        b, hh, w, ch = x.shape
        seq = hh * w
        norm_x_seq = norm_x.reshape(b, seq, ch)
        if self.self_attn:
            self_kv = norm_x_seq
            kv = mx.concatenate([self_kv, kv], axis=1)
        out = self.attention(norm_x_seq, kv)
        out = out.reshape(b, hh, w, ch)
        return x + out


def gen_r_embedding(r: mx.array, c_r: int, max_positions: int = 10000) -> mx.array:
    r = r * max_positions
    half_dim = c_r // 2
    exponent = -math.log(max_positions) / (half_dim - 1)
    emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * exponent)
    emb = r.reshape(-1, 1).astype(mx.float32) * emb.reshape(1, -1)
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=1)
    if c_r % 2 == 1:
        emb = mx.pad(emb, ((0, 0), (0, 1)))
    return emb
