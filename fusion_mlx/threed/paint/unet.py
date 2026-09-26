# SPDX-License-Identifier: Apache-2.0
# Hunyuan3D-2.1 dual-branch multiview paint UNet (issue #989 Session 4).
# diffusers UNet2DConditionModel variant with:
#   - 6 attention kinds per transformer block: attn1 (self, albedo) +
#     attn1_mr (self, mr), attn2 (cross text/clip 1024), attn_dino (cross
#     DINOv2 image 1024), attn_multiview (cross 6 views), attn_refview +
#     attn_refview_mr (cross ref view). Dual albedo/mr output branches.
#   - GEGLU feedforward (net.0.proj gate, net.2 linear).
#   - use_linear_projection=True, transformer_layers_per_block=1.
#   - learned_text_clip_{albedo,mr,ref} (77,1024) fixed embeddings.
#   - image_proj_model_dino: Linear(1536->4096) -> reshape (4,1024) context
#     tokens + LayerNorm(1024).
# Checkpoint is mixed fp16 + 8bit-quant (group 16); quantized linears are
# dequantized to fp16 at load time so the module is uniformly fp16.
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from fusion_mlx.threed.config import PaintUNetConfig

logger = logging.getLogger(__name__)


def _conv(
    in_c: int, out_c: int, k: int = 3, pad: int = 1, bias: bool = True
) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, k, stride=1, padding=pad, bias=bias)


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def geglu(
    x: mx.array, proj_w: mx.array, proj_b: mx.array, lin_w: mx.array, lin_b: mx.array
) -> mx.array:
    # GEGLU: split proj output in half, gate = silu(first) * second, then linear.
    h = x @ proj_w.T + proj_b
    a, b = h[..., : h.shape[-1] // 2], h[..., h.shape[-1] // 2 :]
    return (silu(a) * b) @ lin_w.T + lin_b


def timestep_embedding(
    t: mx.array, dim: int = 320, max_period: int = 10000
) -> mx.array:
    # Diffusers sinusoidal: half=dim//2, freqs=exp(-ln(max)*i/half), sin-first.
    half = dim // 2
    i = np.arange(half, dtype=np.float64)
    freqs = np.exp(-np.log(max_period) * i / half)
    ang = np.asarray(t, dtype=np.float64)[:, None] * freqs[None, :]
    buf = np.empty(ang.shape + (2,), dtype=np.float32)
    buf[..., 0] = np.sin(ang)
    buf[..., 1] = np.cos(ang)
    return mx.array(buf.reshape(-1, dim))


class TimestepBlock(nn.Module):
    # time_embedding.linear_1 (320->1280) + silu + linear_2 (1280->1280).
    def __init__(self, cfg: PaintUNetConfig):
        super().__init__()
        d = cfg.block_out_channels[0]  # 320
        inter = d * 4  # 1280
        self.linear_1 = nn.Linear(d, inter)
        self.linear_2 = nn.Linear(inter, inter)

    def __call__(self, t_emb: mx.array) -> mx.array:
        return self.linear_2(silu(self.linear_1(t_emb)))


class ResnetBlock(nn.Module):
    # norm1 + silu + conv1 + time_emb_proj(silu(temb)) + norm2 + silu + conv2
    # + optional conv_shortcut. GroupNorm(32).
    def __init__(self, in_c: int, out_c: int, temb_dim: int, cfg: PaintUNetConfig):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            cfg.norm_num_groups, in_c, pytorch_compatible=True, eps=cfg.norm_eps
        )
        self.conv1 = _conv(in_c, out_c)
        self.time_emb_proj = nn.Linear(temb_dim, out_c)
        self.norm2 = nn.GroupNorm(
            cfg.norm_num_groups, out_c, pytorch_compatible=True, eps=cfg.norm_eps
        )
        self.conv2 = _conv(out_c, out_c)
        self.in_c = in_c
        self.out_c = out_c
        if in_c != out_c:
            self.conv_shortcut = nn.Conv2d(in_c, out_c, 1, bias=True)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        # x: [B, H, W, C] (NHWC). temb: [B, temb_dim].
        h = self.conv1(silu(self.norm1(x)))
        h = h + self.time_emb_proj(silu(temb))[:, None, None, :]
        h = self.conv2(silu(self.norm2(h)))
        if self.in_c != self.out_c:
            x = self.conv_shortcut(x)
        return x + h


class CrossAttention(nn.Module):
    # to_q (from query_dim), to_k/to_v (from kv_dim), to_out. dual modes:
    #   "full"  (attn1): separate q_mr/k_mr/v_mr/out_mr for the mr branch.
    #   "v_out" (attn_refview): only v_mr + out_mr; q/k SHARED with albedo.
    #   None: single-branch (attn2/dino/multiview).
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        heads: int,
        head_dim: int,
        dual: str | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.dual = dual
        inner = heads * head_dim
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(kv_dim, inner, bias=False)
        self.to_v = nn.Linear(kv_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, query_dim, bias=True)
        if dual == "full":
            self.to_q_mr = nn.Linear(query_dim, inner, bias=False)
            self.to_k_mr = nn.Linear(kv_dim, inner, bias=False)
        if dual is not None:  # both full and v_out have v_mr + out_mr
            self.to_v_mr = nn.Linear(kv_dim, inner, bias=False)
            self.to_out_mr = nn.Linear(inner, query_dim, bias=True)

    def _split(self, x: mx.array) -> mx.array:
        B, N, _ = x.shape
        return x.reshape(B, N, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def _attn(self, q_in, kv_in, q_lin, k_lin, v_lin, out_lin):
        q = self._split(q_lin(q_in))
        k = self._split(k_lin(kv_in))
        v = self._split(v_lin(kv_in))
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        B, H, N, hd = a.shape
        return out_lin(a.transpose(0, 2, 1, 3).reshape(B, N, H * hd))

    def __call__(self, q_in: mx.array, kv_in: mx.array) -> mx.array:
        return self._attn(q_in, kv_in, self.to_q, self.to_k, self.to_v, self.to_out)

    def mr(self, q_in: mx.array, kv_in: mx.array) -> mx.array:
        # mr branch: q/k from _mr if full-dual, else shared with albedo.
        q_lin = self.to_q_mr if self.dual == "full" else self.to_q
        k_lin = self.to_k_mr if self.dual == "full" else self.to_k
        return self._attn(q_in, kv_in, q_lin, k_lin, self.to_v_mr, self.to_out_mr)


class _GEGLU(nn.Module):
    # ff.net.0: GEGLU gate. proj Linear(dim -> 2*inter); split, silu(first)*second.
    def __init__(self, dim: int, inter: int):
        super().__init__()
        self.proj = nn.Linear(dim, inter * 2)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.proj(x)
        a, b = h[..., : h.shape[-1] // 2], h[..., h.shape[-1] // 2 :]
        return silu(a) * b


class _FeedForward(nn.Module):
    # ff.net = [GEGLU(dim, inter*2), identity, Linear(inter, dim)].
    # Keys: ff.net.0.proj.*, ff.net.2.* (net.1 is the activation, no params).
    def __init__(self, dim: int, inter: int):
        super().__init__()
        self.net = [_GEGLU(dim, inter), None, nn.Linear(inter, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        h = self.net[0](x)
        return self.net[2](h)


class TransformerBlock(nn.Module):
    # norm1 + (attn1 self dual: albedo + mr) + norm2 + (attn2 text, attn_dino,
    # attn_multiview, attn_refview dual) + norm3 + ff (GEGLU).
    def __init__(self, dim: int, heads: int, head_dim: int, ctx_dim: int = 1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, dim, heads, head_dim, dual="full")
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, ctx_dim, heads, head_dim)
        self.attn_dino = CrossAttention(dim, ctx_dim, heads, head_dim)
        self.attn_multiview = CrossAttention(dim, dim, heads, head_dim)
        self.attn_refview = CrossAttention(dim, dim, heads, head_dim, dual="v_out")
        self.norm3 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, dim * 4)

    def __call__(
        self,
        h: mx.array,
        ctx_text: mx.array,
        ctx_dino: mx.array,
        ctx_mv: mx.array | None,
        ctx_ref: mx.array | None,
    ) -> mx.array:
        n1 = self.norm1(h)
        h = h + self.attn1(n1, n1) + self.attn1.mr(n1, n1)
        n2 = self.norm2(h)
        h = h + self.attn2(n2, ctx_text) + self.attn_dino(n2, ctx_dino)
        # ctx_mv / ctx_ref: per-block dim matching `dim`. When None (structural
        # scaffold / single-view), zeros -> cross-attn no-op.
        # TODO(reference): real multiview forward runs 6 views jointly with
        # cross-view attention; ctx_mv = the 6 views' tokens at this block dim.
        B, N, D = n2.shape
        if ctx_mv is None:
            ctx_mv = mx.zeros((B, N, D), dtype=n2.dtype)
        if ctx_ref is None:
            ctx_ref = mx.zeros((B, N, D), dtype=n2.dtype)
        h = h + self.attn_multiview(n2, ctx_mv) + self.attn_refview(n2, ctx_ref)
        h = h + self.attn_refview.mr(n2, ctx_ref)
        h = h + self.ff(self.norm3(h))
        return h


class AttentionBlock(nn.Module):
    # norm (GroupNorm 32) + proj_in (Linear) + N transformer_blocks + proj_out.
    # use_linear_projection=True: proj_in/out are Linear over channels, applied
    # per spatial position (BHWC -> reshape).
    def __init__(
        self,
        channels: int,
        heads: int,
        head_dim: int,
        ctx_dim: int,
        num_tb: int = 1,
        num_groups: int = 32,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, pytorch_compatible=True, eps=eps)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = [
            TransformerBlock(channels, heads, head_dim, ctx_dim) for _ in range(num_tb)
        ]
        self.proj_out = nn.Linear(channels, channels)

    def __call__(
        self, x: mx.array, ctx_text, ctx_dino, ctx_mv=None, ctx_ref=None
    ) -> mx.array:
        # x: [B, H, W, C] (NHWC). Reshape to [B, H*W, C] for attention.
        B, H, W, C = x.shape
        r = self.norm(x)
        r = r.reshape(B, H * W, C)
        r = self.proj_in(r)
        for tb in self.transformer_blocks:
            r = tb(r, ctx_text, ctx_dino, ctx_mv, ctx_ref)
        r = self.proj_out(r)
        r = r.reshape(B, H, W, C)
        return x + r


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C] (NHWC). Nearest-neighbor 2x upsample.
        B, H, W, C = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C)).reshape(
            B, H * 2, W * 2, C
        )
        return self.conv(x)


class CrossAttnDownBlock(nn.Module):
    # 2 resnets + 2 attentions + 1 downsample. Appends 3 skips (2 res + 1 ds).
    def __init__(
        self,
        in_c: int,
        out_c: int,
        temb_dim: int,
        heads: int,
        head_dim: int,
        ctx_dim: int,
        cfg: PaintUNetConfig,
    ):
        super().__init__()
        self.resnets = [
            ResnetBlock(in_c if i == 0 else out_c, out_c, temb_dim, cfg)
            for i in range(cfg.layers_per_block)
        ]
        self.attentions = [
            AttentionBlock(out_c, heads, head_dim, ctx_dim)
            for _ in range(cfg.layers_per_block)
        ]
        self.downsamplers = [Downsample(out_c)]

    def __call__(self, x, temb, ctx_text, ctx_dino, ctx_mv, ctx_ref, skips):
        for r, a in zip(self.resnets, self.attentions):
            x = r(x, temb)
            x = a(x, ctx_text, ctx_dino, ctx_mv, ctx_ref)
            skips.append(x)
        x = self.downsamplers[0](x)
        skips.append(x)
        return x


class DownBlock(nn.Module):
    # 2 resnets, no attn, no downsample. Appends 2 skips.
    def __init__(self, in_c: int, out_c: int, temb_dim: int, cfg: PaintUNetConfig):
        super().__init__()
        self.resnets = [
            ResnetBlock(in_c if i == 0 else out_c, out_c, temb_dim, cfg)
            for i in range(cfg.layers_per_block)
        ]

    def __call__(self, x, temb, ctx_text, ctx_dino, ctx_mv, ctx_ref, skips):
        for r in self.resnets:
            x = r(x, temb)
            skips.append(x)
        return x


class MidBlock(nn.Module):
    # resnet, attn, resnet.
    def __init__(
        self,
        channels: int,
        temb_dim: int,
        heads: int,
        head_dim: int,
        ctx_dim: int,
        cfg: PaintUNetConfig,
    ):
        super().__init__()
        self.resnets = [
            ResnetBlock(channels, channels, temb_dim, cfg) for _ in range(2)
        ]
        self.attentions = [AttentionBlock(channels, heads, head_dim, ctx_dim)]

    def __call__(self, x, temb, ctx_text, ctx_dino, ctx_mv, ctx_ref):
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, ctx_text, ctx_dino, ctx_mv, ctx_ref)
        x = self.resnets[1](x, temb)
        return x


class CrossAttnUpBlock(nn.Module):
    # 3 resnets + 3 attentions + optional upsample. Pops 3 skips.
    # skip_channels: the 3 skip channel dims in pop order (last skip first).
    # resnet i: in = (prev_c if i==0 else out_c) + skip_channels[i], out = out_c.
    def __init__(
        self,
        in_c: int,
        out_c: int,
        prev_c: int,
        temb_dim: int,
        heads: int,
        head_dim: int,
        ctx_dim: int,
        cfg: PaintUNetConfig,
        skip_channels: list,
        use_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = []
        for i in range(cfg.layers_per_block + 1):
            rc = prev_c if i == 0 else out_c
            self.resnets.append(
                ResnetBlock(rc + skip_channels[i], out_c, temb_dim, cfg)
            )
        self.attentions = [
            AttentionBlock(out_c, heads, head_dim, ctx_dim)
            for _ in range(cfg.layers_per_block + 1)
        ]
        self.upsamplers = [Upsample(out_c)] if use_upsample else None

    def __call__(self, x, temb, ctx_text, ctx_dino, ctx_mv, ctx_ref, skips):
        for r, a in zip(self.resnets, self.attentions):
            x = mx.concatenate([x, skips.pop()], axis=-1)
            x = r(x, temb)
            x = a(x, ctx_text, ctx_dino, ctx_mv, ctx_ref)
        if self.upsamplers:
            x = self.upsamplers[0](x)
        return x


class UpBlock(nn.Module):
    # 3 resnets + upsample, no attn. Pops 3 skips.
    def __init__(
        self,
        in_c: int,
        out_c: int,
        prev_c: int,
        temb_dim: int,
        cfg: PaintUNetConfig,
        skip_channels: list,
        use_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = []
        for i in range(cfg.layers_per_block + 1):
            rc = prev_c if i == 0 else out_c
            self.resnets.append(
                ResnetBlock(rc + skip_channels[i], out_c, temb_dim, cfg)
            )
        self.upsamplers = [Upsample(out_c)] if use_upsample else None

    def __call__(self, x, temb, ctx_text, ctx_dino, ctx_mv, ctx_ref, skips):
        for r in self.resnets:
            x = mx.concatenate([x, skips.pop()], axis=-1)
            x = r(x, temb)
        if self.upsamplers:
            x = self.upsamplers[0](x)
        return x


class ImageProjDino(nn.Module):
    # proj Linear(1536 -> 4096) reshaped (4, 1024) + LayerNorm(1024).
    # Produces 4 DINOv2 context tokens per image.
    def __init__(self, clip_dim: int = 1536, ctx_dim: int = 1024, n_tokens: int = 4):
        super().__init__()
        self.n_tokens = n_tokens
        self.ctx_dim = ctx_dim
        self.proj = nn.Linear(clip_dim, n_tokens * ctx_dim)
        self.norm = nn.LayerNorm(ctx_dim)

    def __call__(self, dino_feat: mx.array) -> mx.array:
        # dino_feat: [B, 1536] (pooled) -> [B, 4, 1024].
        B = dino_feat.shape[0]
        h = self.proj(dino_feat).reshape(B, self.n_tokens, self.ctx_dim)
        return self.norm(h)


class HunyuanPaintUNet(nn.Module):
    def __init__(self, cfg: PaintUNetConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.block_out_channels  # [320,640,1280,1280]
        heads = cfg.attention_head_dim  # [5,10,20,20]
        ctx = cfg.cross_attention_dim  # 1024
        temb_dim = ch[0] * 4  # 1280
        self.conv_in = nn.Conv2d(
            cfg.in_channels, ch[0], 3, stride=1, padding=1, bias=True
        )
        self.time_embedding = TimestepBlock(cfg)

        # head_dim: diffusers attention_head_dim is num_heads per block when
        # given as int list; head_dim = channels // heads.
        def hd(i):
            return ch[i] // heads[i]

        self.down_blocks = [
            CrossAttnDownBlock(ch[0], ch[0], temb_dim, heads[0], hd(0), ctx, cfg),
            CrossAttnDownBlock(ch[0], ch[1], temb_dim, heads[1], hd(1), ctx, cfg),
            CrossAttnDownBlock(ch[1], ch[2], temb_dim, heads[2], hd(2), ctx, cfg),
            DownBlock(ch[2], ch[3], temb_dim, cfg),
        ]
        self.mid_block = MidBlock(ch[3], temb_dim, heads[3], hd(3), ctx, cfg)
        # Up blocks: out = reversed(block_out_channels) = [1280,1280,640,320];
        # heads = reversed(attention_head_dim) = [20,20,10,5]; prev_c is the
        # previous up block's output (mid=1280 for up0).
        rch = list(reversed(ch))  # [1280, 1280, 640, 320]
        rhd = list(reversed(heads))  # [20, 20, 10, 5]

        def rhd_dim(i):
            return rch[i] // rhd[i]  # always 64

        # Skip channels per up resnet (pop order, last-first):
        # up0 pops [d3r1, d3r0, d2ds] = [ch[3], ch[3], ch[3]];
        # up1 pops [d2r1, d2r0, d1ds] = [ch[2], ch[2], ch[1]];
        # up2 pops [d1r1, d1r0, d0ds] = [ch[1], ch[1], ch[0]];
        # up3 pops [d0r1, d0r0, conv_in] = [ch[0], ch[0], ch[0]].
        self.up_blocks = [
            UpBlock(rch[0], rch[0], ch[3], temb_dim, cfg, [ch[3], ch[3], ch[3]], True),
            CrossAttnUpBlock(
                rch[0],
                rch[1],
                rch[0],
                temb_dim,
                rhd[1],
                rhd_dim(1),
                ctx,
                cfg,
                [ch[2], ch[2], ch[1]],
                True,
            ),
            CrossAttnUpBlock(
                rch[1],
                rch[2],
                rch[1],
                temb_dim,
                rhd[2],
                rhd_dim(2),
                ctx,
                cfg,
                [ch[1], ch[1], ch[0]],
                True,
            ),
            CrossAttnUpBlock(
                rch[2],
                rch[3],
                rch[2],
                temb_dim,
                rhd[3],
                rhd_dim(3),
                ctx,
                cfg,
                [ch[0], ch[0], ch[0]],
                False,
            ),
        ]
        self.conv_norm_out = nn.GroupNorm(
            cfg.norm_num_groups, ch[0], pytorch_compatible=True, eps=cfg.norm_eps
        )
        self.conv_out = nn.Conv2d(
            ch[0], cfg.out_channels, 3, stride=1, padding=1, bias=True
        )
        self.image_proj_model_dino = ImageProjDino(1536, ctx, 4)
        # learned text embeddings (fixed, loaded from checkpoint).
        self.learned_text_clip_albedo = mx.zeros((77, ctx), dtype=mx.float16)
        self.learned_text_clip_mr = mx.zeros((77, ctx), dtype=mx.float16)
        self.learned_text_clip_ref = mx.zeros((77, ctx), dtype=mx.float16)

    def __call__(self, latent, timestep, ctx_text, ctx_dino, ctx_mv=None, ctx_ref=None):
        # latent [B, in_channels, H, W] (NCHW, diffusers convention) -> NHWC.
        # timestep scalar; ctx_text [B,77,1024]; ctx_dino [B,4,1024];
        # ctx_mv [B,N,320]; ctx_ref [B,N,320]. Returns [B, out_channels, H, W] (NCHW).
        B = latent.shape[0]
        t_emb = timestep_embedding(
            mx.array([float(timestep)]), self.cfg.block_out_channels[0]
        )
        temb = self.time_embedding(t_emb.astype(mx.float16))  # [1, 1280]
        temb = mx.broadcast_to(temb, (B, temb.shape[1]))
        x = latent.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.conv_in(x)
        skips = [x]
        for db in self.down_blocks:
            x = db(x, temb, ctx_text, ctx_dino, ctx_mv, ctx_ref, skips)
        x = self.mid_block(x, temb, ctx_text, ctx_dino, ctx_mv, ctx_ref)
        for ub in self.up_blocks:
            x = ub(x, temb, ctx_text, ctx_dino, ctx_mv, ctx_ref, skips)
        x = self.conv_out(silu(self.conv_norm_out(x)))
        return x.transpose(0, 3, 1, 2)  # NHWC -> NCHW


def _dequant_weight(raw: dict, base: str) -> mx.array | None:
    # If base.scales exists, dequantize uint32 packed weight -> fp16 Linear weight.
    wk, sk, bk, biask = (
        f"{base}.weight",
        f"{base}.scales",
        f"{base}.biases",
        f"{base}.bias",
    )
    if sk not in raw:
        return None
    w = raw[wk]  # [out, in//4] uint32
    s = raw[sk]  # [out, groups]
    b = raw[bk]
    out, packed = w.shape
    in_dim = packed * 4
    group_size = in_dim // s.shape[1]
    # mlx.nn.quantized.dequantize(w, scales, biases, group_size, bits=8)
    dq = mx.dequantize(w, s, b, group_size=group_size, bits=8)
    return dq


def load_paint_unet(weights_path: str, cfg: PaintUNetConfig) -> HunyuanPaintUNet:
    from safetensors import safe_open

    model = HunyuanPaintUNet(cfg)
    raw: dict[str, mx.array] = {}
    with safe_open(weights_path, framework="np") as f:
        for k in f.keys():  # noqa: SIM118
            raw[k] = mx.array(f.get_tensor(k))

    # Flatten module params, build remapped dict: quantized linears dequantized
    # to fp16 so every linear is a plain nn.Linear weight/bias.
    flat: dict = {}
    nn.utils.tree_flatten(model, destination=flat)
    module_keys = set(flat.keys())

    remapped: dict[str, mx.array] = {}
    consumed: set[str] = set()
    for k, v in raw.items():
        # quantized linear: {base}.weight(uint32) + .scales + .biases [+ .bias]
        if k.endswith(".scales") or k.endswith(".biases"):
            consumed.add(k)
            continue
        if k.endswith(".weight") and k.replace(".weight", ".scales") in raw:
            base = k[: -len(".weight")]
            dq = _dequant_weight(raw, base)
            remapped[f"{base}.weight"] = dq.astype(mx.float16)
            consumed.add(k)
            if f"{base}.bias" in raw:
                remapped[f"{base}.bias"] = raw[f"{base}.bias"]
                consumed.add(f"{base}.bias")
            continue
        remapped[k] = v

    model.load_weights(list(remapped.items()), strict=False)
    # Conv2d stride/padding tuples flatten to non-weight keys (conv.stride.1
    # etc); they carry no tensor and are not in the checkpoint — filter noise.
    _is_weight = lambda k: (
        not any(
            k.endswith(s)
            for s in (
                ".stride.0",
                ".stride.1",
                ".padding.0",
                ".padding.1",
                ".net.1",
            )
        )
    )
    missing = [k for k in module_keys if k not in remapped and _is_weight(k)]
    skipped = [k for k in remapped if k not in module_keys]
    if missing:
        logger.warning("PaintUNet missing (%d): %s", len(missing), missing[:10])
    if skipped:
        logger.warning("PaintUNet unexpected (%d): %s", len(skipped), skipped[:10])
    logger.info(
        "PaintUNet loaded: %d raw, %d mapped, %d dequantized",
        len(raw),
        len(remapped),
        len(consumed),
    )
    return model
