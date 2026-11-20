# SPDX-License-Identifier: MIT
# BiRefNet-lite MLX port (vendored from moonbox-dev/BiRefNet_lite-MLX, MIT).
# Upstream: https://huggingface.co/moonbox-dev/BiRefNet_lite-MLX (fp16/mlx_model.py).
# Unmodified except this header. Upstream-derived vendor code: lint-excluded
# via pyproject (see [tool.ruff] extend-exclude).
"""BiRefNet (ZhengPeng7/BiRefNet, BiRefNet_lite; MIT) in MLX: a port of the Hub repo's `birefnet.py` for the
released configuration (Swin v1 backbone run at full and half resolution and concatenated, 3 context features,
ASPPDeformable decoder blocks, image-patch inputs, gradient attention).

    import mlx_model
    model = mlx_model.load("bf16")                       # a folder with config.json + model.safetensors
    alpha = mlx_model.matte(model, rgb)                  # rgb: HxWx3 uint8 -> HxW uint8 alpha (255 = foreground)

Runs on its own next to its weights (mlx and numpy). NHWC throughout. Each BatchNorm is folded into the conv before
it at conversion; torchvision's deform_conv2d (DCNv2, with modulation) is written out as bilinear gathers.
Quantized builds (8bit / 6bit / 4bit) quantize the Swin backbone's linears whose input size is a multiple of 64;
the convolutions stay bf16.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SWIN = {"swin_v1_t": dict(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7),
        "swin_v1_l": dict(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12)}
CONFIG = {"model_type": "birefnet", "backbone": "swin_v1_t", "input_size": 1024}
PRECISIONS = ["bf16", "fp16", "8bit", "6bit", "4bit"]
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
GROUP = 64


def quantizable(path: str, module) -> bool:
    return isinstance(module, nn.Linear) and module.weight.shape[-1] % GROUP == 0


# ------------------------------------------------------------------------------------------------ ops

def resize(x, size, align_corners: bool):
    """torch F.interpolate(mode="bilinear") for NHWC (no antialiasing)."""
    b, h, w, c = x.shape
    oh, ow = int(size[0]), int(size[1])
    if (oh, ow) == (h, w):
        return x

    def axis(n_in, n_out):
        if align_corners:
            s = np.arange(n_out, dtype=np.float64) * ((n_in - 1) / (n_out - 1) if n_out > 1 else 0.0)
        else:
            s = np.clip((np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5, 0, None)
        i0 = np.minimum(np.floor(s).astype(np.int32), n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return mx.array(i0), mx.array(i1), mx.array((s - i0).astype(np.float32))

    y0, y1, fy = axis(h, oh)
    x0, x1, fx = axis(w, ow)
    fy = fy.astype(x.dtype)[None, :, None, None]
    fx = fx.astype(x.dtype)[None, None, :, None]
    top, bot = x[:, y0], x[:, y1]
    top = top[:, :, x0] * (1 - fx) + top[:, :, x1] * fx
    bot = bot[:, :, x0] * (1 - fx) + bot[:, :, x1] * fx
    return top * (1 - fy) + bot * fy


def to_patches(x, ref_hw):
    """einops 'b c (hg h) (wg w) -> b (c hg wg) h w' in NHWC."""
    b, H, W, c = x.shape
    h, w = ref_hw
    hg, wg = H // h, W // w
    x = x.reshape(b, hg, h, wg, w, c).transpose(0, 2, 4, 5, 1, 3)
    return x.reshape(b, h, w, c * hg * wg)


def deform_conv(x, offset, mask, weight, bias, k: int):
    """torchvision.ops.deform_conv2d (stride 1, padding k // 2, dilation 1, one offset group) with a modulation mask.
    x (1, H, W, C), offset (1, H, W, 2 k k) as (dy, dx) per kernel position, mask (1, H, W, k k), weight (O, k, k, C)."""
    b, h, w, c = x.shape
    assert b == 1, "batch 1"
    pad = k // 2
    flat = x.reshape(h * w, c)
    gy = mx.arange(h, dtype=mx.float32)[:, None]
    gx = mx.arange(w, dtype=mx.float32)[None, :]
    off = offset[0].astype(mx.float32)
    out = None
    for i in range(k * k):
        ky, kx = divmod(i, k)
        py = gy + (ky - pad) + off[:, :, 2 * i]
        px = gx + (kx - pad) + off[:, :, 2 * i + 1]
        y0, x0 = mx.floor(py), mx.floor(px)
        ly, lx = py - y0, px - x0
        acc = None
        for yy, xx, wt in ((y0, x0, (1 - ly) * (1 - lx)), (y0, x0 + 1, (1 - ly) * lx),
                           (y0 + 1, x0, ly * (1 - lx)), (y0 + 1, x0 + 1, ly * lx)):
            ok = (yy >= 0) & (yy <= h - 1) & (xx >= 0) & (xx <= w - 1)
            idx = (mx.clip(yy, 0, h - 1) * w + mx.clip(xx, 0, w - 1)).astype(mx.int32).reshape(-1)
            v = mx.take(flat, idx, axis=0) * mx.where(ok, wt, 0.0).reshape(-1, 1).astype(x.dtype)
            acc = v if acc is None else acc + v
        acc = acc * mask[0, :, :, i].reshape(-1, 1)
        y = acc @ weight[:, ky, kx, :].T
        out = y if out is None else out + y
    return (out + bias).reshape(1, h, w, -1)


# ------------------------------------------------------------------------------------------------ Swin v1

def window_partition(x, ws):
    b, h, w, c = x.shape
    x = x.reshape(b, h // ws, ws, w // ws, ws, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, c)


def window_reverse(win, ws, h, w):
    c = win.shape[-1]
    b = win.shape[0] // ((h // ws) * (w // ws))
    x = win.reshape(b, h // ws, w // ws, ws, ws, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, w, c)


def relative_index(ws: int) -> np.ndarray:
    coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws), indexing="ij")).reshape(2, -1)
    rel = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0) + (ws - 1)
    return rel[:, :, 0] * (2 * ws - 1) + rel[:, :, 1]


def shift_mask(hp: int, wp: int, ws: int, shift: int) -> np.ndarray:
    img = np.zeros((1, hp, wp, 1), np.float32)
    cnt = 0
    for hs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
        for wsl in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            img[:, hs, wsl, :] = cnt
            cnt += 1
    m = img.reshape(1, hp // ws, ws, wp // ws, ws, 1).transpose(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws)
    m = m[:, None, :] - m[:, :, None]
    return np.where(m != 0, -100.0, 0.0).astype(np.float32)


class WindowAttention(nn.Module):
    def __init__(self, dim, ws, heads):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.relative_position_bias_table = mx.zeros(((2 * ws - 1) ** 2, heads))
        self._index = mx.array(relative_index(ws).reshape(-1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        bias = self.relative_position_bias_table[self._index].reshape(n, n, -1).transpose(2, 0, 1)
        attn = (q @ k.transpose(0, 1, 3, 2)) + bias[None].astype(q.dtype)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.reshape(b // nw, nw, self.heads, n, n) + mask[None, :, None].astype(q.dtype)
            attn = attn.reshape(b, self.heads, n, n)
        attn = mx.softmax(attn, axis=-1, precise=True)
        return self.proj((attn @ v).transpose(0, 2, 1, 3).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads)
        self.mlp = Mlp(dim)

    def __call__(self, x, h, w, mask):
        b, L, c = x.shape
        ws, s = self.ws, self.shift
        y = self.norm1(x).reshape(b, h, w, c)
        pr, pb = (ws - w % ws) % ws, (ws - h % ws) % ws
        if pr or pb:
            y = mx.pad(y, [(0, 0), (0, pb), (0, pr), (0, 0)])
        hp, wp = h + pb, w + pr
        if s:
            y = mx.roll(y, (-s, -s), axis=(1, 2))
        y = window_reverse(self.attn(window_partition(y, ws), mask if s else None), ws, hp, wp)
        if s:
            y = mx.roll(y, (s, s), axis=(1, 2))
        y = y[:, :h, :w].reshape(b, h * w, c)
        x = x + y
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def __call__(self, x, h, w):
        b, L, c = x.shape
        x = x.reshape(b, h, w, c)
        if h % 2 or w % 2:
            x = mx.pad(x, [(0, 0), (0, h % 2), (0, w % 2), (0, 0)])
        x = mx.concatenate([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x.reshape(b, -1, 4 * c)))


class SwinLayer(nn.Module):
    def __init__(self, dim, depth, heads, ws, down):
        super().__init__()
        self.ws = ws
        self.blocks = [SwinBlock(dim, heads, ws, 0 if i % 2 == 0 else ws // 2) for i in range(depth)]
        if down:
            self.downsample = PatchMerging(dim)
        self._masks = {}

    def __call__(self, x, h, w):
        ws = self.ws
        hp, wp = -(-h // ws) * ws, -(-w // ws) * ws
        if (hp, wp) not in self._masks:
            self._masks[(hp, wp)] = mx.array(shift_mask(hp, wp, ws, ws // 2))
        mask = self._masks[(hp, wp)]
        for blk in self.blocks:
            x = blk(x, h, w, mask)
        if "downsample" in self:
            return x, self.downsample(x, h, w), (h + 1) // 2, (w + 1) // 2
        return x, x, h, w


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, 4, stride=4)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x):
        b, h, w, _ = x.shape
        if w % 4 or h % 4:
            x = mx.pad(x, [(0, 0), (0, (4 - h % 4) % 4), (0, (4 - w % 4) % 4), (0, 0)])
        return self.norm(self.proj(x))


class Swin(nn.Module):
    def __init__(self, embed_dim, depths, num_heads, window_size):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim)
        self.dims = [embed_dim * 2 ** i for i in range(4)]
        self.layers = [SwinLayer(self.dims[i], depths[i], num_heads[i], window_size, i < 3) for i in range(4)]
        for i, d in enumerate(self.dims):
            setattr(self, f"norm{i}", nn.LayerNorm(d))

    def __call__(self, x):
        x = self.patch_embed(x)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        outs = []
        for i, layer in enumerate(self.layers):
            xo, x, nh, nw = layer(x, h, w)
            outs.append(getattr(self, f"norm{i}")(xo).reshape(b, h, w, self.dims[i]))
            h, w = nh, nw
        return outs


# ------------------------------------------------------------------------------------------------ decoder

class DeformConv(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.k = k
        self.offset_conv = nn.Conv2d(cin, 2 * k * k, k, padding=k // 2)
        self.modulator_conv = nn.Conv2d(cin, k * k, k, padding=k // 2)
        self.regular_conv = nn.Conv2d(cin, cout, k, padding=k // 2)      # its weight (BatchNorm folded in)

    def __call__(self, x):
        off = self.offset_conv(x)
        mask = 2 * mx.sigmoid(self.modulator_conv(x))
        return deform_conv(x, off, mask, self.regular_conv.weight, self.regular_conv.bias, self.k)


class ASPPDeformable(nn.Module):
    def __init__(self, cin, mid=256, sizes=(1, 3, 7)):
        super().__init__()
        self.aspp1 = DeformConv(cin, mid, 1)
        self.aspp_deforms = [DeformConv(cin, mid, k) for k in sizes]
        self.gap_conv = nn.Conv2d(cin, mid, 1)
        self.conv1 = nn.Conv2d(mid * (2 + len(sizes)), cin, 1)

    def __call__(self, x):
        b, h, w, _ = x.shape
        parts = [nn.relu(self.aspp1(x))] + [nn.relu(d(x)) for d in self.aspp_deforms]
        g = x.astype(mx.float32).mean(axis=(1, 2), keepdims=True).astype(x.dtype)   # fp16 sums overflow
        g = nn.relu(self.gap_conv(g))
        parts.append(mx.broadcast_to(g, (b, h, w, g.shape[-1])))
        return nn.relu(self.conv1(mx.concatenate(parts, -1)))


class DecBlk(nn.Module):
    def __init__(self, cin, cout, inter=64):
        super().__init__()
        self.conv_in = nn.Conv2d(cin, inter, 3, padding=1)
        self.dec_att = ASPPDeformable(inter)
        self.conv_out = nn.Conv2d(inter, cout, 3, padding=1)

    def __call__(self, x):
        return self.conv_out(self.dec_att(nn.relu(self.conv_in(x))))


class SimpleConvs(nn.Module):
    def __init__(self, cin, cout, inter=64):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, inter, 3, padding=1)
        self.conv_out = nn.Conv2d(inter, cout, 3, padding=1)

    def __call__(self, x):
        return self.conv_out(self.conv1(x))


class Decoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ipt_blk5 = SimpleConvs(2 ** 10 * 3, ch[0] // 8)
        self.ipt_blk4 = SimpleConvs(2 ** 8 * 3, ch[0] // 8)
        self.ipt_blk3 = SimpleConvs(2 ** 6 * 3, ch[1] // 8)
        self.ipt_blk2 = SimpleConvs(2 ** 4 * 3, ch[2] // 8)
        self.ipt_blk1 = SimpleConvs(3, ch[3] // 8)
        self.decoder_block4 = DecBlk(ch[0] + ch[0] // 8, ch[1])
        self.decoder_block3 = DecBlk(ch[1] + ch[0] // 8, ch[2])
        self.decoder_block2 = DecBlk(ch[2] + ch[1] // 8, ch[3])
        self.decoder_block1 = DecBlk(ch[3] + ch[2] // 8, ch[3] // 2)
        self.conv_out1 = nn.Conv2d(ch[3] // 2 + ch[3] // 8, 1, 1)
        self.lateral_block4 = nn.Conv2d(ch[1], ch[1], 1)
        self.lateral_block3 = nn.Conv2d(ch[2], ch[2], 1)
        self.lateral_block2 = nn.Conv2d(ch[3], ch[3], 1)
        for i, c in ((4, ch[1]), (3, ch[2]), (2, ch[3])):
            setattr(self, f"gdt_convs_{i}", nn.Conv2d(c, 16, 3, padding=1))
            setattr(self, f"gdt_attn_{i}", nn.Conv2d(16, 1, 1))

    def gdt(self, i, p):
        return p * mx.sigmoid(getattr(self, f"gdt_attn_{i}")(nn.relu(getattr(self, f"gdt_convs_{i}")(p))))

    def __call__(self, x, x1, x2, x3, x4):
        up = lambda a, ref: resize(a, ref.shape[1:3], True)
        x4 = mx.concatenate([x4, self.ipt_blk5(to_patches(x, x4.shape[1:3]))], -1)
        p4 = self.gdt(4, self.decoder_block4(x4))
        p3 = up(p4, x3) + self.lateral_block4(x3)
        p3 = mx.concatenate([p3, self.ipt_blk4(to_patches(x, p3.shape[1:3]))], -1)
        p3 = self.gdt(3, self.decoder_block3(p3))
        p2 = up(p3, x2) + self.lateral_block3(x2)
        p2 = mx.concatenate([p2, self.ipt_blk3(to_patches(x, p2.shape[1:3]))], -1)
        p2 = self.gdt(2, self.decoder_block2(p2))
        p1 = up(p2, x1) + self.lateral_block2(x1)
        p1 = mx.concatenate([p1, self.ipt_blk2(to_patches(x, p1.shape[1:3]))], -1)
        p1 = up(self.decoder_block1(p1), x)
        p1 = mx.concatenate([p1, self.ipt_blk1(x)], -1)
        return self.conv_out1(p1)


class Model(nn.Module):
    def __init__(self, config: dict = CONFIG):
        super().__init__()
        sw = SWIN[config["backbone"]]
        self.size = config["input_size"]
        self.bb = Swin(**sw)
        ch = [d * 2 for d in self.bb.dims[::-1]]                       # mul_scl_ipt = "cat"
        cxt = ch[1:][::-1]
        self.squeeze_module = DecBlk(ch[0] + sum(cxt), ch[0])
        self.decoder = Decoder(ch)

    def __call__(self, x):
        """x: (1, S, S, 3) ImageNet-normalized -> (1, S, S, 1) logits."""
        b, h, w, _ = x.shape
        f = self.bb(x)
        g = self.bb(resize(x, (h // 2, w // 2), True))
        f = [mx.concatenate([a, resize(c, a.shape[1:3], True)], -1) for a, c in zip(f, g)]
        x1, x2, x3, x4 = f
        x4 = mx.concatenate([resize(x1, x4.shape[1:3], True), resize(x2, x4.shape[1:3], True),
                             resize(x3, x4.shape[1:3], True), x4], -1)
        x4 = self.squeeze_module(x4)
        return self.decoder(x, x1, x2, x3, x4)


def load(folder) -> Model:
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    model = Model(config)
    q = config.get("quantization")
    if q:
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"], class_predicate=quantizable)
    model.load_weights(str(folder / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    return model


def dtype(model: Model):
    return model.decoder.conv_out1.weight.dtype


def predict(model: Model, rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> HxW float32 foreground probability: resized to the model size (bilinear, as PIL in the
    original handler) and back."""
    from PIL import Image
    h, w, _ = rgb.shape
    s = model.size
    x = np.asarray(Image.fromarray(rgb).resize((s, s), Image.BILINEAR), np.float32) / 255.0
    x = mx.array(((x - MEAN) / STD)[None]).astype(dtype(model))
    y = mx.sigmoid(model(x).astype(mx.float32))
    return np.array(resize(y, (h, w), True))[0, :, :, 0]


def matte(model: Model, rgb: np.ndarray) -> np.ndarray:
    return (np.clip(predict(model, rgb), 0, 1) * 255).round().astype(np.uint8)


# ------------------------------------------------------------------------------------------------ conversion

def _fold(sd, conv, bn):
    w = sd[conv + ".weight"].astype(np.float64)
    b = sd.get(conv + ".bias", np.zeros(w.shape[0])).astype(np.float64)
    if bn is not None:
        g = sd[bn + ".weight"] / np.sqrt(sd[bn + ".running_var"].astype(np.float64) + 1e-5)
        w = w * g[:, None, None, None]
        b = (b - sd[bn + ".running_mean"]) * g + sd[bn + ".bias"]
    return w.astype(np.float32).transpose(0, 2, 3, 1), b.astype(np.float32)


def convert(source: Path) -> dict:
    """runs/source/model.safetensors -> MLX names: BatchNorms folded, Sequential indices renamed, buffers and the
    training-only heads (conv_ms_spvn_*, gdt_convs_pred_*) dropped."""
    from safetensors.numpy import load_file
    sd = load_file(str(Path(source) / "model.safetensors"))
    out = {}
    for k, v in sd.items():
        if k.startswith("bb."):
            if k.endswith("relative_position_index"):
                continue
            out[k] = v.transpose(0, 2, 3, 1) if v.ndim == 4 else v
    blocks = sorted({k.split(".dec_att.")[0] for k in sd if ".dec_att." in k})
    for blk in blocks:                                  # decoder.decoder_blockN, squeeze_module.0
        dst = "squeeze_module" if blk == "squeeze_module.0" else blk
        out[dst + ".conv_in.weight"], out[dst + ".conv_in.bias"] = _fold(sd, blk + ".conv_in", blk + ".bn_in")
        out[dst + ".conv_out.weight"], out[dst + ".conv_out.bias"] = _fold(sd, blk + ".conv_out", blk + ".bn_out")
        a = blk + ".dec_att"
        for src_m, dst_m in [(a + ".aspp1", dst + ".dec_att.aspp1")] + [
                (f"{a}.aspp_deforms.{i}", f"{dst}.dec_att.aspp_deforms.{i}") for i in range(3)]:
            for c in ("offset_conv", "modulator_conv"):
                out[f"{dst_m}.{c}.weight"], out[f"{dst_m}.{c}.bias"] = _fold(sd, f"{src_m}.atrous_conv.{c}", None)
            out[f"{dst_m}.regular_conv.weight"], out[f"{dst_m}.regular_conv.bias"] = _fold(
                sd, f"{src_m}.atrous_conv.regular_conv", f"{src_m}.bn")
        out[dst + ".dec_att.gap_conv.weight"], out[dst + ".dec_att.gap_conv.bias"] = _fold(
            sd, a + ".global_avg_pool.1", a + ".global_avg_pool.2")
        out[dst + ".dec_att.conv1.weight"], out[dst + ".dec_att.conv1.bias"] = _fold(sd, a + ".conv1", a + ".bn1")
    d = "decoder."
    for i in range(1, 6):
        for c in ("conv1", "conv_out"):
            out[f"{d}ipt_blk{i}.{c}.weight"], out[f"{d}ipt_blk{i}.{c}.bias"] = _fold(sd, f"{d}ipt_blk{i}.{c}", None)
    for i in (2, 3, 4):
        out[f"{d}lateral_block{i}.weight"], out[f"{d}lateral_block{i}.bias"] = _fold(sd, f"{d}lateral_block{i}.conv", None)
        out[f"{d}gdt_convs_{i}.weight"], out[f"{d}gdt_convs_{i}.bias"] = _fold(sd, f"{d}gdt_convs_{i}.0", f"{d}gdt_convs_{i}.1")
        out[f"{d}gdt_attn_{i}.weight"], out[f"{d}gdt_attn_{i}.bias"] = _fold(sd, f"{d}gdt_convs_attn_{i}.0", None)
    out[d + "conv_out1.weight"], out[d + "conv_out1.bias"] = _fold(sd, d + "conv_out1.0", None)
    return out
