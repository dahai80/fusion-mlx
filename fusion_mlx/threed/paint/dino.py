# SPDX-License-Identifier: Apache-2.0
# DINOv2-Giant ViT conditioner for Hunyuan3D-2.1 paint (issue #989 Session 4).
# Image (B,3,518,518) -> 1370 tokens x 1536 (1 cls + 37*37 patch tokens).
# 8bit mlx-quantized (uint32 packed); MOST linears are 8-bit group 64, but the
# large mlp.w_out (8192->1536) is 4-bit group 128 (mixed precision to save
# space). Bits detected per-tensor at load from packed-cols vs expected in_dim;
# all weights dequantized to fp16 nn.Linear so the module is uniform fp16.
# Separate q/k/v/out (not fused QKV). mlp w_in/w_out (Giant naming).
# patch_embed.weight already MLX Conv2d layout (out,kh,kw,in) -> no transpose.
# Output feeds the paint UNet image_proj_model_dino (1536->4096 -> 4x1024).
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.threed.config import PaintDinoConfig

logger = logging.getLogger(__name__)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.out(attn)


class _MLP(nn.Module):
    # SwiGLU: w_in dim->mlp_hidden (fused gate+up, 2*intermediate), split,
    # gelu(gate)*up -> intermediate, w_out intermediate->dim.
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_in = nn.Linear(dim, hidden, bias=True)
        self.w_out = nn.Linear(hidden // 2, dim, bias=True)
        self.intermediate = hidden // 2

    def __call__(self, x: mx.array) -> mx.array:
        h = self.w_in(x)
        gate, up = h[..., : self.intermediate], h[..., self.intermediate :]
        return self.w_out(nn.gelu_approx(gate) * up)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, hidden)
        self.ls1 = mx.ones((dim,))
        self.ls2 = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class PaintDINOv2(nn.Module):
    # DINOv2-Giant ViT: 40 layers, hidden 1536, 24 heads (head_dim 64),
    # patch 14, image 518, mlp_hidden 8192. Outputs (B, 1370, 1536).

    def __init__(self, cfg: PaintDinoConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.hidden
        self.patch_embed = nn.Conv2d(
            3, dim, kernel_size=cfg.patch, stride=cfg.patch, bias=True
        )
        self.cls_token = mx.zeros((1, 1, dim))
        self.pos_embed = mx.zeros((1, cfg.num_tokens, dim))
        self.blocks = [
            _Block(dim, cfg.heads, cfg.head_dim, cfg.mlp_hidden)
            for _ in range(cfg.layers)
        ]
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, 3, 518, 518) ImageNet-normalized [0,1]. MLX Conv2d = NHWC.
        B = x.shape[0]
        dim = self.cfg.hidden
        x = x.transpose(0, 2, 3, 1)
        patches = self.patch_embed(x)  # (B, 37, 37, dim)
        h = patches.shape[1]
        patches = patches.reshape(B, h * h, -1)  # (B, 1369, dim)
        cls = mx.broadcast_to(self.cls_token, (B, 1, dim))
        x = mx.concatenate([cls, patches], axis=1)  # (B, 1370, dim)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


def _detect_bits(packed: int, in_dim: int) -> tuple[int, int]:
    # packed = in_dim * bits / 32 -> bits = 32 * packed / in_dim.
    # Must be 4 or 8. group_size = in_dim // groups (caller passes groups).
    bits = round(32 * packed / in_dim)
    if bits not in (4, 8):
        # fallback: try 8 then 4
        bits = 8 if in_dim % (packed * 4) == 0 else 4
    return bits


def load_paint_dinov2(weights_path: str, cfg: PaintDinoConfig) -> PaintDINOv2:
    # dino.safetensors keys map 1:1 except "layers.N" -> "blocks.N".
    # patch_embed.weight already MLX layout -> no transpose. Quantized linears
    # dequantized to fp16 nn.Linear at load (bits detected per-tensor: most are
    # 8-bit gs=64, mlp.w_out is 4-bit gs=128).
    from safetensors import safe_open

    model = PaintDINOv2(cfg)
    raw: dict[str, mx.array] = {}
    with safe_open(weights_path, framework="np") as f:
        for k in f.keys():  # noqa: SIM118
            raw[k] = mx.array(f.get_tensor(k))

    flat: dict = {}
    nn.utils.tree_flatten(model, destination=flat)
    module_keys = set(flat.keys())

    # module expected in_dim per Linear weight (from nn.Linear shape [out,in]).
    expected_in: dict[str, int] = {}
    for k in flat:
        if k.endswith(".weight") and flat[k].ndim == 2:
            expected_in[k] = flat[k].shape[1]

    remapped: dict[str, mx.array] = {}
    consumed: set[str] = set()
    for k, v in raw.items():
        nk = "blocks." + k[len("layers.") :] if k.startswith("layers.") else k
        if nk.endswith(".scales") or nk.endswith(".biases"):
            consumed.add(k)
            continue
        if nk.endswith(".weight") and k.replace(".weight", ".scales") in raw:
            base = nk[: -len(".weight")]
            scales = raw[f"{k[: -len('.weight')]}.scales"]
            biases = raw[f"{k[: -len('.weight')]}.biases"]
            packed = v.shape[1]
            in_dim = expected_in.get(nk, packed * 4)
            groups = scales.shape[1]
            bits = _detect_bits(packed, in_dim)
            group_size = in_dim // groups
            dq = mx.dequantize(v, scales, biases, group_size=group_size, bits=bits)
            remapped[f"{base}.weight"] = dq.astype(mx.float16)
            consumed.add(k)
            bias_key = f"{base}.bias"
            raw_bias = k[: -len(".weight")] + ".bias"
            if raw_bias in raw:
                remapped[bias_key] = raw[raw_bias]
                consumed.add(raw_bias)
            continue
        if k not in consumed:
            remapped[nk] = v

    to_load = [(k, v) for k, v in remapped.items() if k in module_keys]
    missing = [
        k
        for k in module_keys
        if k not in remapped
        and not k.endswith((".stride.0", ".stride.1", ".padding.0", ".padding.1"))
    ]
    unexpected = [k for k in remapped if k not in module_keys]
    model.load_weights(to_load, strict=False)
    if missing:
        logger.warning("PaintDINOv2 missing %d: %s", len(missing), missing[:6])
    if unexpected:
        logger.warning("PaintDINOv2 unexpected %d: %s", len(unexpected), unexpected[:6])
    logger.info(
        "PaintDINOv2 loaded: %d/%d tensors (missing=%d unexpected=%d), %d layers",
        len(to_load),
        len(flat),
        len(missing),
        len(unexpected),
        cfg.layers,
    )
    return model
