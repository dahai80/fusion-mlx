# SPDX-License-Identifier: Apache-2.0
# Hunyuan3D-2.1 flow-match DiT denoiser (issue #989 Session 3).
# U-ViT (depth 21, hidden 2048, 16 heads, head_dim 128) with timestep-as-token-0,
# U-Net skips (blocks 0..9 push, 11..20 pop LIFO), cross-attn into DINOv2 image
# tokens, and MoE FFN on the last 6 blocks (8 experts top-2, softmax-then-topk
# NO renorm, plus an always-on shared expert). Reversed flow-match Euler with
# CFG. Forward ported from ddalcu/mlx-serve src/hunyuan3d.zig (Dit + DitBlock +
# DitAttn + Moe + denoise + buildSigmas + timestepEmbed).
# Weights: dit.safetensors (MLX-native 8bit, group 64; experts stacked [E,...]).
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ShapeConfig

logger = logging.getLogger(__name__)


def _qlinear(
    dim_in: int, dim_out: int, group_size: int = 64, bias: bool = True
) -> nn.quantized.QuantizedLinear:
    return nn.quantized.QuantizedLinear(
        dim_in, dim_out, bias=bias, group_size=group_size, bits=8
    )


def gelu_erf(x: mx.array) -> mx.array:
    # Exact (erf) GELU — every GELU in this model uses erf, not tanh approx.
    return x * 0.5 * (1.0 + mx.erf(x / 1.4142135623730951))


def timestep_embed(t: float, dim: int) -> mx.array:
    # Sincos [1, dim] f32. half=dim/2, f_i = exp(-ln(1e4)*i/half),
    # emb = [sin(t*f_0..), cos(t*f_0..)] — SIN FIRST, f32 (bf16/f16 sincos of
    # tiny sigma args is a parity killer).
    half = dim // 2
    i = np.arange(half, dtype=np.float64)
    freqs = np.exp(-np.log(10000.0) * i / half)
    ang = float(t) * freqs
    buf = np.empty(dim, dtype=np.float32)
    buf[:half] = np.sin(ang)
    buf[half:] = np.cos(ang)
    return mx.array(buf.reshape(1, dim))


class _RMSNorm(nn.Module):
    # RMSNorm with weight only (no bias, no eps shift) — the DiT per-head qk-norm.
    # Operates over the last axis. weight shape [head_dim].
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,), dtype=mx.float16)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: [..., dim]. Compute RMS over last axis in f32 for parity.
        xf = x.astype(mx.float32)
        ms = mx.mean(xf * xf, axis=-1, keepdims=True)
        rs = mx.rsqrt(ms + self.eps)
        return (xf * rs * self.weight.astype(mx.float32)).astype(x.dtype)


class DiTAttention(nn.Module):
    # q/k/v/out quantized. q from q_in, k/v from kv_in. Per-head RMSNorm(128)
    # on q and k (weight only). v not normed. Self-attn: q_in=kv_in=dim.
    # Cross-attn: q_in=dim, kv_in=context_dim.
    def __init__(
        self,
        dim: int,
        q_in: int,
        kv_in: int,
        num_heads: int,
        head_dim: int,
        group_size: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.q = _qlinear(q_in, dim, group_size, bias=False)
        self.k = _qlinear(kv_in, dim, group_size, bias=False)
        self.v = _qlinear(kv_in, dim, group_size, bias=False)
        self.out = _qlinear(dim, dim, group_size, bias=True)
        self.q_norm = _RMSNorm(head_dim)
        self.k_norm = _RMSNorm(head_dim)

    def _project_q(self, xq: mx.array) -> mx.array:
        B, N, _ = xq.shape
        q = self.q(xq).reshape(B, N, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        return q.transpose(0, 2, 1, 3)

    def _project_kv(self, xkv: mx.array) -> tuple[mx.array, mx.array]:
        B, N, _ = xkv.shape
        k = self.k(xkv).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v(xkv).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_norm(k)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        return k, v

    def __call__(self, xq: mx.array, xkv: mx.array) -> mx.array:
        q = self._project_q(xq)
        k, v = self._project_kv(xkv)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        B, H, N, hd = attn.shape
        merged = attn.transpose(0, 2, 1, 3).reshape(B, N, H * hd)
        return self.out(merged)


class DiTMLP(nn.Module):
    # fc1 (quant) + erf-gelu + fc2 (quant). Both have bias.
    def __init__(self, dim: int, inter: int, group_size: int):
        super().__init__()
        self.fc1 = _qlinear(dim, inter, group_size, bias=True)
        self.fc2 = _qlinear(inter, dim, group_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(gelu_erf(self.fc1(x)))


class ExpertStack(nn.Module):
    # E independent QuantizedLinears (sliced from the stacked checkpoint tensor
    # at load time). Forward applies expert e to ALL tokens; the caller masks.
    def __init__(self, dim_in: int, dim_out: int, n_experts: int, group_size: int):
        super().__init__()
        self.linears = [
            _qlinear(dim_in, dim_out, group_size, bias=True) for _ in range(n_experts)
        ]

    def __call__(self, x: mx.array, expert_idx: int) -> mx.array:
        return self.linears[expert_idx](x)


class _Experts(nn.Module):
    # Container matching checkpoint key layout moe.experts.fc1 / moe.experts.fc2.
    # Each ExpertStack holds E independent QuantizedLinears (sliced from the
    # stacked [E, ...] tensor at load time).
    def __init__(self, dim: int, inter: int, n_experts: int, group_size: int):
        super().__init__()
        self.fc1 = ExpertStack(dim, inter, n_experts, group_size)
        self.fc2 = ExpertStack(inter, dim, n_experts, group_size)


class MoE(nn.Module):
    # gate Linear(dim, E) fp16. experts (fc1/fc2 stacked). shared always-on MLP.
    # top-k routing: softmax over ALL experts, then topk on probs, NO renorm.
    def __init__(
        self, dim: int, inter: int, n_experts: int, top_k: int, group_size: int
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, n_experts, bias=False)  # fp16, no gate bias
        self.experts = _Experts(dim, inter, n_experts, group_size)
        self.shared = DiTMLP(dim, inter, group_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, L, D].
        B, L, D = x.shape
        logits = self.gate(x)  # [B, L, E]
        probs = mx.softmax(logits, axis=-1)
        part = mx.argpartition(-probs, self.top_k - 1, axis=-1)
        inds = part[..., : self.top_k]  # [B, L, K]
        weights = mx.take_along_axis(probs, inds, axis=-1)  # [B, L, K]

        out = self.shared(x)  # always-on shared expert [B, L, D]
        for e in range(self.n_experts):
            h = self.experts.fc2(gelu_erf(self.experts.fc1(x, e)), e)  # [B, L, D]
            for k in range(self.top_k):
                sel = (inds[..., k] == e).astype(x.dtype)  # [B, L]
                w = weights[..., k]  # [B, L]
                out = out + (sel * w)[..., None] * h
        return out


class _Skip(nn.Module):
    # U-ViT skip fuse: LayerNorm(Linear(cat([skip, h], -1))).
    def __init__(self, dim: int, group_size: int):
        super().__init__()
        self.linear = _qlinear(dim * 2, dim, group_size, bias=True)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, skip: mx.array, h: mx.array) -> mx.array:
        cat = mx.concatenate([skip, h], axis=-1)
        return self.norm(self.linear(cat))


class DiTBlock(nn.Module):
    # norm1 + self-attn (attn1) + norm2 + cross-attn (attn2) + norm3 + ffn
    # (dense MLP or MoE) + optional U-ViT skip. Attribute names (mlp/moe/skip)
    # match the checkpoint key layout for direct weight loading.
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        group_size: int,
        use_moe: bool,
        inter: int,
        n_experts: int,
        top_k: int,
        has_skip: bool,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = DiTAttention(dim, dim, dim, num_heads, head_dim, group_size)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = DiTAttention(
            dim, dim, context_dim, num_heads, head_dim, group_size
        )
        self.norm3 = nn.LayerNorm(dim)
        if use_moe:
            self.moe = MoE(dim, inter, n_experts, top_k, group_size)
        else:
            self.mlp = DiTMLP(dim, inter, group_size)
        self.has_skip = has_skip
        if has_skip:
            self.skip = _Skip(dim, group_size)

    def __call__(self, h: mx.array, cond: mx.array, skip: mx.array | None) -> mx.array:
        if skip is not None:
            h = self.skip(skip, h)
        h = h + self.attn1(self.norm1(h), self.norm1(h))
        h = h + self.attn2(self.norm2(h), cond)
        ffn = self.moe if self.use_moe else self.mlp
        h = h + ffn(self.norm3(h))
        return h


class _TEmbedder(nn.Module):
    # t_embedder.mlp1 / t_embedder.mlp2 — matches checkpoint key layout.
    def __init__(self, dim: int, inter: int, group_size: int):
        super().__init__()
        self.mlp1 = _qlinear(dim, inter, group_size, bias=True)
        self.mlp2 = _qlinear(inter, dim, group_size, bias=True)

    def __call__(self, t_vec: mx.array) -> mx.array:
        return self.mlp2(gelu_erf(self.mlp1(t_vec)))


class _Final(nn.Module):
    # final.norm / final.linear — matches checkpoint key layout.
    def __init__(self, dim: int, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, embed_dim)

    def __call__(self, h: mx.array) -> mx.array:
        return self.linear(self.norm(h))


class HunyuanDiT(nn.Module):
    def __init__(self, cfg: ShapeConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.hidden_size  # 2048
        ctx = cfg.context_dim  # 1024
        heads = cfg.num_heads  # 16
        head_dim = dim // heads  # 128
        gs = cfg.dino_group_size  # 64
        inter = dim * 4  # 8192
        moe_start = cfg.depth - cfg.num_moe_layers  # 15
        pop_from = (cfg.depth - 1) // 2 + 1  # 11
        self.half_push = (cfg.depth - 1) // 2  # 10
        self.pop_from = pop_from
        self.x_embedder = nn.Linear(cfg.embed_dim, dim)  # 64 -> 2048 fp16
        self.t_embedder = _TEmbedder(dim, inter, gs)
        self.blocks = [
            DiTBlock(
                dim,
                ctx,
                heads,
                head_dim,
                gs,
                use_moe=(i >= moe_start),
                inter=inter,
                n_experts=cfg.num_experts,
                top_k=cfg.moe_top_k,
                has_skip=(i >= pop_from),
            )
            for i in range(cfg.depth)
        ]
        self.final = _Final(dim, cfg.embed_dim)

    def forward(self, x: mx.array, cond: mx.array, sigma: float) -> mx.array:
        # x: [B, 4096, 64] f16. cond: [B, 1370, 1024] f16. sigma in [0,1].
        # Returns velocity [B, 4096, 64] f16.
        B = x.shape[0]
        t_vec = timestep_embed(sigma, self.cfg.hidden_size)  # [1, dim] f32
        t2 = self.t_embedder(t_vec.astype(mx.float16))  # [1, dim]
        t_tok = mx.broadcast_to(t2, (B, 1, self.cfg.hidden_size))  # [B, 1, dim]
        x_e = self.x_embedder(x)  # [B, 4096, dim]
        h = mx.concatenate([t_tok, x_e], axis=1)  # [B, 4097, dim]

        skips: list[mx.array] = []
        for i, blk in enumerate(self.blocks):
            skip = None
            if i >= self.pop_from:
                skip = skips.pop()
            h = blk(h, cond, skip)
            if i < self.half_push:
                skips.append(h)
        h = self.final(h)
        # Drop token 0 (timestep) AFTER final norm + projection.
        return h[:, 1:, :]  # [B, 4096, embed_dim]

    def __call__(self, x: mx.array, cond: mx.array, sigma: float) -> mx.array:
        return self.forward(x, cond, sigma)


def build_sigmas(steps: int) -> np.ndarray:
    # Reversed flow-match: linspace(0,1,steps) ASCENDING + appended 1.0.
    # len = steps+1. The trailing sigma's delta is 0 (skipped in the loop).
    if steps < 2:
        raise ValueError("steps must be >= 2")
    out = np.empty(steps + 1, dtype=np.float32)
    out[:steps] = np.arange(steps, dtype=np.float32) / (steps - 1)
    out[steps] = 1.0
    return out


def denoise(
    dit: HunyuanDiT,
    cond: mx.array,
    steps: int = 50,
    guidance: float = 5.0,
    seed: int = 0,
    init_noise: mx.array | None = None,
) -> mx.array:
    # Flow-match Euler with CFG. cond [1, 1370, 1024]. Returns latent [1, 4096, 64] f32.
    # ctx2 = [cond; zeros] — uncond is output-level zeros, not a black image.
    cfg = dit.cfg
    sigmas = build_sigmas(steps)
    if init_noise is not None:
        x = init_noise.astype(mx.float32)
    else:
        key = mx.random.key(seed)
        x = mx.random.normal(
            (1, cfg.num_latents, cfg.embed_dim), dtype=mx.float32, key=key
        )
    zeros_ctx = mx.zeros(cond.shape, dtype=mx.float16)
    cond_h = cond.astype(mx.float16)
    ctx2 = mx.concatenate([cond_h, zeros_ctx], axis=0)  # [2, 1370, 1024]
    g = float(guidance)
    for i in range(steps):
        ds = float(sigmas[i + 1] - sigmas[i])
        if ds == 0.0:
            continue
        xh = x.astype(mx.float16)
        x2 = mx.concatenate([xh, xh], axis=0)  # [2, 4096, 64]
        v = dit.forward(x2, ctx2, float(sigmas[i]))  # [2, 4096, 64]
        v_c = v[0:1].astype(mx.float32)
        v_u = v[1:2].astype(mx.float32)
        guided = v_u + g * (v_c - v_u)
        x = x + guided * ds
        mx.eval(x)
        if (i + 1) % 10 == 0:
            logger.info("denoise: step %d/%d (sigma=%.3f)", i + 1, steps, sigmas[i])
    logger.info("denoise: done (%d steps, guidance=%.2f)", steps, g)
    return x


def _split_expert_weights(
    weights: dict[str, mx.array], n_experts: int, prefix: str
) -> dict[str, list[mx.array]]:
    # Stacked expert tensor [E, out, in//4] -> E slices per parameter.
    # Returns {param_name: [tensor_e for e in range(E)]}.
    out: dict[str, list[mx.array]] = {}
    base = f"{prefix}."
    for k, v in weights.items():
        if not k.startswith(base):
            continue
        suffix = k[len(base) :]  # e.g. "weight", "scales", "biases", "bias"
        if v.ndim == 3 and v.shape[0] == n_experts:
            out.setdefault(suffix, []).extend([v[e] for e in range(n_experts)])
        elif v.ndim == 2 and v.shape[0] == n_experts:
            # bias [E, out]
            out.setdefault(suffix, []).extend([v[e] for e in range(n_experts)])
    return out


def load_dit(weights_path: str, cfg: ShapeConfig) -> HunyuanDiT:
    from safetensors import safe_open

    model = HunyuanDiT(cfg)
    raw: dict[str, mx.array] = {}
    with safe_open(weights_path, framework="np") as f:
        for k in f.keys():  # noqa: SIM118
            raw[k] = mx.array(f.get_tensor(k))

    moe_start = cfg.depth - cfg.num_moe_layers
    # Split stacked expert weights into per-expert params and inject under
    # experts_fc1.<e>.<param> / experts_fc2.<e>.<param>.
    remapped: dict[str, mx.array] = {}
    consumed: set[str] = set()
    for i in range(moe_start, cfg.depth):
        for sub in ("fc1", "fc2"):
            pfx = f"blocks.{i}.moe.experts.{sub}"
            splits = _split_expert_weights(raw, cfg.num_experts, pfx)
            for suffix, lst in splits.items():
                for e, tensor in enumerate(lst):
                    remapped[f"blocks.{i}.moe.experts.{sub}.linears.{e}.{suffix}"] = (
                        tensor
                    )
            for k in raw:
                if k.startswith(pfx + "."):
                    consumed.add(k)

    flat: dict = {}
    nn.utils.tree_flatten(model, destination=flat)
    module_keys = set(flat.keys())
    # Map non-expert weights directly (names already align).
    direct = {k: v for k, v in raw.items() if k not in consumed}
    all_weights = {**direct, **remapped}
    model.load_weights(list(all_weights.items()), strict=False)
    missing = [k for k in module_keys if k not in all_weights]
    skipped = [k for k in all_weights if k not in module_keys]
    if missing:
        logger.warning("DiT missing weights (%d): %s", len(missing), missing[:8])
    if skipped:
        logger.warning("DiT unexpected weights (%d): %s", len(skipped), skipped[:8])
    logger.info(
        "DiT loaded: %d tensors, depth %d, %d MoE blocks (experts=%d, k=%d)",
        len(raw),
        cfg.depth,
        cfg.num_moe_layers,
        cfg.num_experts,
        cfg.moe_top_k,
    )
    return model
