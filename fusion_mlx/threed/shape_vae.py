# SPDX-License-Identifier: Apache-2.0
# Hunyuan3D-2.1 ShapeVAE decoder + geo-decoder (issue #989 Session 2).
# Latent (1, 4096, 64) -> /scale_factor -> post_kl(1024) -> 16 self-attn
# decoder blocks (per-head affine LayerNorm(64) qk-norm) -> geo cross-attn
# decoder (Fourier-encoded 3D query points 51->1024, K/V from ln2(latent))
# -> out_proj(1) occupancy -> marching cubes -> mesh.
# Weights: vae.safetensors (MLX-native 8bit, group 64).
# Forward ported from ddalcu/mlx-serve src/hunyuan3d.zig (VaeBlock + GeoDecoder).
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


class _DecoderAttention(nn.Module):
    # Self/cross-attn: q/k/v/out quantized (8bit, group 64). Per-head affine
    # LayerNorm(head_dim) on q and k (NOT RMSNorm — the VAE checkpoint ships
    # weight+bias for q_norm/k_norm). No v_norm.
    def __init__(self, dim: int, num_heads: int, head_dim: int, group_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.q = _qlinear(dim, dim, group_size, bias=False)
        self.k = _qlinear(dim, dim, group_size, bias=False)
        self.v = _qlinear(dim, dim, group_size, bias=False)
        self.out = _qlinear(dim, dim, group_size, bias=True)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

    def project_qk(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # x: (B, N, C). Returns q,k head-transposed (B, H, N, hd) after per-head LN.
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        return q, k

    def project_v(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim)
        return v.transpose(0, 2, 1, 3)

    def merge_heads(self, attn: mx.array) -> mx.array:
        B, H, N, hd = attn.shape
        return attn.transpose(0, 2, 1, 3).reshape(B, N, H * hd)


class _DecoderMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, group_size: int):
        super().__init__()
        self.fc1 = _qlinear(dim, hidden, group_size)
        self.fc2 = _qlinear(hidden, dim, group_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class DecoderBlock(nn.Module):
    # Self-attn block: x = x + attn(ln1(x)); x = x + mlp(ln2(x)).
    # q,k,v all projected from ln1(x) (shared input, standard ViT pre-norm).
    def __init__(self, dim: int, num_heads: int, head_dim: int, group_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _DecoderAttention(dim, num_heads, head_dim, group_size)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = _DecoderMLP(dim, dim * 4, group_size)

    def __call__(self, x: mx.array) -> mx.array:
        n1 = self.ln1(x)
        q, k = self.attn.project_qk(n1)
        v = self.attn.project_v(n1)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.attn.scale)
        o = self.attn.out(self.attn.merge_heads(attn))
        x = x + o
        x = x + self.mlp(self.ln2(x))
        return x


class FourierEncoder(nn.Module):
    # 3D point (N, 3) -> (N, 51). num_freqs=8, include_input=True:
    # [x, sin(2^0..2^7 * x), cos(...)] per axis, coordinate-major. NO pi
    # (the reference uses raw power-of-2 frequencies, not 2*pi*2^k).
    def __init__(self, num_freqs: int = 8, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        freqs = mx.array([2.0**i for i in range(num_freqs)], dtype=mx.float32)
        self.freqs = freqs  # (num_freqs,)

    def __call__(self, pts: mx.array) -> mx.array:
        # pts: (N, 3) float32 in [-1, 1].
        N = pts.shape[0]
        out_list = []
        if self.include_input:
            out_list.append(pts)  # (N, 3)
        scaled = pts[:, :, None] * self.freqs[None, None, :]  # (N, 3, num_freqs)
        out_list.append(mx.sin(scaled).reshape(N, -1))  # (N, 3*nf)
        out_list.append(mx.cos(scaled).reshape(N, -1))
        return mx.concatenate(out_list, axis=-1)  # (N, 51)


class GeoDecoder(nn.Module):
    # Cross-attn SDF head. K/V = projections of ln2(latent), computed once
    # per mesh (GeoKv). Q = Fourier-encoded query points -> query_proj ->
    # ln1 -> q_proj -> per-head q_norm. Forward: h1 = q_emb + attn; h2 = h1 +
    # mlp(ln3(h1)); logits = out_proj(ln_post(h2)). ln2 used ONLY for K/V prep.
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        group_size: int,
        query_in: int = 51,
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_in, dim)  # fp16
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.ln_post = nn.LayerNorm(dim)
        self.attn = _DecoderAttention(dim, num_heads, head_dim, group_size)
        self.mlp = _DecoderMLP(dim, dim * 4, group_size)
        self.out_proj = nn.Linear(dim, 1)  # fp16

    def prepare_kv(self, latent: mx.array) -> tuple[mx.array, mx.array]:
        # latent: (1, M, 1024). K/V head-split, k_norm applied.
        n2 = self.ln2(latent)
        q_dummy, k = self.attn.project_qk(n2)  # q unused; reuse k proj path
        del q_dummy
        v = self.attn.project_v(n2)
        return k, v

    def __call__(
        self, query_pts_fourier: mx.array, kv_k: mx.array, kv_v: mx.array
    ) -> mx.array:
        # query_pts_fourier: (N, 51). kv_k/kv_v: (1, H, M, hd).
        q_emb = self.query_proj(query_pts_fourier)  # (N, 1024)
        q_emb = q_emb[None, :, :]  # (1, N, 1024)
        n1 = self.ln1(q_emb)
        q, _ = self.attn.project_qk(n1)  # (1, H, N, hd), k from n1 unused
        attn = mx.fast.scaled_dot_product_attention(
            q, kv_k, kv_v, scale=self.attn.scale
        )
        o = self.attn.out(self.attn.merge_heads(attn))
        h1 = q_emb + o
        h2 = h1 + self.mlp(self.ln3(h1))
        np = self.ln_post(h2)
        logits = self.out_proj(np)  # (1, N, 1)
        # Cast to f32 (fp16 grid → stair-step artifacts; matches zig).
        return logits[0, :, 0].astype(mx.float32)  # (N,)


class ShapeVAEDecoder(nn.Module):
    def __init__(self, cfg: ShapeConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.vae_width  # 1024
        gs = cfg.dino_group_size  # 64
        head_dim = dim // cfg.vae_heads  # 64
        self.post_kl = nn.Linear(cfg.embed_dim, dim)  # 64 -> 1024 (fp16)
        self.blocks = [
            DecoderBlock(dim, cfg.vae_heads, head_dim, gs)
            for _ in range(cfg.vae_decoder_layers)  # 16
        ]
        self.geo = GeoDecoder(
            dim, cfg.vae_heads, head_dim, gs, query_in=2 * cfg.num_freqs * 3 + 3
        )
        self.fourier = FourierEncoder(num_freqs=cfg.num_freqs, include_input=True)
        self.scale_factor = cfg.scale_factor

    def decode_latent(self, latent: mx.array) -> mx.array:
        # latent: (1, 4096, 64) -> (1, 4096, 1024). Divide by scale_factor FIRST
        # (the reference `_export` order), then post_kl, then 16 self-attn blocks.
        x = self.post_kl(latent / self.scale_factor)
        for blk in self.blocks:
            x = blk(x)
        return x

    def prepare_geo_kv(self, decoded_latent: mx.array) -> tuple[mx.array, mx.array]:
        return self.geo.prepare_kv(decoded_latent)

    def query_occupancy(
        self, query_pts_xyz: mx.array, kv_k: mx.array, kv_v: mx.array
    ) -> mx.array:
        feats = self.fourier(query_pts_xyz)  # (N, 51)
        return self.geo(feats, kv_k, kv_v)  # (N,)

    def __call__(self, latent: mx.array, query_pts_xyz: mx.array) -> mx.array:
        # Convenience: full path (rebuilds kv each call — use prepare_geo_kv +
        # query_occupancy for chunked volume decode).
        decoded = self.decode_latent(latent)
        kv_k, kv_v = self.prepare_geo_kv(decoded)
        return self.query_occupancy(query_pts_xyz, kv_k, kv_v)


def load_shape_vae(weights_path: str, cfg: ShapeConfig) -> ShapeVAEDecoder:
    from safetensors import safe_open

    model = ShapeVAEDecoder(cfg)
    weights: dict[str, mx.array] = {}
    with safe_open(weights_path, framework="np") as f:
        for k in f.keys():  # noqa: SIM118
            weights[k] = mx.array(f.get_tensor(k))
    flat_before: dict = {}
    nn.utils.tree_flatten(model, destination=flat_before)
    model.load_weights(list(weights.items()), strict=False)
    remapped_keys = set(weights.keys())
    module_keys = set(flat_before.keys())
    missing = [k for k in module_keys if k not in remapped_keys]
    skipped = [k for k in remapped_keys if k not in module_keys]
    if missing:
        logger.warning("ShapeVAE missing weights (%d): %s", len(missing), missing[:8])
    if skipped:
        logger.warning(
            "ShapeVAE unexpected weights (%d): %s", len(skipped), skipped[:8]
        )
    logger.info(
        "ShapeVAE loaded: %d tensors, %d blocks, mapped=%d",
        len(weights),
        cfg.vae_decoder_layers,
        len(weights) - len(skipped),
    )
    return model


def decode_volume(
    model: ShapeVAEDecoder,
    latent: mx.array,
    res: int = 128,
    bound: float = 0.8,
    chunk_size: int = 8192,
) -> np.ndarray:
    # Build the SDF scalar grid (res+1)^3 over [-bound, bound]^3 by chunked
    # geo-decoder queries. latent: (1, 4096, 64). Returns numpy (n, n, n) f32,
    # x-major ij order idx=(i*n+j)*n+k. Inside = positive.
    n = res + 1
    total = n * n * n
    step = (2.0 * bound) / float(res)
    decoded = model.decode_latent(latent)
    kv_k, kv_v = model.prepare_geo_kv(decoded)
    grid = np.empty(total, dtype=np.float32)
    coords = np.empty((max(chunk_size, 1024), 3), dtype=np.float32)
    n_chunks = (total + chunk_size - 1) // chunk_size
    start = 0
    ci = 0
    while start < total:
        count = min(chunk_size, total - start)
        for j in range(count):
            idx = start + j
            ix = idx // (n * n)
            rem = idx % (n * n)
            iy = rem // n
            iz = rem % n
            coords[j, 0] = -bound + step * ix
            coords[j, 1] = -bound + step * iy
            coords[j, 2] = -bound + step * iz
        q = mx.array(coords[:count], dtype=mx.float32)
        logits = model.query_occupancy(q, kv_k, kv_v)
        mx.eval(logits)
        grid[start : start + count] = np.asarray(logits)
        start += count
        ci += 1
        if ci % 16 == 0:
            logger.info("decode_volume: chunk %d/%d", ci, n_chunks)
    logger.info("decode_volume: done %d points (res=%d, bound=%.3f)", total, res, bound)
    return grid.reshape(n, n, n)
