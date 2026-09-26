# SPDX-License-Identifier: Apache-2.0
# #989 Hunyuan3D-2.1 MLX port — Session 2 tests: ShapeVAE decoder + geo
# decoder + marching cubes mesh extraction. Real-weight smoke gated by
# real_model marker (needs 310MB vae.safetensors on disk).
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.threed.config import ShapeConfig, load_shape_config
from fusion_mlx.threed.shape_vae import (
    FourierEncoder,
    GeoDecoder,
    ShapeVAEDecoder,
)

_HUNYUAN3D_DIR = Path(
    os.path.expanduser("~/.fusion-mlx/models/Hunyuan3D-2.1-MLX-Serve-8bit")
)


def test_fourier_encoder_shape_and_freqs():
    enc = FourierEncoder(num_freqs=8, include_input=True)
    pts = mx.array(np.linspace(-1, 1, 5 * 3, dtype=np.float32).reshape(5, 3))
    out = enc(pts)
    mx.eval(out)
    assert out.shape == (5, 51)  # 3 + 2*8*3
    # NO pi scaling: freqs are raw 2^k (1,2,4,...,128), not 2*pi*2^k.
    assert float(enc.freqs[0]) == 1.0
    assert float(enc.freqs[1]) == 2.0
    assert float(enc.freqs[7]) == 128.0
    assert bool(mx.all(mx.isfinite(out)).item())


def test_shape_vae_structural_forward():
    cfg = ShapeConfig()
    model = ShapeVAEDecoder(cfg)
    latent = mx.random.normal((1, 4096, 64), dtype=mx.float16) * 0.3
    decoded = model.decode_latent(latent)
    mx.eval(decoded)
    assert decoded.shape == (1, 4096, cfg.vae_width)
    assert bool(mx.all(mx.isfinite(decoded)).item())
    kv_k, kv_v = model.prepare_geo_kv(decoded)
    mx.eval(kv_k, kv_v)
    assert kv_k.shape == (1, cfg.vae_heads, 4096, cfg.vae_width // cfg.vae_heads)
    assert kv_v.shape == kv_k.shape
    q = mx.array(np.random.uniform(-0.8, 0.8, (8, 3)).astype(np.float32))
    occ = model.query_occupancy(q, kv_k, kv_v)
    mx.eval(occ)
    assert occ.shape == (8,)
    assert bool(mx.all(mx.isfinite(occ)).item())


def test_geo_decoder_module_structure():
    cfg = ShapeConfig()
    head_dim = cfg.vae_width // cfg.vae_heads
    geo = GeoDecoder(
        cfg.vae_width, cfg.vae_heads, head_dim, cfg.dino_group_size, query_in=51
    )
    # q/k/v bias=False (no linear bias in checkpoint); out/mlp bias=True.
    assert not hasattr(geo.attn.q, "bias") or geo.attn.q.bias is None
    assert hasattr(geo.attn.out, "bias")
    # ln1 (pre-attn), ln2 (K/V prep), ln3 (pre-mlp), ln_post (final) all present.
    for name in ("ln1", "ln2", "ln3", "ln_post"):
        assert hasattr(geo, name)
    # q_norm + k_norm (per-head affine LayerNorm on head_dim).
    assert hasattr(geo.attn, "q_norm")
    assert hasattr(geo.attn, "k_norm")


# ---------- marching cubes (pure numpy, no model) ----------


def _sphere_grid(np_pts: int, center, r0: float) -> np.ndarray:
    step = 2.0 / (np_pts - 1)
    g = np.empty((np_pts, np_pts, np_pts), dtype=np.float32)
    for i in range(np_pts):
        for j in range(np_pts):
            for k in range(np_pts):
                x = i * step - 1.0 - center[0]
                y = j * step - 1.0 - center[1]
                z = k * step - 1.0 - center[2]
                g[i, j, k] = r0 - float(np.sqrt(x * x + y * y + z * z))
    return g


def _edge_stats(indices: np.ndarray) -> tuple[int, bool]:
    counts: dict[int, int] = {}
    for tri in indices:
        for e in range(3):
            lo, hi = int(tri[e]), int(tri[(e + 1) % 3])
            if lo > hi:
                lo, hi = hi, lo
            key = (lo << 32) | hi
            counts[key] = counts.get(key, 0) + 1
    return len(counts), all(v == 2 for v in counts.values())


def test_marching_cubes_sphere_closed_manifold():
    from fusion_mlx.threed.marching_cubes import extract

    np_pts = 48
    r0 = 0.35
    grid = _sphere_grid(np_pts, (0, 0, 0), r0)
    step = 2.0 / (np_pts - 1)
    verts, norms, idx = extract(
        grid, level=0.0, scale=(step, step, step), offset=(-1, -1, -1)
    )
    assert verts.shape[0] > 0 and idx.shape[0] > 0
    n_edges, all_twice = _edge_stats(idx)
    assert all_twice  # closed 2-manifold
    # Euler characteristic V - E + F == 2 for genus-0 closed surface.
    chi = verts.shape[0] - n_edges + idx.shape[0]
    assert chi == 2


def test_marching_cubes_sphere_outward_normals():
    from fusion_mlx.threed.marching_cubes import extract

    np_pts = 48
    r0 = 0.35
    grid = _sphere_grid(np_pts, (0, 0, 0), r0)
    step = 2.0 / (np_pts - 1)
    verts, norms, idx = extract(
        grid, level=0.0, scale=(step, step, step), offset=(-1, -1, -1)
    )
    r = np.sqrt((verts**2).sum(axis=1))
    assert float(np.abs(r - r0).max()) < step  # vertices on sphere surface
    # Normals unit length + pointing radially outward.
    nl = np.sqrt((norms**2).sum(axis=1))
    assert float(np.abs(nl - 1.0).max()) < 1e-3
    dot = (verts * norms).sum(axis=1) / r
    assert float(dot.min()) > 0.9


def test_marching_cubes_empty_grid():
    from fusion_mlx.threed.marching_cubes import extract

    grid = np.full((16, 16, 16), -1.0, dtype=np.float32)
    verts, norms, idx = extract(grid, level=0.0)
    assert verts.shape[0] == 0
    assert idx.shape[0] == 0


def test_marching_cubes_two_spheres_euler_4():
    from fusion_mlx.threed.marching_cubes import extract

    np_pts = 64
    step = 2.0 / (np_pts - 1)
    grid = np.empty((np_pts, np_pts, np_pts), dtype=np.float32)
    for i in range(np_pts):
        for j in range(np_pts):
            for k in range(np_pts):
                x = i * step - 1.0
                y = j * step - 1.0
                z = k * step - 1.0
                d0 = float(np.sqrt((x + 0.45) ** 2 + y * y + z * z))
                d1 = float(np.sqrt((x - 0.45) ** 2 + y * y + z * z))
                grid[i, j, k] = max(0.2 - d0, 0.2 - d1)
    verts, norms, idx = extract(
        grid, level=0.0, scale=(step, step, step), offset=(-1, -1, -1)
    )
    n_edges, all_twice = _edge_stats(idx)
    assert all_twice
    chi = verts.shape[0] - n_edges + idx.shape[0]
    assert chi == 4  # two disjoint genus-0 surfaces


# ---------- real-weight smoke ----------


@pytest.mark.real_model
def test_shape_vae_real_weight_load_and_forward():
    if not _HUNYUAN3D_DIR.exists():
        pytest.skip("Hunyuan3D-2.1-MLX-Serve-8bit weights not downloaded")
    from fusion_mlx.threed.shape_vae import load_shape_vae

    cfg = load_shape_config(_HUNYUAN3D_DIR)
    model = load_shape_vae(str(_HUNYUAN3D_DIR / "vae.safetensors"), cfg)
    latent = mx.random.normal((1, 4096, 64), dtype=mx.float16) * 0.3
    decoded = model.decode_latent(latent)
    mx.eval(decoded)
    assert decoded.shape == (1, 4096, cfg.vae_width)
    da = np.asarray(decoded)
    assert bool(np.isfinite(da).all())
    # Real weights produce nonzero spread (not zeros).
    assert float(da.astype(np.float32).std()) > 0.1
    kv_k, kv_v = model.prepare_geo_kv(decoded)
    mx.eval(kv_k, kv_v)
    assert kv_k.shape == (1, cfg.vae_heads, 4096, cfg.vae_width // cfg.vae_heads)
    q = mx.array(np.random.uniform(-0.8, 0.8, (16, 3)).astype(np.float32))
    occ = model.query_occupancy(q, kv_k, kv_v)
    mx.eval(occ)
    assert occ.shape == (16,)
    assert bool(np.all(np.isfinite(np.asarray(occ))))


@pytest.mark.real_model
def test_shape_vae_decode_volume_pipeline():
    # End-to-end: latent -> volume grid -> marching cubes -> OBJ file.
    # Random latent yields no real object (field stays negative), so we only
    # assert the pipeline runs and produces a finite grid; MC on the raw grid
    # may be empty. Real mesh extraction needs a DiT-sampled latent (Session 3).
    if not _HUNYUAN3D_DIR.exists():
        pytest.skip("Hunyuan3D-2.1-MLX-Serve-8bit weights not downloaded")
    from fusion_mlx.threed.marching_cubes import extract, write_obj
    from fusion_mlx.threed.shape_vae import decode_volume, load_shape_vae

    cfg = load_shape_config(_HUNYUAN3D_DIR)
    model = load_shape_vae(str(_HUNYUAN3D_DIR / "vae.safetensors"), cfg)
    latent = mx.random.normal((1, 4096, 64), dtype=mx.float16) * 0.3
    grid = decode_volume(model, latent, res=24, bound=0.8, chunk_size=4096)
    assert grid.shape == (25, 25, 25)
    assert bool(np.isfinite(grid).all())
    # MC must run on the decoded grid without error (mesh may be empty).
    step = 1.6 / 24
    verts, norms, idx = extract(
        grid, level=0.0, scale=(step, step, step), offset=(-0.8, -0.8, -0.8)
    )
    obj_path = _HUNYUAN3D_DIR.parent / "threed_session2_smoke.obj"
    write_obj(str(obj_path), verts, norms, idx)
    assert obj_path.exists()
    obj_path.unlink()  # cleanup process data
