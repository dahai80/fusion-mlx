# SPDX-License-Identifier: Apache-2.0
# #989 Hunyuan3D-2.1 paint — Session 4 bake/rasterizer/UV/GLB tests.
from __future__ import annotations

import os

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
xatlas = pytest.importorskip("xatlas")


def _sphere():
    m = trimesh.creation.icosphere(subdivisions=2)
    return (
        np.asarray(m.vertices, dtype=np.float32),
        np.asarray(m.faces, dtype=np.int32),
        np.asarray(m.vertex_normals, dtype=np.float32),
    )


_AZ = [0, 90, 180, 270, 0, 180]
_EL = [0, 0, 0, 0, 90, -90]


def test_rasterize_views_shapes():
    from fusion_mlx.threed.paint.bake import rasterize_views

    verts, faces, normals = _sphere()
    r = rasterize_views(verts, faces, normals, _AZ, _EL, res=64)
    assert r["positions"].shape == (6, 64, 64, 3)
    assert r["normals"].shape == (6, 64, 64, 3)
    assert r["mask"].shape == (6, 64, 64)
    assert r["tri_id"].shape == (6, 64, 64)
    assert r["view_eye"].shape == (6, 3)
    # each view covers some pixels (sphere visible from all 6 views).
    assert all(r["mask"][i].any() for i in range(6))


def test_uv_unwrap_seam_split():
    from fusion_mlx.threed.paint.bake import uv_unwrap

    verts, faces, _ = _sphere()
    nv, nf, uv, vremap = uv_unwrap(verts, faces)
    # seam split increases vertex count; uv per new vertex; vremap maps back.
    assert nv.shape[1] == 3
    assert nf.shape[1] == 3
    assert uv.shape == (nv.shape[0], 2)
    assert vremap.shape == (nv.shape[0],)
    assert nv.shape[0] >= verts.shape[0]
    # uv in [0,1]
    assert uv.min() >= 0.0 and uv.max() <= 1.0


def test_bake_to_atlas_coverage():
    from fusion_mlx.threed.paint.bake import bake_to_atlas, rasterize_views, uv_unwrap

    verts, faces, normals = _sphere()
    r = rasterize_views(verts, faces, normals, _AZ, _EL, res=64)
    vt = np.zeros((6, 64, 64, 3), dtype=np.float32)
    colors = [
        [0.9, 0.1, 0.1],
        [0.1, 0.9, 0.1],
        [0.1, 0.1, 0.9],
        [0.9, 0.9, 0.1],
        [0.9, 0.1, 0.9],
        [0.1, 0.9, 0.9],
    ]
    for i, c in enumerate(colors):
        vt[i] = np.array(c, dtype=np.float32)
    nv, nf, uv, vremap = uv_unwrap(verts, faces)
    atlas = bake_to_atlas(nv, nf, uv, vremap, verts, normals, vt, r, atlas_res=128)
    assert atlas.shape == (128, 128, 3)
    # vertex coverage 1.0 (sphere seen from 6 views) -> atlas meaningfully filled.
    assert float(atlas.mean()) > 0.05


def test_export_glb(tmp_path):
    from fusion_mlx.threed.paint.bake import (
        bake_to_atlas,
        export_glb,
        rasterize_views,
        uv_unwrap,
    )

    verts, faces, normals = _sphere()
    r = rasterize_views(verts, faces, normals, _AZ, _EL, res=64)
    vt = np.zeros((6, 64, 64, 3), dtype=np.float32)
    for i in range(6):
        vt[i] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    nv, nf, uv, vremap = uv_unwrap(verts, faces)
    atlas = bake_to_atlas(nv, nf, uv, vremap, verts, normals, vt, r, atlas_res=128)
    glb_path = str(tmp_path / "test.glb")
    export_glb(nv, nf, uv, atlas, glb_path, texture_in_unit=False)
    assert os.path.exists(glb_path)
    assert os.path.getsize(glb_path) > 1000
    # reload to confirm valid GLB.
    m = trimesh.load(glb_path)
    assert m is not None
