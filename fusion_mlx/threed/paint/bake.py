# SPDX-License-Identifier: Apache-2.0
# Hunyuan3D-2.1 paint multiview rasterizer + UV unwrap + texture bake-back +
# GLB export (issue #989 Session 4).
#
# Stages:
#   1. rasterize_views: numpy software rasterizer (z-buffer, per-triangle
#      barycentric) renders the mesh from N camera azimuths/elevs at `res` ->
#      position map, normal map, mask, triangle-id map, barycentric per pixel.
#   2. uv_unwrap: xatlas parameterizes the mesh -> UV coords + seam-split
#      verts/faces.
#   3. bake_to_atlas: for each atlas texel, map to mesh point, find the view
#      that sees it (smallest viewing angle), sample the generated view texture.
#   4. export_glb: trimesh Trimesh with UV + texture -> .glb.
#
# Pure numpy + xatlas + trimesh. No GPU/Metal. Resolution configurable
# (default 256 — 512 is the paint target but slow in pure numpy).
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# Camera basis for azimuth (deg, CCW around Y) + elevation (deg, up from XZ).
def _camera(
    azim: float, elev: float, radius: float = 2.5
) -> tuple[np.ndarray, np.ndarray]:
    az = np.radians(azim)
    el = np.radians(elev)
    eye = np.array(
        [
            radius * np.cos(el) * np.sin(az),
            radius * np.sin(el),
            radius * np.cos(el) * np.cos(az),
        ]
    )
    target = np.zeros(3)
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    # view matrix rows: right, up, -fwd (camera-space); translate by -eye.
    R = np.stack([right, up, -fwd], axis=0)  # (3,3)
    view = np.eye(4)
    view[:3, :3] = R
    view[:3, 3] = -R @ eye
    return view, eye


def _project(
    verts: np.ndarray, view: np.ndarray, res: int
) -> tuple[np.ndarray, np.ndarray]:
    # verts (V,3) -> screen (V,2) in [0,res], depth (V,).
    V = verts.shape[0]
    homog = np.concatenate([verts, np.ones((V, 1))], axis=1)  # (V,4)
    cam = (view @ homog.T).T  # (V,4) camera space
    # perspective: assume mesh ~ unit cube at origin, radius 2.5 -> near/far.
    z = cam[:, 2]
    # orthographic-ish: map cam z [-3, -0.5] -> keep; project x,y.
    # Use simple perspective: x' = cam.x / -cam.z
    depth = -z  # positive in front
    valid = depth > 1e-3
    x = cam[:, 0] / np.where(valid, depth, 1.0)
    y = cam[:, 1] / np.where(valid, depth, 1.0)
    # ndc [-1,1] -> pixel [0,res]. mesh ~ unit radius -> x,y ~ [-0.5,0.5]/depth.
    # Scale to fill: multiply by res*0.5 * scale.
    scale = res * 0.5 * 1.4
    sx = res * 0.5 + x * scale
    sy = res * 0.5 - y * scale  # flip y
    return np.stack([sx, sy], axis=1), depth, valid


def rasterize_views(
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    azims: list[float],
    elevs: list[float],
    res: int = 256,
) -> dict:
    # Returns dict with keys: positions (N,res,res,3), normals_map (N,res,res,3),
    # mask (N,res,res), tri_id (N,res,res) int32 (-1 = empty), bary (N,res,res,3),
    # view_eye (N,3), view_dir (N,3).
    N = len(azims)
    V = verts.shape[0]
    positions = np.zeros((N, res, res, 3), dtype=np.float32)
    normals_map = np.zeros((N, res, res, 3), dtype=np.float32)
    mask = np.zeros((N, res, res), dtype=bool)
    tri_id = np.full((N, res, res), -1, dtype=np.int32)
    bary = np.zeros((N, res, res, 3), dtype=np.float32)
    view_eye = np.zeros((N, 3), dtype=np.float32)
    view_dir = np.zeros((N, 3), dtype=np.float32)

    for vi, (az, el) in enumerate(zip(azims, elevs)):
        view, eye = _camera(az, el)
        view_eye[vi] = eye
        view_dir[vi] = -eye / np.linalg.norm(eye)
        screen, depth, valid = _project(verts, view, res)  # (V,2),(V,),(V,)
        # per-triangle rasterization
        zbuf = np.full((res, res), np.inf, dtype=np.float32)
        for fi in range(faces.shape[0]):
            i0, i1, i2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            if not (valid[i0] and valid[i1] and valid[i2]):
                continue
            s0, s1, s2 = screen[i0], screen[i1], screen[i2]
            # triangle bounding box
            xs = np.array([s0[0], s1[0], s2[0]])
            ys = np.array([s0[1], s1[1], s2[1]])
            x0 = max(0, int(np.floor(xs.min())))
            x1 = min(res - 1, int(np.ceil(xs.max())))
            y0 = max(0, int(np.floor(ys.min())))
            y1 = min(res - 1, int(np.ceil(ys.max())))
            if x1 < x0 or y1 < y0:
                continue
            # barycentric via edge functions
            px = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
            py = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
            gx, gy = np.meshgrid(px, py, indexing="xy")  # (h,w)
            # edge function area
            area = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s2[0] - s0[0]) * (s1[1] - s0[1])
            if abs(area) < 1e-6:
                continue
            w0 = ((s1[0] - gx) * (s2[1] - gy) - (s2[0] - gx) * (s1[1] - gy)) / area
            w1 = ((s2[0] - gx) * (s0[1] - gy) - (s0[0] - gx) * (s2[1] - gy)) / area
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            # depth at pixels (interp)
            d = w0 * depth[i0] + w1 * depth[i1] + w2 * depth[i2]
            sub_z = zbuf[y0 : y1 + 1, x0 : x1 + 1]
            write = inside & (d < sub_z)
            if not write.any():
                continue
            zbuf[y0 : y1 + 1, x0 : x1 + 1] = np.where(write, d, sub_z)
            mask[vi, y0 : y1 + 1, x0 : x1 + 1] = (
                mask[vi, y0 : y1 + 1, x0 : x1 + 1] | write
            )
            tri_id[vi, y0 : y1 + 1, x0 : x1 + 1] = np.where(
                write, fi, tri_id[vi, y0 : y1 + 1, x0 : x1 + 1]
            )
            # interpolated position
            p0, p1, p2 = verts[i0], verts[i1], verts[i2]
            pos = w0[..., None] * p0 + w1[..., None] * p1 + w2[..., None] * p2
            positions[vi, y0 : y1 + 1, x0 : x1 + 1] = np.where(
                write[..., None], pos, positions[vi, y0 : y1 + 1, x0 : x1 + 1]
            )
            n0, n1, n2 = normals[i0], normals[i1], normals[i2]
            nrm = w0[..., None] * n0 + w1[..., None] * n1 + w2[..., None] * n2
            normals_map[vi, y0 : y1 + 1, x0 : x1 + 1] = np.where(
                write[..., None], nrm, normals_map[vi, y0 : y1 + 1, x0 : x1 + 1]
            )
            bary[vi, y0 : y1 + 1, x0 : x1 + 1] = np.where(
                write[..., None],
                np.stack([w0, w1, w2], axis=-1),
                bary[vi, y0 : y1 + 1, x0 : x1 + 1],
            )
        logger.info(
            "rasterize view %d/%d az=%.0f el=%.0f cov=%.3f",
            vi + 1,
            N,
            az,
            el,
            mask[vi].mean(),
        )
    return {
        "positions": positions,
        "normals": normals_map,
        "mask": mask,
        "tri_id": tri_id,
        "bary": bary,
        "view_eye": view_eye,
        "view_dir": view_dir,
    }


def uv_unwrap(
    verts: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # xatlas parameterize. Returns new_verts (seam-split, positions), new_faces
    # (indices into new_verts), uv (V',2), vremap (V',) -> orig vertex index.
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(verts.astype(np.float32), faces.astype(np.int32))
    atlas.generate()
    vremap, new_faces, uv = atlas.get_mesh(0)
    vremap = np.asarray(vremap).astype(np.int32)
    new_faces = np.asarray(new_faces).astype(np.int32)
    uv = np.asarray(uv).astype(np.float32)
    new_verts = verts[vremap].copy()
    logger.info(
        "uv_unwrap: %d verts -> %d (split), %d tris, uv %s",
        len(verts),
        len(new_verts),
        len(new_faces),
        uv.shape,
    )
    return new_verts, new_faces, uv, vremap


def _vertex_colors(
    verts: np.ndarray,
    normals: np.ndarray,
    view_textures: np.ndarray,
    raster: dict,
) -> np.ndarray:
    # For each orig vertex, project into every view's screen, pick the best
    # front-facing visible view, sample view_texture -> per-vertex color (V,3).
    N, vres, _, _ = view_textures.shape
    V = verts.shape[0]
    best_score = np.full(V, -1.0, dtype=np.float32)
    best_color = np.zeros((V, 3), dtype=np.float32)
    for vi in range(N):
        eye = raster["view_eye"][vi]
        fwd = -eye / np.linalg.norm(eye)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.stack([right, up, -fwd], axis=0)
        view = np.eye(4)
        view[:3, :3] = R
        view[:3, 3] = -R @ eye
        screen, depth, valid = _project(verts, view, vres)
        sx = np.clip(screen[:, 0].astype(np.int32), 0, vres - 1)
        sy = np.clip(screen[:, 1].astype(np.int32), 0, vres - 1)
        visible = valid & raster["mask"][vi, sy, sx]
        to_cam = eye[None, :] - verts
        to_cam = to_cam / (np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-6)
        score = (normals * to_cam).sum(axis=1)
        score = np.where(visible & (score > 0.05), score, -1.0)
        color = view_textures[vi, sy, sx]
        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_color = np.where(better[:, None], color, best_color)
    logger.info(
        "_vertex_colors: %d verts, covered=%.3f", V, float((best_score > 0).mean())
    )
    return best_color


def bake_to_atlas(
    new_verts: np.ndarray,
    new_faces: np.ndarray,
    uv: np.ndarray,
    vremap: np.ndarray,
    orig_verts: np.ndarray,
    orig_normals: np.ndarray,
    view_textures: np.ndarray,
    raster: dict,
    atlas_res: int = 512,
) -> np.ndarray:
    # Vertex-color bake: per orig vertex pick best view -> color; map to
    # seam-split verts via vremap; rasterize each atlas triangle in UV space
    # and barycentric-interpolate the 3 vertex colors.
    vcolors = _vertex_colors(orig_verts, orig_normals, view_textures, raster)
    new_colors = vcolors[vremap]

    atlas = np.zeros((atlas_res, atlas_res, 3), dtype=np.float32)
    coverage = np.zeros((atlas_res, atlas_res), dtype=np.float32)
    for fi in range(new_faces.shape[0]):
        i0, i1, i2 = int(new_faces[fi, 0]), int(new_faces[fi, 1]), int(new_faces[fi, 2])
        p0 = uv[i0] * atlas_res
        p1 = uv[i1] * atlas_res
        p2 = uv[i2] * atlas_res
        xs = np.array([p0[0], p1[0], p2[0]])
        ys = np.array([p0[1], p1[1], p2[1]])
        x0 = max(0, int(np.floor(xs.min()) - 1))
        x1 = min(atlas_res - 1, int(np.ceil(xs.max()) + 1))
        y0 = max(0, int(np.floor(ys.min()) - 1))
        y1 = min(atlas_res - 1, int(np.ceil(ys.max()) + 1))
        if x1 < x0 or y1 < y0:
            continue
        px = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
        py = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(px, py, indexing="xy")
        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        if abs(area) < 1e-6:
            continue
        w0 = ((p1[0] - gx) * (p2[1] - gy) - (p2[0] - gx) * (p1[1] - gy)) / area
        w1 = ((p2[0] - gx) * (p0[1] - gy) - (p0[0] - gx) * (p2[1] - gy)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        c0, c1, c2 = new_colors[i0], new_colors[i1], new_colors[i2]
        col = w0[..., None] * c0 + w1[..., None] * c1 + w2[..., None] * c2
        atlas[y0 : y1 + 1, x0 : x1 + 1] = np.where(
            inside[..., None], col, atlas[y0 : y1 + 1, x0 : x1 + 1]
        )
        coverage[y0 : y1 + 1, x0 : x1 + 1] = np.where(
            inside, 1.0, coverage[y0 : y1 + 1, x0 : x1 + 1]
        )
    logger.info(
        "bake_to_atlas: %dx%d coverage=%.3f",
        atlas_res,
        atlas_res,
        float(coverage.mean()),
    )
    return atlas


def export_glb(
    verts: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    texture: np.ndarray,  # (H,W,3) in [0,1] or [-1,1]
    path: str,
    texture_in_unit: bool = False,
) -> str:
    # trimesh Trimesh with UV + texture -> .glb.
    import trimesh

    tex = np.asarray(texture)
    if not texture_in_unit:
        # map [-1,1] -> [0,255]
        tex = ((tex + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    else:
        tex = (tex.clip(0, 1) * 255).astype(np.uint8)
    # PIL image
    from PIL import Image

    img = Image.fromarray(tex)
    material = trimesh.visual.texture.SimpleMaterial(image=img)
    visual = trimesh.visual.texture.TextureVisuals(
        uv=uv.astype(np.float32), image=img, material=material
    )
    mesh = trimesh.Trimesh(
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        visual=visual,
        process=False,
    )
    mesh.export(path)
    logger.info("export_glb: %s (%d verts, %d tris)", path, len(verts), len(faces))
    return path
