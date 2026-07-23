#!/usr/bin/env python3
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("gen_fusion_glyph")

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "apps/fusion-mac/Resources/AppIcon.icon/Assets/fusion_glyph_1024.png"
REF = REPO / "apps/fusion-mac/Resources/AppIcon.icon/Assets/fusion-mlx_glyph_1024.png"
WHITE_THRESH = 230


def flood_fill_background(luma, thresh):
    h, w = luma.shape
    near_white = luma >= thresh
    bg = np.zeros_like(near_white)
    q = deque()
    for y in range(h):
        for x in (0, w - 1):
            if near_white[y, x] and not bg[y, x]:
                bg[y, x] = True
                q.append((y, x))
    for x in range(w):
        for y in (0, h - 1):
            if near_white[y, x] and not bg[y, x]:
                bg[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and near_white[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                q.append((ny, nx))
    return bg


def main():
    if not SRC.exists():
        log.error("source glyph missing: %s", SRC)
        sys.exit(1)
    im = Image.open(SRC).convert("RGB")
    a = np.array(im).astype(np.int16)
    luma = (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]).astype(np.int16)
    log.info("loaded %s size=%s mode-after-convert=RGB", SRC.name, im.size)

    bg = flood_fill_background(luma, WHITE_THRESH)
    bg_cov = bg.mean() * 100
    log.info("flood-fill background coverage=%.1f%%", bg_cov)

    # alpha: background -> 0; logo -> solid 255 with luminance-softened edges
    alpha = np.where(bg, 0, np.clip((255 - luma) * 3, 0, 255)).astype(np.uint8)
    logo_cov = (alpha > 0).mean() * 100
    log.info("logo silhouette coverage=%.1f%% alpha_extrema=%s", logo_cov, (int(alpha.min()), int(alpha.max())))

    ys, xs = np.where(alpha > 0)
    if len(ys) == 0:
        log.error("empty silhouette - aborting")
        sys.exit(1)
    log.info(
        "logo bbox x[%d-%d] y[%d-%d] centroid=(%d,%d) center=(512,512)",
        xs.min(), xs.max(), ys.min(), ys.max(),
        int(xs.mean()), int(ys.mean()),
    )

    # icon-composer fill=automatic recolors via alpha; RGB=black matches fusion-mlx ref
    rgba = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha
    out = Image.fromarray(rgba, "RGBA")

    backup = SRC.with_suffix(".png.orig")
    if not backup.exists():
        backup.write_bytes(SRC.read_bytes())
        log.info("backed up original -> %s", backup.name)
    out.save(SRC, format="PNG")
    log.info("wrote alpha glyph -> %s (mode=%s)", SRC.name, out.mode)

    ref = Image.open(REF)
    log.info("ref %s mode=%s alpha_cov=%.1f%%", REF.name, ref.mode, (np.array(ref)[:, :, 3] > 0).mean() * 100)


if __name__ == "__main__":
    main()
