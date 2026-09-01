# tests/unit/test_sr_generate.py
import numpy as np


def test_super_resolve_small_image_no_tiling(tmp_path):
    # Image smaller than tile_size -> single forward, no split.
    from fusion_mlx.image.sr.generate import super_resolve
    imgs = np.random.rand(1, 16, 20, 3).astype(np.float32)
    out = super_resolve(imgs, model_path=None, scale=4, tile_size=512,
                        tile_overlap=64)
    assert out.shape == (1, 64, 80, 3), out.shape
    assert out.dtype == np.float32


def test_super_resolve_tile_equals_whole(tmp_path):
    # Tiled output must match whole-image output (the SR-tiling correctness gate).
    # Real-context padding feeds each tile with adjacent image pixels so the
    # conv's zero-pad border never appears at a tile seam, making tiled output
    # match the whole-image pass to floating-point precision.
    import mlx.core as mx

    from fusion_mlx.image.sr.config import RealESRGANConfig
    from fusion_mlx.image.sr.generate import _set_net_for_test, super_resolve
    from fusion_mlx.image.sr.rrdb import RRDBNet

    cfg = RealESRGANConfig(num_block=1, scale=2)
    net = RRDBNet(cfg)
    mx.eval(net.parameters())
    _set_net_for_test(net, scale=2)

    imgs = (np.random.rand(1, 40, 48, 3) * 0.5 + 0.2).astype(np.float32)
    whole = super_resolve(imgs, scale=2, tile_size=512, tile_overlap=64)
    tiled = super_resolve(imgs, scale=2, tile_size=16, tile_overlap=8)
    diff = np.abs(whole - tiled).max()
    assert diff < 1e-3, f"tile vs whole max diff {diff}"


def test_super_resolve_preserves_range(tmp_path):
    from fusion_mlx.image.sr.config import RealESRGANConfig
    from fusion_mlx.image.sr.generate import _set_net_for_test, super_resolve
    from fusion_mlx.image.sr.rrdb import RRDBNet
    cfg = RealESRGANConfig(num_block=1, scale=2)
    _set_net_for_test(RRDBNet(cfg), scale=2)
    imgs = np.zeros((1, 12, 12, 3), dtype=np.float32)
    out = super_resolve(imgs, scale=2, tile_size=512, tile_overlap=64)
    assert np.isfinite(out).all()
