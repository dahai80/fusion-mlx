from fusion_mlx.image.sr.config import RealESRGANConfig


def test_realesrgan_x4plus_defaults():
    cfg = RealESRGANConfig()
    assert cfg.num_in_ch == 3
    assert cfg.num_out_ch == 3
    assert cfg.scale == 4
    assert cfg.num_feat == 64
    assert cfg.num_block == 23
    assert cfg.num_grow_ch == 32
    assert cfg.res_scale == 1.0
