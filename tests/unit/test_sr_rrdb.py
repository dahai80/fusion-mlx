import numpy as np

from fusion_mlx.image.sr.config import RealESRGANConfig
from fusion_mlx.image.sr.rrdb import RRDBNet, leaky_relu, pixel_shuffle


def test_leaky_relu_pos_and_neg():
    import mlx.core as mx

    x = mx.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    out = leaky_relu(x, 0.2)
    out_np = np.array(out)
    expected = np.array([[-0.4, -0.2, 0.0, 1.0, 2.0]])
    assert np.allclose(out_np, expected, atol=1e-6)


def test_pixel_shuffle_x2_doubles_dims():
    import mlx.core as mx

    x = mx.arange(1 * 2 * 3 * 16, dtype=mx.float32).reshape(1, 2, 3, 16)
    out = pixel_shuffle(x, 2)
    assert out.shape == (1, 4, 6, 4), out.shape


def test_rrdbnet_forward_shape_scale4():
    import mlx.core as mx

    cfg = RealESRGANConfig()
    net = RRDBNet(cfg)
    x = mx.zeros((1, 8, 8, 3), dtype=mx.float32)
    out = net(x)
    assert out.shape == (1, 32, 32, 3), out.shape


def test_rrdbnet_forward_shape_scale2():
    import mlx.core as mx

    cfg = RealESRGANConfig(scale=2)
    net = RRDBNet(cfg)
    x = mx.zeros((1, 10, 12, 3), dtype=mx.float32)
    out = net(x)
    assert out.shape == (1, 20, 24, 3), out.shape
