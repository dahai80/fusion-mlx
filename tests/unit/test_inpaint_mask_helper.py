import mlx.core as mx
import pytest

from fusion_mlx.engines.video_backends._inpaint import (
    apply_inpaint_mask,
    patch_downsample_mask,
)


def test_apply_inpaint_mask_reactive_and_frozen():
    latents = mx.array([[1.0, 2.0], [3.0, 4.0]])
    init = mx.array([[10.0, 20.0], [30.0, 40.0]])
    mask = mx.array([[1.0, 0.0], [1.0, 0.0]])
    out = apply_inpaint_mask(latents, init, mask)
    expected = mx.array([[1.0, 20.0], [3.0, 40.0]])
    assert mx.allclose(out, expected).item()


def test_apply_inpaint_mask_none_passthrough():
    latents = mx.array([1.0, 2.0, 3.0])
    assert mx.array_equal(apply_inpaint_mask(latents, None, None), latents).item()


def test_apply_inpaint_mask_shape_mismatch_raises():
    latents = mx.zeros((1, 2, 2, 2))
    init = mx.zeros((1, 2, 3, 3))
    mask = mx.ones((1, 2, 2, 2))
    with pytest.raises(ValueError, match="init_latent shape"):
        apply_inpaint_mask(latents, init, mask)


def test_patch_downsample_mask_2x2_to_1x1():
    mask = mx.array([[1.0, 1.0], [0.0, 0.0]])
    out = patch_downsample_mask(
        mask,
        vae_stride=(4, 2, 2),
        patch_size=(1, 2, 2),
        t_latent=1,
        h_latent=1,
        w_latent=1,
    )
    assert out.shape == (1, 1, 1, 1)
    assert abs(float(out[0, 0, 0, 0]) - 0.5) < 1e-6


def test_patch_downsample_mask_broadcasts_temporal():
    mask = mx.ones((4, 4))
    out = patch_downsample_mask(
        mask,
        vae_stride=(4, 2, 2),
        patch_size=(1, 2, 2),
        t_latent=3,
        h_latent=2,
        w_latent=2,
    )
    assert out.shape == (1, 3, 2, 2)
    assert mx.allclose(out, mx.ones((1, 3, 2, 2))).item()
