import mlx.core as mx
import pytest

from fusion_mlx.engines.video_backends._inpaint import apply_inpaint_mask


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
