import mlx.core as mx
import pytest

from fusion_mlx.engines.video import VideoGenEngine


class _FakeBackend:
    def __init__(self):
        self._loaded = True
        self.calls = []

    async def denoise(
        self,
        latent,
        pos_embed,
        neg_embed,
        steps,
        cfg,
        seed,
        num_frames,
        control=None,
        inpaint_mask=None,
        init_latent=None,
    ):
        self.calls.append(
            {
                "control": control,
                "inpaint_mask": inpaint_mask,
                "init_latent": init_latent,
            }
        )
        return latent


@pytest.mark.asyncio
async def test_denoise_threads_inpaint_mask_and_init_latent():
    engine = VideoGenEngine.__new__(VideoGenEngine)
    engine._backend = _FakeBackend()
    engine._model_name = "fake"
    mask = mx.array([1.0])
    init = mx.array([2.0])
    out = await engine.denoise(
        mx.zeros((1,)),
        mx.zeros((1,)),
        None,
        1,
        1.0,
        0,
        1,
        inpaint_mask=mask,
        init_latent=init,
    )
    assert mx.array_equal(out, mx.zeros((1,))).item()
    assert engine._backend.calls[0]["inpaint_mask"] is mask
    assert engine._backend.calls[0]["init_latent"] is init


@pytest.mark.asyncio
async def test_denoise_defaults_inpaint_none_backcompat():
    engine = VideoGenEngine.__new__(VideoGenEngine)
    engine._backend = _FakeBackend()
    engine._model_name = "fake"
    await engine.denoise(mx.zeros((1,)), mx.zeros((1,)), None, 1, 1.0, 0, 1)
    assert engine._backend.calls[0]["inpaint_mask"] is None
    assert engine._backend.calls[0]["init_latent"] is None
