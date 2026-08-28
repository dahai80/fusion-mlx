import asyncio
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.engines.image_gen import ImageGenEngine, _infer_flux2_config


def test_flux2_config_default_9b():
    assert _infer_flux2_config("flux-2") == "flux2_klein_9b"
    assert _infer_flux2_config("black-forest-labs/FLUX.2-klein-9B") == "flux2_klein_9b"
    assert _infer_flux2_config("flux2-klein-9b") == "flux2_klein_9b"


def test_flux2_config_4b():
    assert _infer_flux2_config("FLUX.2-klein-4B") == "flux2_klein_4b"
    assert _infer_flux2_config("flux2-klein-4b") == "flux2_klein_4b"


def test_flux2_config_9b_kv():
    assert _infer_flux2_config("flux2-klein-9b-kv") == "flux2_klein_9b_kv"
    assert _infer_flux2_config("FLUX.2-klein-9B-kv") == "flux2_klein_9b_kv"


def test_flux2_config_empty_or_none_defaults_to_9b():
    assert _infer_flux2_config("") == "flux2_klein_9b"
    assert _infer_flux2_config(None) == "flux2_klein_9b"


def test_flux2_config_base_4b():
    assert _infer_flux2_config("FLUX.2-klein-base-4B") == "flux2_klein_base_4b"
    assert _infer_flux2_config("flux2-klein-base-4b") == "flux2_klein_base_4b"


def test_flux2_config_base_9b():
    assert _infer_flux2_config("FLUX.2-klein-base-9B") == "flux2_klein_base_9b"
    assert _infer_flux2_config("flux2-klein-base-9b") == "flux2_klein_base_9b"


def test_flux2_config_quantized_4bit_suffix_not_misclassified():
    # Regression for #449: model id "flux2-klein-9b-4bit" contains the
    # substring "4b" (from "4bit"); it must NOT be classified as the 4b
    # config (heads=24), which mismatches the 9b weights (inner_dim=4096)
    # and breaks the transformer reshape.
    assert _infer_flux2_config("flux2-klein-9b-4bit") == "flux2_klein_9b"
    assert _infer_flux2_config("mlx-community/flux2-klein-9b-4bit") == "flux2_klein_9b"
    assert _infer_flux2_config("mlx-community/flux2-klein-4b-4bit") == "flux2_klein_4b"
    assert (
        _infer_flux2_config("mlx-community/flux2-klein-9b-kv-4bit")
        == "flux2_klein_9b_kv"
    )
    assert (
        _infer_flux2_config("mlx-community/FLUX.2-klein-base-4B-4bit")
        == "flux2_klein_base_4b"
    )
    assert (
        _infer_flux2_config("mlx-community/FLUX.2-klein-base-9B-4bit")
        == "flux2_klein_base_9b"
    )


def _make_engine_with_mock_vae_for_encode():
    eng = ImageGenEngine("fake-flux")
    vae = SimpleNamespace(
        encode=lambda image: mx.zeros((1, 32, 64, 64)),
        bn=SimpleNamespace(
            running_mean=mx.zeros((128,)),
            running_var=mx.ones((128,)),
            eps=1e-5,
        ),
    )
    flux = SimpleNamespace(vae=vae, tiling_config=None)
    eng._flux = flux
    eng._mflux_missing = False
    return eng, flux


class _InlineImageExecutor:
    # run_in_executor(ex, fn) runs fn on the calling (main) thread so mx.eval
    # finds the MLX stream the test process owns. The real image worker thread
    # has no stream without a loaded model — the executor path is exercised by
    # the Tier 4 real-model roundtrip, not these unit tests.
    def submit(self, fn, *args, **kwargs):
        import concurrent.futures

        fut = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:
            fut.set_exception(exc)
        return fut


def _patch_image_executor(monkeypatch):
    import fusion_mlx.engines.image_gen as ig_mod

    monkeypatch.setattr(
        ig_mod, "get_executor", lambda name: _InlineImageExecutor(), raising=False
    )


def test_image_encode_requires_started():
    eng = ImageGenEngine("x")
    with pytest.raises(RuntimeError, match="not started"):
        asyncio.run(eng.encode(mx.zeros((1, 1024, 1024, 3))))


def test_image_encode_requires_vae():
    eng, flux = _make_engine_with_mock_vae_for_encode()
    flux.vae = None
    with pytest.raises(RuntimeError, match="vae is unloaded"):
        asyncio.run(eng.encode(mx.zeros((1, 1024, 1024, 3))))


def test_image_encode_ndim_guard():
    eng, _ = _make_engine_with_mock_vae_for_encode()
    with pytest.raises(ValueError, match="encode expects"):
        asyncio.run(eng.encode(mx.zeros((1024, 1024, 3))))


def test_image_encode_divisibility_guard():
    eng, _ = _make_engine_with_mock_vae_for_encode()
    with pytest.raises(ValueError, match="divisible by 16"):
        asyncio.run(eng.encode(mx.zeros((1, 1023, 1024, 3))))


def test_image_encode_packed_output(monkeypatch):
    pytest.importorskip("mflux")
    eng, flux = _make_engine_with_mock_vae_for_encode()

    def fake_vaeutil_encode(vae, image, tiling_config=None):
        assert vae is flux.vae
        return mx.zeros((1, 32, 64, 64))

    monkeypatch.setattr(
        "mflux.models.common.vae.vae_util.VAEUtil.encode", fake_vaeutil_encode
    )
    _patch_image_executor(monkeypatch)
    out = asyncio.run(eng.encode(mx.zeros((1, 1024, 1024, 3))))
    assert out.ndim == 4
    assert out.shape == (1, 128, 32, 32)


def test_image_encode_bn_normalized(monkeypatch):
    pytest.importorskip("mflux")
    eng, flux = _make_engine_with_mock_vae_for_encode()
    flux.vae.bn.running_mean = mx.ones((128,)) * 5.0
    flux.vae.bn.running_var = mx.ones((128,)) * 4.0
    flux.vae.bn.eps = 0.0

    def fake_vaeutil_encode(vae, image, tiling_config=None):
        return mx.ones((1, 32, 64, 64)) * 3.0

    monkeypatch.setattr(
        "mflux.models.common.vae.vae_util.VAEUtil.encode", fake_vaeutil_encode
    )
    _patch_image_executor(monkeypatch)
    out = asyncio.run(eng.encode(mx.zeros((1, 1024, 1024, 3))))
    assert out.shape == (1, 128, 32, 32)
    arr = np.array(out)
    expected = (3.0 - 5.0) / mx.sqrt(mx.array(4.0)).item()
    assert np.allclose(
        arr, expected, atol=1e-4
    ), f"bn-norm not applied: got {arr.flatten()[:3]}, expected {expected}"
