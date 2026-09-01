import inspect

import pytest

from fusion_mlx.engines.video_backends import (
    cogvideox as cogvideox_mod,
)
from fusion_mlx.engines.video_backends import (
    cosmos as cosmos_mod,
)
from fusion_mlx.engines.video_backends import (
    hunyuanvideo as hunyuanvideo_mod,
)
from fusion_mlx.engines.video_backends import (
    ltx2 as ltx2_mod,
)
from fusion_mlx.engines.video_backends import (
    ltx2_5 as ltx2_5_mod,
)
from fusion_mlx.engines.video_backends import (
    ltx_video_legacy as ltx_video_legacy_mod,
)
from fusion_mlx.engines.video_backends import (
    minimax_h3 as minimax_h3_mod,
)
from fusion_mlx.engines.video_backends import (
    opensora as opensora_mod,
)
from fusion_mlx.engines.video_backends import (
    svd as svd_mod,
)
from fusion_mlx.engines.video_backends.base import VideoBackend

_BACKENDS = [
    ("CosmosBackend", cosmos_mod.CosmosBackend, "cosmos"),
    ("CogVideoBackend", cogvideox_mod.CogVideoBackend, "cogvideox"),
    ("HunyuanVideoBackend", hunyuanvideo_mod.HunyuanVideoBackend, "hunyuanvideo"),
    ("LegacyLTXBackend", ltx_video_legacy_mod.LegacyLTXBackend, "ltx_video_legacy"),
    ("LTX2Backend", ltx2_mod.LTX2Backend, "ltx2"),
    ("LTX2_5Backend", ltx2_5_mod.LTX2_5Backend, "ltx2_5"),
    ("MiniMaxH3Backend", minimax_h3_mod.MiniMaxH3Backend, "minimax_h3"),
    ("OpenSoraBackend", opensora_mod.OpenSoraBackend, "opensora"),
    ("SVDBackend", svd_mod.SVDBackend, "svd"),
]


@pytest.mark.parametrize(
    "cls_name,backend_cls,family", _BACKENDS, ids=[b[2] for b in _BACKENDS]
)
def test_encode_surface_overridden(cls_name, backend_cls, family):
    assert (
        backend_cls.encode is not VideoBackend.encode
    ), f"{family}: encode not overridden"
    assert (
        backend_cls.load_vae_encoder is not VideoBackend.load_vae_encoder
    ), f"{family}: load_vae_encoder not overridden"
    assert (
        backend_cls.unload_vae_encoder is not VideoBackend.unload_vae_encoder
    ), f"{family}: unload_vae_encoder not overridden"


@pytest.mark.parametrize(
    "cls_name,backend_cls,family", _BACKENDS, ids=[b[2] for b in _BACKENDS]
)
def test_encode_signature_accepts_pixels(cls_name, backend_cls, family):
    sig = inspect.signature(backend_cls.encode)
    params = list(sig.parameters)
    assert "pixels" in params, f"{family}: encode missing pixels param, got {params}"
    assert sig.parameters["pixels"].annotation is not inspect.Parameter.empty


@pytest.mark.parametrize(
    "cls_name,backend_cls,family", _BACKENDS, ids=[b[2] for b in _BACKENDS]
)
def test_encode_uses_numpy_bridge_and_executor(cls_name, backend_cls, family):
    src = inspect.getsource(backend_cls.encode)
    assert (
        "np.array" in src or "np.asarray" in src
    ), f"{family}: encode missing numpy-bridge (#630)"
    assert "run_in_executor" in src, f"{family}: encode missing executor (#630)"
    assert "mx.eval" in src, f"{family}: encode missing mx.eval on worker thread (#630)"
    assert "ndim" in src, f"{family}: encode missing ndim guard (wan2.py:692 precedent)"
