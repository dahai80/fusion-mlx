# SPDX-License-Identifier: Apache-2.0
# #1236: TurboQuant v2 quantized-state pairs must survive an SSD round trip
# with their NamedTuple classes intact, or prefix-cache reconstruction
# (_concat_state type dispatch) raises TypeError and reuse never happens.

import mlx.core as mx
import pytest

from fusion_mlx.cache.paged_ssd_cache import PagedSSDCacheManager

pytest.importorskip("mlx.core")


def _mk_mgr(tmp_path):
    return PagedSSDCacheManager(
        cache_dir=tmp_path / "tq_ssd",
        max_size_bytes=64 * 1024 * 1024,
        hot_cache_only=True,
    )


def test_tq_mse_state_roundtrip(tmp_path):
    from mlx_vlm.turboquant import TurboQuantMSEState

    mgr = _mk_mgr(tmp_path)
    ks = TurboQuantMSEState(
        norms=mx.ones((4, 2), dtype=mx.float16),
        indices=mx.zeros((4, 2), dtype=mx.int32),
    )
    vs = TurboQuantMSEState(
        norms=mx.full((4, 2), 2.0, dtype=mx.float16),
        indices=mx.ones((4, 2), dtype=mx.int32),
    )
    cache_data = [
        ("__turboquant_v2__", (ks, vs)),
        (
            mx.zeros((4, 2, 2, 4), dtype=mx.float16),
            mx.ones((4, 2, 2, 4), dtype=mx.float16),
        ),
    ]
    assert mgr.save_block(b"tqmse1", cache_data, token_count=8, model_name="m")

    loaded = mgr.load_block(b"tqmse1")
    assert loaded is not None
    assert loaded[0][0] == "__turboquant_v2__"
    l_ks, l_vs = loaded[0][1]
    assert type(l_ks) is TurboQuantMSEState
    assert type(l_vs) is TurboQuantMSEState
    assert l_ks.norms.shape == (4, 2)
    assert float(mx.max(mx.abs(l_ks.norms - ks.norms)).item()) == 0.0
    assert float(mx.max(mx.abs(l_vs.norms - vs.norms)).item()) == 0.0
    assert type(loaded[1]) is tuple and len(loaded[1]) == 2
    assert loaded[1][0].shape == (4, 2, 2, 4)


def test_tq_split_state_roundtrip(tmp_path):
    from mlx_vlm.turboquant import TurboQuantMSEState, TurboQuantSplitState

    mgr = _mk_mgr(tmp_path)
    # SplitState nests two MSEState halves (low/high precision split).
    ks = TurboQuantSplitState(
        low=TurboQuantMSEState(
            norms=mx.full((4, 2), 0.5, dtype=mx.float16),
            indices=mx.zeros((4, 2), dtype=mx.int32),
        ),
        high=TurboQuantMSEState(
            norms=mx.full((4, 2), 1.5, dtype=mx.float16),
            indices=mx.ones((4, 2), dtype=mx.int32),
        ),
    )
    vs = TurboQuantMSEState(
        norms=mx.full((4, 2), 3.0, dtype=mx.float16),
        indices=mx.zeros((4, 2), dtype=mx.int32),
    )
    cache_data = [("__turboquant_v2__", (ks, vs))]
    assert mgr.save_block(b"tqsplit1", cache_data, token_count=8, model_name="m")

    loaded = mgr.load_block(b"tqsplit1")
    assert loaded is not None
    l_ks, l_vs = loaded[0][1]
    assert type(l_ks) is TurboQuantSplitState
    assert type(l_ks.low) is TurboQuantMSEState
    assert type(l_vs) is TurboQuantMSEState
    assert float(mx.max(mx.abs(l_ks.high.norms - ks.high.norms)).item()) == 0.0


def test_tq_polar_state_roundtrip(tmp_path):
    from mlx_vlm.turboquant import TurboQuantPolarState

    mgr = _mk_mgr(tmp_path)
    # PolarState.level_indices is a plain tuple of arrays inside the
    # NamedTuple — exercises the nested plain-tuple container path.
    ks = TurboQuantPolarState(
        radii=mx.full((4, 2), 1.0, dtype=mx.float16),
        level_indices=(
            mx.zeros((4, 2), dtype=mx.int32),
            mx.ones((4, 2), dtype=mx.int32),
        ),
    )
    vs = TurboQuantPolarState(
        radii=mx.full((4, 2), 2.0, dtype=mx.float16),
        level_indices=(mx.ones((4, 2), dtype=mx.int32),),
    )
    cache_data = [("__turboquant_v2__", (ks, vs))]
    assert mgr.save_block(b"tqpol1", cache_data, token_count=8, model_name="m")

    loaded = mgr.load_block(b"tqpol1")
    assert loaded is not None
    l_ks, l_vs = loaded[0][1]
    assert type(l_ks) is TurboQuantPolarState
    assert type(l_ks.level_indices) is tuple
    assert len(l_ks.level_indices) == 2
    assert len(l_vs.level_indices) == 1
    assert float(mx.max(mx.abs(l_ks.radii - ks.radii)).item()) == 0.0


def test_extract_tensor_bytes_rejects_marker_string():
    from fusion_mlx.cache.paged_ssd_cache import _extract_tensor_bytes

    with pytest.raises(TypeError):
        _extract_tensor_bytes("__turboquant_v2__")
