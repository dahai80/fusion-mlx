# SPDX-License-Identifier: Apache-2.0
"""#912: Metal Indirect Command Buffer (ICB) batched-encode tests."""

import mlx.core as mx
import pytest

from fusion_mlx.metal.icb import (
    _MAX_ICB_STAGES,
    IndirectCommandBuffer,
    UNetBlockParams,
    make_multi_stage_icb,
    make_single_stage_icb,
)


def test_unet_block_params_defaults():
    p = UNetBlockParams(name="stage0")
    assert p.name == "stage0"
    assert p.weight is None
    assert p.bias is None
    assert p.extra == {}


def test_single_stage_icb_creation():
    stage = UNetBlockParams(name="s0")
    icb = make_single_stage_icb(stage)
    assert isinstance(icb, IndirectCommandBuffer)
    assert len(icb.stages) == 1
    assert icb.segment_count() == 1


def test_multi_stage_under_limit():
    stages = [UNetBlockParams(name=f"s{i}") for i in range(8)]
    icb = make_multi_stage_icb(stages)
    assert len(icb.stages) == 8
    # 8 <= 16 → single segment.
    assert icb.segment_count() == 1


def test_multi_stage_segmentation():
    n = _MAX_ICB_STAGES * 2 + 3
    stages = [UNetBlockParams(name=f"s{i}") for i in range(n)]
    icb = make_multi_stage_icb(stages)
    expected = (n + _MAX_ICB_STAGES - 1) // _MAX_ICB_STAGES
    assert icb.segment_count() == expected


def test_encode_runs_all_fns():
    stages = [UNetBlockParams(name=f"s{i}") for i in range(5)]
    icb = IndirectCommandBuffer(stages, native=False)

    def make_fn(i):
        def fn(x):
            return x + i

        return fn

    fns = [make_fn(i) for i in range(5)]
    x = mx.array([0.0])
    out = icb.encode(fns, x)
    expected = 0 + 1 + 2 + 3 + 4
    assert float(out[0]) == float(expected)


def test_encode_mismatch_raises():
    stages = [UNetBlockParams(name="s0"), UNetBlockParams(name="s1")]
    icb = IndirectCommandBuffer(stages, native=False)
    with pytest.raises(AssertionError):
        icb.encode([lambda x: x], mx.array([0.0]))


def test_is_native_false_when_no_ext():
    icb = IndirectCommandBuffer([UNetBlockParams(name="s0")], native=False)
    assert icb.is_native is False


def test_is_native_true_when_forced():
    icb = IndirectCommandBuffer([UNetBlockParams(name="s0")], native=True)
    assert icb.is_native is True


def test_max_stages_constant():
    assert _MAX_ICB_STAGES == 16
