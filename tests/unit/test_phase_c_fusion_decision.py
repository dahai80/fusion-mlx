# SPDX-License-Identifier: Apache-2.0
# O5.2/O5.3 tests: dequant fusion registry + runtime fusion decision.

import pytest


def test_quant_format_registry():
    from fusion_mlx.custom_kernels.phase_c import QUANT_FORMAT_REGISTRY

    assert set(QUANT_FORMAT_REGISTRY.keys()) == {"q4", "q6", "q8", "nvfp4"}
    assert QUANT_FORMAT_REGISTRY["q4"]["bits"] == 4
    assert QUANT_FORMAT_REGISTRY["q6"]["bits"] == 6
    assert QUANT_FORMAT_REGISTRY["q8"]["bits"] == 8
    assert QUANT_FORMAT_REGISTRY["nvfp4"]["cls"] == "NVFP4FusedLinear"


def test_create_fused_linear_unknown_format():
    from fusion_mlx.custom_kernels.phase_c import create_fused_linear

    with pytest.raises(ValueError, match="unknown quant format"):
        create_fused_linear(None, fmt="bogus")


def test_w4a8_linear_bits_validation():
    from fusion_mlx.custom_kernels.phase_c.w4a8_kernel import W4A8Linear

    with pytest.raises(ValueError, match="bits=3 not in"):
        W4A8Linear(8, 8, bits=3)


def test_decide_fusion_empty():
    from fusion_mlx.custom_kernels.phase_c import decide_fusion

    plan = decide_fusion([])
    assert plan.pattern_name is None
    assert "empty" in plan.reason


def test_decide_fusion_w4a8():
    from fusion_mlx.custom_kernels.phase_c import OpDescriptor, decide_fusion

    ops = [
        OpDescriptor("quantize_act", {"act_quant": "int8"}),
        OpDescriptor("matmul", {"weight_quant": "q4", "act_quant": "int8"}),
    ]
    plan = decide_fusion(ops)
    assert plan.pattern_name == "w4a8_fused_matmul"
    assert plan.ops_consumed == 2


def test_decide_fusion_gdn_non_diagonal():
    from fusion_mlx.custom_kernels.phase_c import OpDescriptor, decide_fusion

    ops = [
        OpDescriptor("square"),
        OpDescriptor("matmul", {"diagonal": False}),
        OpDescriptor("add"),
        OpDescriptor("rsqrt"),
        OpDescriptor("div"),
    ]
    plan = decide_fusion(ops)
    assert plan.pattern_name == "fused_gdn"


def test_decide_fusion_gdn_diagonal_skip():
    from fusion_mlx.custom_kernels.phase_c import OpDescriptor, decide_fusion

    ops = [
        OpDescriptor("square"),
        OpDescriptor("matmul", {"diagonal": True}),
        OpDescriptor("add"),
        OpDescriptor("rsqrt"),
        OpDescriptor("div"),
    ]
    plan = decide_fusion(ops)
    # diagonal GDN has no matmul to fuse -> no match
    assert plan.pattern_name is None


def test_decide_fusion_moe_ffn():
    from fusion_mlx.custom_kernels.phase_c import OpDescriptor, decide_fusion

    ops = [
        OpDescriptor("matmul", {"role": "gate"}),
        OpDescriptor("silu"),
        OpDescriptor("matmul", {"role": "up"}),
        OpDescriptor("matmul", {"role": "down"}),
    ]
    plan = decide_fusion(ops)
    # native not built -> falls through (native_required)
    # may match name but fused_callable None, OR fall through entirely
    assert plan.pattern_name in ("glm_moe_ffn_fused", None)


def test_decide_fusion_no_match():
    from fusion_mlx.custom_kernels.phase_c import OpDescriptor, decide_fusion

    ops = [OpDescriptor("rms_norm"), OpDescriptor("softmax")]
    plan = decide_fusion(ops)
    assert plan.pattern_name is None
    assert "no pattern" in plan.reason


def test_registered_patterns():
    from fusion_mlx.custom_kernels.phase_c import registered_patterns

    names = registered_patterns()
    assert "fused_gdn" in names
    assert "w4a8_fused_matmul" in names
    assert "glm_moe_ffn_fused" in names


def test_nvfp4_fused_linear_import():
    from fusion_mlx.custom_kernels.phase_c import NVFP4FusedLinear

    assert NVFP4FusedLinear is not None
