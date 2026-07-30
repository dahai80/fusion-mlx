# SPDX-License-Identifier: Apache-2.0
"""oQ backward-compatibility shim — delegates to fusion_mlx.oq package."""

from fusion_mlx.oq import (
    QuantPlan,
    universal_quant_predicate,
    resolve_output_name,
    validate_quantizable,
    make_predicate,
    estimate_bpw_and_size,
    estimate_memory,
    quantize_oq_streaming,
)

__all__ = [
    "QuantPlan",
    "universal_quant_predicate",
    "resolve_output_name",
    "validate_quantizable",
    "make_predicate",
    "estimate_bpw_and_size",
    "estimate_memory",
    "quantize_oq_streaming",
]
