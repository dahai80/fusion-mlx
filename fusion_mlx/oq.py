# SPDX-License-Identifier: Apache-2.0
"""oQ backward-compatibility shim — delegates to fusion_mlx.oq package."""

from fusion_mlx.oq import (
    _LEVEL_BITS,
    QuantPlan,
    estimate_bpw_and_size,
    estimate_memory,
    make_predicate,
    quantize_oq_streaming,
    resolve_output_name,
    universal_quant_predicate,
    validate_quantizable,
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
    "_LEVEL_BITS",
]
