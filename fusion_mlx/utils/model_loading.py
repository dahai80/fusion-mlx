# SPDX-License-Identifier: Apache-2.0
"""Model loading utilities."""

from typing import Any


def materialize_lazy_state(state: dict[str, Any]) -> dict[str, Any]:
    """Force-evaluate any lazy tensors in the model state dict."""
    result = {}
    for key, value in state.items():
        if hasattr(value, "materialize"):
            value.materialize()
        result[key] = value
    return result
