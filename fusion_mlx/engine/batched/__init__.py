# SPDX-License-Identifier: Apache-2.0
"""MTP dispatch shim package (was the duplicate BatchedEngine, #422).

The standalone ``BatchedEngine`` class that previously lived here was a
duplicate of the production implementation in
``fusion_mlx/engines/batched.py`` and had zero production importers
(server.py / engine_pool.py / engines/__init__.py all use
``engines.batched.BatchedEngine``). It was removed as dead code.

What remains is the live ``_mtp_dispatch`` submodule, which the
production engine imports
(``engines/batched.py: from ..engine.batched._mtp_dispatch import
_apply_mtp_dispatch``) and re-exports here so legacy import paths
(``from fusion_mlx.engine.batched import _DISPATCH_ATTACHED``) keep
working.
"""

from ._mtp_dispatch import (
    _DISPATCH_ATTACHED,
    _DISPATCH_NO_INJECT,
    _DISPATCH_REJECTED,
    _DISPATCH_UNRESOLVED,
    _apply_mtp_dispatch,
    _decide_mtp_dispatch_action,
    _get_mtp_dispatch_timeout_sec,
    _log_mtp_dispatch_timeout,
    _resolve_hf_model_type,
    _run_dispatch_mtp_inject,
)

__all__ = [
    "_DISPATCH_ATTACHED",
    "_DISPATCH_NO_INJECT",
    "_DISPATCH_REJECTED",
    "_DISPATCH_UNRESOLVED",
    "_apply_mtp_dispatch",
    "_decide_mtp_dispatch_action",
    "_get_mtp_dispatch_timeout_sec",
    "_log_mtp_dispatch_timeout",
    "_resolve_hf_model_type",
    "_run_dispatch_mtp_inject",
]
