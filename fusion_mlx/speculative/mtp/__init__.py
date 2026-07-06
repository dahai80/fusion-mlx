# SPDX-License-Identifier: Apache-2.0
"""MTP (Multi-Token Prediction) speculative decoding.

Public surface:
- MTPAcceptCounter / get_global_counter — process-local stats
- MTPEligibility / detect_mtp_eligibility — config-based detection
- dispatch_mtp_inject / dispatch_mtp_validate — model-family router
- patch_arrays_cache_rollback_state — ArraysCache rollback support
- DepthController — adaptive draft-K controller (Ollama-style EV)
- mtp_generate_step — enhanced MTP generator with XTC/chain-of-K/EOS holdout
- inject_mtp_support / validate_mtp_support — per-family inject (deferred)
"""

from .accept_counter import MTPAcceptCounter, get_global_counter
from .cache_patch import patch_arrays_cache_rollback_state
from .detect import MTPEligibility, detect_mtp_eligibility
from .dispatch import dispatch_mtp_inject, dispatch_mtp_validate
from .draft_k_controller_v2 import DepthController, get_or_create_controller
from .generator import mtp_generate_step

__all__ = [
    "MTPAcceptCounter",
    "MTPEligibility",
    "DepthController",
    "detect_mtp_eligibility",
    "dispatch_mtp_inject",
    "dispatch_mtp_validate",
    "get_global_counter",
    "get_or_create_controller",
    "mtp_generate_step",
    "patch_arrays_cache_rollback_state",
]
