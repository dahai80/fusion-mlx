# SPDX-License-Identifier: Apache-2.0
"""Per-request decode-time repair quota for the LSA selection path.

fusion-lsa issue #12: a small query-dependent page quota on top of the
static tier. The quota rides a ContextVar from the scheduler step (which
knows the decoding request) down to the LSA attention layer (which only
sees tensors) - the same plumbing pattern as the packed-attention mask
in fusion_mlx.training.packing. Default 0 = pure static tier, and the
N=0 decode path must stay bit-identical (regression guard).
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

logger = logging.getLogger(__name__)

_REPAIR_PAGES: ContextVar[int] = ContextVar("lsa_repair_pages", default=0)


def set_repair_pages(count: int):
    return _REPAIR_PAGES.set(max(0, int(count)))


def reset_repair_pages(token) -> None:
    _REPAIR_PAGES.reset(token)


def current_repair_pages() -> int:
    return _REPAIR_PAGES.get()
