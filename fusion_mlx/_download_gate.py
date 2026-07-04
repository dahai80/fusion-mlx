# SPDX-License-Identifier: Apache-2.0
"""Download gate stub — auto-pull confirmation skipped in this build."""

import logging

logger = logging.getLogger(__name__)


def confirm_or_abort(model_name: str, estimated_bytes: int | None = None) -> None:
    pass


def estimate_repo_size_bytes(model_name: str) -> int | None:
    return None


def is_repo_cached(model_name: str) -> bool:
    return False
