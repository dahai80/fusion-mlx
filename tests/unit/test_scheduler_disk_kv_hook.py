# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the scheduler's disk-KV hook wiring.

Migrated from Rapid-MLX. The scheduler disk-KV hook
(``_maybe_disk_checkpoint`` / ``_safe_disk_checkpoint``) has NOT been
migrated to fusion-mlx's scheduler. The ``disk_kv_checkpoint`` runtime
module exists, but the scheduler hook methods that call it are absent.
All tests are skipped with a clear reason.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)

_SKIP_REASON = (
    "Scheduler disk-KV hook (_maybe_disk_checkpoint / _safe_disk_checkpoint) "
    "has not been migrated to fusion_mlx.scheduler. The runtime module "
    "fusion_mlx.runtime.disk_kv_checkpoint exists but the scheduler hook "
    "methods are absent. Re-enable when the hook is wired into the scheduler."
)


@pytest.mark.skip(reason=_SKIP_REASON)
def test_scheduler_hook_increments_writes_at_256_tok_boundary():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_scheduler_hook_no_op_when_interval_disabled():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_scheduler_hook_no_op_when_batch_generator_absent():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_safe_disk_checkpoint_records_silent_failure():
    pass
