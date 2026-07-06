# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest


def test_scheduler_hook_increments_writes_at_256_tok_boundary(
    tmp_path,
) -> None:
    pytest.skip("Scheduler disk-KV hook not migrated to fusion-mlx")


def test_scheduler_hook_no_op_when_interval_disabled(tmp_path) -> None:
    pytest.skip("Scheduler disk-KV hook not migrated to fusion-mlx")


def test_scheduler_hook_no_op_when_batch_generator_absent(
    tmp_path,
) -> None:
    pytest.skip("Scheduler disk-KV hook not migrated to fusion-mlx")


def test_safe_disk_checkpoint_records_silent_failure(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    pytest.skip("Scheduler disk-KV hook not migrated to fusion-mlx")
