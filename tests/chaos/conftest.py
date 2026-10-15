# SPDX-License-Identifier: Apache-2.0
"""Chaos fault-injection suite (#1195, PRD opti-1014 §6.2).

Per-scenario contract (issue DoD):
- detection time < 60s (watchdog/metric fires within its configured window)
- recovery time < 120s (service returns to admitting/serving work)
- zero silent request loss (every request served or explicitly errored)

Tests inject fault SHAPES at the Python boundary where each watchdog
observes them (real GPU/Metal hangs cannot be injected from pytest without
risking the runner itself — see docs/chaos-suite.md for the mapping and
the H1 rationale). Windows are scaled down via the same env knobs prod
uses, so the measured detection bound maps 1:1 to production config.
"""

import pytest

DETECTION_BOUND_S = 60.0
RECOVERY_BOUND_S = 120.0


@pytest.fixture(autouse=True)
def _chaos_state_reset(monkeypatch):
    # Every scenario starts from a clean, deterministic watchdog state and
    # leaves nothing behind for the next test (singletons + env knobs).
    monkeypatch.delenv("FUSION_MLX_LOCK_DEBUG", raising=False)
    monkeypatch.delenv("FUSION_MLX_LOCK_DEBUG_STRICT", raising=False)
    monkeypatch.delenv("FUSION_MLX_FD_WARN_RATIO", raising=False)
    monkeypatch.delenv("FUSION_MLX_FD_REJECT_RATIO", raising=False)
    monkeypatch.delenv("FUSION_MLX_HEAVY_ADMISSION", raising=False)
    monkeypatch.delenv("FUSION_MLX_HEAVY_ADMISSION_LINE", raising=False)
    yield
    from fusion_mlx.fd_watchdog import reset_fd_watchdog

    reset_fd_watchdog()
    from fusion_mlx.pool.lock_order import reset_lock_order_tracker

    reset_lock_order_tracker()
    from fusion_mlx.executor_watchdog import reset_executor_watchdogs

    reset_executor_watchdogs()
    from fusion_mlx.admission import reset_heavy_admission_gate

    reset_heavy_admission_gate()
    from fusion_mlx import engine_core as ec

    ec.reset_llm_executor_poison()
    ec._media_defer_start = 0.0
    ec._media_defer_logged = False
    ec._llm_step_deadline = 0.0
    ec._llm_step_window = 0.0
