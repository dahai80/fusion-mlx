# SPDX-License-Identifier: Apache-2.0
"""Tests for the signal-observability hook installed by the FastAPI lifespan.

Adapted from Rapid-MLX. The ``_signal_observability`` module does not exist
in fusion-mlx — all tests are skipped.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="rapid-mlx-only: fusion_mlx._signal_observability does not exist")


def test_install_is_idempotent_and_saves_prior_handlers():
    pass


def test_signal_chain_calls_prior_handler():
    pass


def test_install_chains_to_sig_dfl_via_restore_and_raise():
    pass


def test_install_returns_false_when_no_signals_could_be_installed():
    pass


def test_per_signal_latch_does_not_block_later_default_install():
    pass


def test_install_skipped_off_main_thread():
    pass


def test_faulthandler_is_enabled_after_install():
    pass


def test_subprocess_sigterm_emits_warning_and_stack_dump():
    pass


def test_subprocess_sighup_default_disposition_dumps_and_stays_alive():
    pass


def test_subprocess_sigterm_default_disposition_still_terminates():
    pass
