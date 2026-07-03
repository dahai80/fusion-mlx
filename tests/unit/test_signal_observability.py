# SPDX-License-Identifier: Apache-2.0
"""Tests for the signal-observability hook installed by the FastAPI lifespan.

Adapted from Rapid-MLX for fusion-mlx.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import textwrap
import threading

import pytest


def _read_ready_with_timeout(proc: subprocess.Popen, *, timeout: float = 10.0) -> str:
    fd = proc.stdout.fileno()
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        proc.kill()
        out_tail, err_tail = proc.communicate(timeout=5)
        raise AssertionError(
            f"subprocess did not emit READY within {timeout:.1f}s;"
            f" stdout-tail={out_tail!r}, stderr-tail={err_tail!r}"
        )
    line = proc.stdout.readline()
    if line == "":
        try:
            err_tail = proc.stderr.read() or ""
        except (OSError, ValueError):
            err_tail = "<stderr read failed>"
        raise AssertionError(
            "subprocess died before emitting READY;"
            f" returncode={proc.returncode!r}, stderr={err_tail!r}"
        )
    return line


def test_install_is_idempotent_and_saves_prior_handlers():
    from fusion_mlx import _signal_observability as so

    so._reset_for_tests()
    try:
        sentinel_calls: list[int] = []

        def _sentinel(signum, frame):
            sentinel_calls.append(signum)

        prior_usr1 = signal.signal(signal.SIGUSR1, _sentinel)
        try:
            ok = so.install_signal_observability(observed_signals=(signal.SIGUSR1,))
            assert ok is True
            handlers_after_first = dict(so._get_installed_handlers())
            assert signal.SIGUSR1 in handlers_after_first
            assert handlers_after_first[signal.SIGUSR1] is _sentinel

            ok2 = so.install_signal_observability(observed_signals=(signal.SIGUSR1,))
            assert ok2 is True
            handlers_after_second = dict(so._get_installed_handlers())
            assert handlers_after_second == handlers_after_first
        finally:
            so._reset_for_tests()
            signal.signal(signal.SIGUSR1, prior_usr1)
    finally:
        so._reset_for_tests()


def test_signal_chain_calls_prior_handler():
    from fusion_mlx import _signal_observability as so

    so._reset_for_tests()

    invoked: list[int] = []

    def _prior(signum, frame):
        invoked.append(signum)

    prior_usr1 = signal.signal(signal.SIGUSR1, _prior)
    try:
        so.install_signal_observability(observed_signals=(signal.SIGUSR1,))
        os.kill(os.getpid(), signal.SIGUSR1)
        assert invoked == [signal.SIGUSR1]
    finally:
        so._reset_for_tests()
        signal.signal(signal.SIGUSR1, prior_usr1)


def test_install_chains_to_sig_dfl_via_restore_and_raise():
    from fusion_mlx import _signal_observability as so

    so._reset_for_tests()

    program = textwrap.dedent(
        """
        import faulthandler, logging, os, signal, sys, time
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                            format="%(levelname)s %(name)s: %(message)s")
        assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL
        from fusion_mlx._signal_observability import install_signal_observability
        assert install_signal_observability(observed_signals=(signal.SIGUSR1,)) is True
        sys.stdout.write("READY\\n"); sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(2.0)
        os._exit(99)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _read_ready_with_timeout(proc)
        assert ready.strip() == "READY", ready
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert "received signal SIGUSR1" in stderr, stderr
    assert proc.returncode != 99, (
        f"chain swallowed the signal; stderr={stderr!r}"
    )


def test_install_returns_false_when_no_signals_could_be_installed():
    from fusion_mlx import _signal_observability as so

    so._reset_for_tests()
    try:
        result = so.install_signal_observability(observed_signals=())
        assert result is False
        assert so._get_installed_handlers() == {}
    finally:
        so._reset_for_tests()


def test_per_signal_latch_does_not_block_later_default_install():
    from fusion_mlx import _signal_observability as so

    so._reset_for_tests()

    def _sentinel1(signum, frame):
        pass

    def _sentinel2(signum, frame):
        pass

    prior_usr1 = signal.signal(signal.SIGUSR1, _sentinel1)
    prior_usr2 = signal.signal(signal.SIGUSR2, _sentinel2)
    try:
        ok1 = so.install_signal_observability(observed_signals=(signal.SIGUSR1,))
        assert ok1 is True
        assert signal.SIGUSR1 in so._get_installed_handlers()
        assert signal.SIGUSR2 not in so._get_installed_handlers()

        ok2 = so.install_signal_observability(
            observed_signals=(signal.SIGUSR1, signal.SIGUSR2)
        )
        assert ok2 is True
        handlers = so._get_installed_handlers()
        assert signal.SIGUSR1 in handlers
        assert signal.SIGUSR2 in handlers
    finally:
        so._reset_for_tests()
        signal.signal(signal.SIGUSR1, prior_usr1)
        signal.signal(signal.SIGUSR2, prior_usr2)


def test_install_skipped_off_main_thread():
    from fusion_mlx import _signal_observability as so

    result_box: list[bool] = []

    def _worker():
        result_box.append(so.install_signal_observability())

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert result_box == [False]


def test_faulthandler_is_enabled_after_install():
    import faulthandler

    from fusion_mlx import _signal_observability as so

    was_enabled = faulthandler.is_enabled()
    so._reset_for_tests()
    try:
        faulthandler.disable()
        assert not faulthandler.is_enabled()
        so.install_signal_observability(observed_signals=())
        assert faulthandler.is_enabled()
    finally:
        so._reset_for_tests()
        if not was_enabled:
            faulthandler.disable()


def test_subprocess_sigterm_emits_warning_and_stack_dump():
    program = textwrap.dedent(
        """
        import logging, os, signal, sys, time
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                            format="%(levelname)s %(name)s: %(message)s")
        def _exit_handler(signum, frame):
            sys.stderr.flush()
            os._exit(0)
        signal.signal(signal.SIGTERM, _exit_handler)

        from fusion_mlx._signal_observability import install_signal_observability
        assert install_signal_observability() is True

        sys.stdout.write("READY\\n")
        sys.stdout.flush()

        for _ in range(50):
            time.sleep(0.1)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready_line = _read_ready_with_timeout(proc)
        assert ready_line.strip() == "READY", ready_line
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert "received signal SIGTERM" in stderr, stderr
    assert "Thread" in stderr or "Current thread" in stderr, stderr


def test_subprocess_sighup_default_disposition_dumps_and_stays_alive():
    program = textwrap.dedent(
        """
        import logging, os, signal, sys, time
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                            format="%(levelname)s %(name)s: %(message)s")
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_DFL
        from fusion_mlx._signal_observability import install_signal_observability
        assert install_signal_observability() is True
        sys.stdout.write("READY\\n"); sys.stdout.flush()
        for _ in range(20):
            time.sleep(0.1)
        sys.stdout.write("ALIVE\\n"); sys.stdout.flush()
        os._exit(0)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready_line = _read_ready_with_timeout(proc)
        assert ready_line.strip() == "READY", ready_line
        proc.send_signal(signal.SIGHUP)
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert "received signal SIGHUP" in stderr, stderr
    assert "Thread" in stderr or "Current thread" in stderr, stderr
    assert proc.returncode == 0, (
        f"SIGHUP terminated the process; returncode={proc.returncode}"
    )
    assert "ALIVE" in stdout, f"SIGHUP did not allow subprocess to continue; stdout={stdout!r}"


def test_subprocess_sigterm_default_disposition_still_terminates():
    program = textwrap.dedent(
        """
        import logging, os, signal, sys, time
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                            format="%(levelname)s %(name)s: %(message)s")
        assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL
        from fusion_mlx._signal_observability import install_signal_observability
        assert install_signal_observability(observed_signals=(signal.SIGUSR1,)) is True
        sys.stdout.write("READY\\n"); sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(2.0)
        os._exit(99)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready_line = _read_ready_with_timeout(proc)
        assert ready_line.strip() == "READY", ready_line
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert "received signal SIGUSR1" in stderr, stderr
    assert proc.returncode != 99, (
        f"non-SIGHUP signal was swallowed; stderr={stderr!r}"
    )
