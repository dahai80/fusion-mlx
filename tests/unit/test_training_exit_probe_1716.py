# SPDX-License-Identifier: Apache-2.0
"""#1716: training child SIGKILL forensics.

The parent previously only saw a bare exitcode; an external SIGKILL was
indistinguishable from any other unreported death. These tests cover the
child-side signal probe (SIGTERM/SIGQUIT self-report over the training
pipe), the exitcode decoder, and the enriched TrainingProcessKilled
message.
"""

import os
import signal
import textwrap
import time
from types import SimpleNamespace

import pytest

from fusion_mlx.training.proc import (
    TrainingProcessKilled,
    _exit_reason,
    run_training_process,
)

STUB_MODULE = """
import time


def run_slow(spec, report):
    report({"type": "train_loss", "step": 1, "total_steps": 1000})
    time.sleep(120)
    return {}
"""


@pytest.fixture
def stub_entrypoint(tmp_path, monkeypatch):
    stub_file = tmp_path / "fmlx_stub_1716.py"
    stub_file.write_text(textwrap.dedent(STUB_MODULE))
    monkeypatch.syspath_prepend(str(tmp_path))
    return "fmlx_stub_1716"


def _spec(entrypoint_fn: str, tmp_path) -> dict:
    return {
        "config": {},
        "model_path": "/nonexistent",
        "dataset": "/nonexistent",
        "adapter_path": str(tmp_path / "adapter_out"),
        "entrypoint": entrypoint_fn,
    }


class TestExitReasonDecoder:
    def test_negative_exitcode_decodes_signal_name(self):
        proc = SimpleNamespace(exitcode=-9)
        assert _exit_reason(proc) == "signal 9 (SIGKILL)"

    def test_negative_exitcode_unknown_signal(self):
        proc = SimpleNamespace(exitcode=-64)
        assert _exit_reason(proc) == "signal 64 (unknown)"

    def test_positive_exitcode(self):
        assert _exit_reason(SimpleNamespace(exitcode=1)) == "exit 1"

    def test_none_exitcode(self):
        assert _exit_reason(SimpleNamespace(exitcode=None)) == "still-running"


class TestSignalProbe:
    def test_sigterm_self_report_lands_in_killed_message(
        self, stub_entrypoint, tmp_path
    ):
        holder = {}

        def on_started(proc):
            holder["proc"] = proc

        # Wait for the child to be inside run_slow's sleep, then SIGTERM it.
        def on_event(event):
            time.sleep(0.2)
            if not holder.get("signaled"):
                holder["signaled"] = True
                os.kill(holder["proc"].pid, signal.SIGTERM)

        with pytest.raises(TrainingProcessKilled) as exc_info:
            run_training_process(
                _spec(f"{stub_entrypoint}:run_slow", tmp_path),
                on_event,
                0.0,
                on_started,
            )
        msg = str(exc_info.value)
        assert "last-reported-signal=SIGTERM" in msg
        assert "signal 15 (SIGTERM)" in msg

    def test_sigkill_reports_external_signal_name(self, stub_entrypoint, tmp_path):
        # SIGKILL is unhandleable — no self-report is possible, but the
        # killed message must still decode the signal for forensics.
        holder = {}

        def on_started(proc):
            holder["proc"] = proc

        def on_event(event):
            time.sleep(0.2)
            if not holder.get("killed"):
                holder["killed"] = True
                holder["proc"].kill()

        with pytest.raises(TrainingProcessKilled) as exc_info:
            run_training_process(
                _spec(f"{stub_entrypoint}:run_slow", tmp_path),
                on_event,
                0.0,
                on_started,
            )
        msg = str(exc_info.value)
        assert "reason=signal 9 (SIGKILL)" in msg
        assert "last-reported-signal" not in msg
