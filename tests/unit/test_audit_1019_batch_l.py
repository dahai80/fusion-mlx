# SPDX-License-Identifier: Apache-2.0
# Regression tests for the L-series of
# audit/fusion-mlx-codeing-audit-report-1019.md:
#   L1 racing poison swaps two executors, L4 silent empty stream, L21 silent
#   fail-unfinished, L22 unlocked media-defer state, L26 ps value flags,
#   L28 stop reports success on any OSError, L29 gated repo shown as 404,
#   L33 zero-division on tiny max_area.

from __future__ import annotations

import ast
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import fusion_mlx.engine_core as ec


def _bare_core():
    core = ec.EngineCore.__new__(ec.EngineCore)
    core._active_contexts = {}
    core._finished_at = {}
    core._engine_id = "test-engine"
    return core


class _Ctx:
    def __init__(self, put_raises=False):
        self.finished_event = threading.Event()
        self.puts = []
        self._put_raises = put_raises

    def put(self, _output):
        if self._put_raises:
            raise RuntimeError("collector closed")
        self.puts.append(_output)


class _Collector:
    def __init__(self, ctx):
        self._ctx = ctx

    def put(self, output):
        self._ctx.put(output)


class TestPoisonIsAtomic:
    # L1: the video/image check and set lived under different locks, so two
    # racing callers each installed a replacement executor.

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.setattr(ec, "_global_executors", {})
        monkeypatch.setattr(ec, "_video_executor_poisoned", False)
        monkeypatch.setattr(ec, "_image_executor_poisoned", False)
        monkeypatch.setattr(ec, "_audio_executor_poisoned", False)
        monkeypatch.setattr(ec, "_laya_executor_poisoned", False)
        monkeypatch.setattr(ec, "_llm_executor_poisoned", False)
        yield

    def _counting_executor(self, monkeypatch):
        created = []

        class _Exec:
            def __init__(self, **kwargs):
                created.append(kwargs.get("thread_name_prefix"))
                self.shutdown_called = False

            def shutdown(self, *a, **k):
                self.shutdown_called = True

        monkeypatch.setattr(ec.concurrent.futures, "ThreadPoolExecutor", _Exec)
        return created

    @pytest.mark.parametrize("pool_type", ["video", "image", "audio", "laya"])
    def test_concurrent_poison_swaps_once(self, monkeypatch, pool_type):
        created = self._counting_executor(monkeypatch)
        ec._global_executors[pool_type] = object()
        results = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            ec.poison_executor(pool_type)
            results.append(True)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5.0)

        assert len(results) == 4
        assert len(created) == 1, f"{pool_type}: {len(created)} replacements"
        assert isinstance(ec._global_executors[pool_type], object)

    def test_second_poison_is_a_noop(self, monkeypatch):
        created = self._counting_executor(monkeypatch)
        ec._global_executors["video"] = object()

        ec.poison_executor("video")
        first = ec._global_executors["video"]
        ec.poison_executor("video")

        assert len(created) == 1
        assert ec._global_executors["video"] is first


class TestStreamOutputsFailsVisibly:
    # L4: a missing context returned an empty generator — the client got a
    # 200 with no body and no error frame.

    @pytest.mark.asyncio
    async def test_missing_context_raises(self, caplog):
        core = _bare_core()
        with caplog.at_level(logging.ERROR, logger="fusion_mlx.engine_core"):
            with pytest.raises(RuntimeError, match="No context for request gone"):
                async for _ in core.stream_outputs("gone"):
                    pass

        assert any("no context for request gone" in r.message for r in caplog.records)


class TestFailUnfinishedIsNotSilent:
    # L21: a collector that refused the terminal put left the consumer
    # waiting on finished_event with no trace.

    def test_put_failure_logged_and_counted(self, caplog):
        core = _bare_core()
        good, bad = _Ctx(), _Ctx(put_raises=True)
        core._active_contexts = {
            "good": SimpleNamespace(
                finished_event=good.finished_event, collector=_Collector(good)
            ),
            "bad": SimpleNamespace(
                finished_event=bad.finished_event, collector=_Collector(bad)
            ),
        }

        with caplog.at_level(logging.ERROR, logger="fusion_mlx.engine_core"):
            n = core._fail_unfinished_contexts("engine died")

        assert n == 2
        assert len(good.puts) == 1
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("could not push terminal output to bad" in r.message for r in errors)
        assert any("1/2 request(s) got no terminal output" in r.message for r in errors)
        # Both events are still signalled so consumers cannot hang.
        assert good.finished_event.is_set()
        assert bad.finished_event.is_set()


class TestMediaDeferIsLocked:
    # L22: the watchdog thread read and armed the defer window without
    # _media_job_active_lock, racing set_media_job_active(False).

    @pytest.fixture(autouse=True)
    def _reset(self):
        ec.set_media_job_active(False)
        yield
        ec.set_media_job_active(False)

    def test_defer_window_arms_once(self):
        first = ec._media_defer_elapsed_locked(100.0)
        second = ec._media_defer_elapsed_locked(130.0)

        assert first == 0.0
        assert second == pytest.approx(30.0)

    def test_reset_restarts_the_window(self):
        ec._media_defer_elapsed_locked(100.0)
        ec.set_media_job_active(True)
        ec.set_media_job_active(False)

        assert ec._media_defer_elapsed_locked(500.0) == 0.0

    def test_log_slot_claimed_once(self):
        assert ec._media_defer_claim_log_slot() is True
        assert ec._media_defer_claim_log_slot() is False


class TestPsValueFlagsMatchServeParser:
    # L26: the hand-maintained whitelist drifted and printed flag values as
    # the model name. Derive the truth from the serve parser source.

    def test_constant_covers_every_value_taking_flag(self):
        from fusion_mlx.cli_commands import _SERVE_VALUE_FLAGS

        derive = Path(__file__).parent.parent.parent / "fusion_mlx" / "cli.py"
        tree = ast.parse(derive.read_text())
        derived = set()
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "serve_parser"
            ):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            action = kw.get("action")
            act = None
            if isinstance(action, ast.Constant):
                act = action.value
            elif isinstance(action, ast.Attribute):
                act = action.attr
            elif isinstance(action, ast.Name):
                act = action.id
            if act in (
                "store_true",
                "store_false",
                "count",
                "help",
                "version",
                "BooleanOptionalAction",
            ):
                continue
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.startswith("--")
                ):
                    derived.add(arg.value)

        assert derived, "AST extraction found no serve flags — parser moved"
        missing = derived - _SERVE_VALUE_FLAGS
        assert (
            not missing
        ), f"ps would print these flags' values as the model: {sorted(missing)}"
        assert set(_SERVE_VALUE_FLAGS) == derived


class TestLifecycleStopDoesNotLie:
    # L28: every OSError/ValueError was reported as "already stopped".

    def _args(self):
        return SimpleNamespace(command="stop", timeout=5.0, no_wait=True)

    def test_socket_gone_is_stopped(self, monkeypatch, capsys):
        import fusion_mlx.cli_lifecycle as lc

        monkeypatch.setattr(lc, "is_app_bundle", lambda: True)
        monkeypatch.setattr(
            lc,
            "_send_app_control",
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no socket")),
        )

        assert lc.lifecycle_command(self._args()) == 0
        assert "stopped" in capsys.readouterr().out

    def test_permission_error_is_a_failure(self, monkeypatch, capsys):
        import fusion_mlx.cli_lifecycle as lc

        monkeypatch.setattr(lc, "is_app_bundle", lambda: True)
        monkeypatch.setattr(
            lc,
            "_send_app_control",
            lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied")),
        )

        assert lc.lifecycle_command(self._args()) == 1
        assert "Failed to control" in capsys.readouterr().out


class TestGatedRepoIsNot404:
    # L29: GatedRepoError subclasses RepositoryNotFoundError, so a 401 gated
    # repo was reported as "not found".

    def test_gated_repo_message(self, monkeypatch):
        import huggingface_hub

        import fusion_mlx.cli_serve.model_download as md

        gated_cls = type("GatedRepoError", (OSError,), {})
        repo_cls = type("RepositoryNotFoundError", (OSError,), {})

        monkeypatch.setattr(md, "_check_disk_space", lambda *a, **k: None)
        monkeypatch.setattr(md, "_try_mirror_prefetch", lambda *a, **k: False)
        monkeypatch.setattr(
            huggingface_hub,
            "snapshot_download",
            lambda *a, **k: (_ for _ in ()).throw(
                gated_cls("401 Client Error: gated repo")
            ),
        )
        monkeypatch.setattr(huggingface_hub, "model_info", lambda *a, **k: None)

        with pytest.raises(RuntimeError, match="gated"):
            md._ensure_model_downloaded("org/gated-model-1019")

        # A genuine 404 still reports "not found".
        monkeypatch.setattr(
            huggingface_hub,
            "snapshot_download",
            lambda *a, **k: (_ for _ in ()).throw(repo_cls("404 Client Error")),
        )
        with pytest.raises(RuntimeError, match="not found"):
            md._ensure_model_downloaded("org/gated-model-1019")


class TestBestOutputSizeNoZeroDivision:
    # L33: an alignment step larger than the max_area side floored to 0 and
    # the following division blew up.

    def test_tiny_area_and_large_step(self):
        from fusion_mlx.video.wan2.generate import _best_output_size

        w, h = _best_output_size(8, 8, 16, 16, 100)
        assert w >= 16
        assert h >= 16

    def test_degenerate_inputs(self):
        from fusion_mlx.video.wan2.generate import _best_output_size

        assert _best_output_size(0, 0, 16, 16, 100) == (16, 16)
        assert _best_output_size(64, 64, 16, 16, 0) == (64, 64)


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
