# SPDX-License-Identifier: Apache-2.0
# Regression tests for the configuration-slimming findings of
# audit/fusion-mlx-simple-conf-1019.md (L23): --enable-prefix-cache /
# --gc-control were store_true/default=True no-ops that could never turn the
# feature off, and --continuous-batching was never read at all.

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

import fusion_mlx.cli as cli_mod

_ALIAS = "qwen3.5-9b-4bit"


def _parse_serve(monkeypatch, extra):
    captured = {}

    def fake_serve(args):
        captured.update(vars(args))
        return 0

    argv = ["fusion-mlx", "serve", _ALIAS] + list(extra)
    with (
        patch.object(cli_mod, "serve_command", fake_serve),
        patch.object(sys, "argv", argv),
    ):
        try:
            cli_mod.main()
        except SystemExit as exc:
            pytest.fail(f"serve {extra} exited: {exc}")
    return captured


class TestServeBooleanOptionalFlags:
    def test_prefix_cache_defaults_on(self, monkeypatch):
        assert _parse_serve(monkeypatch, [])["prefix_cache"] is True

    def test_no_prefix_cache_turns_it_off(self, monkeypatch):
        assert _parse_serve(monkeypatch, ["--no-prefix-cache"])["prefix_cache"] is False

    def test_legacy_disable_alias(self, monkeypatch):
        assert (
            _parse_serve(monkeypatch, ["--disable-prefix-cache"])["prefix_cache"]
            is False
        )

    def test_legacy_enable_alias(self, monkeypatch):
        assert (
            _parse_serve(monkeypatch, ["--enable-prefix-cache"])["prefix_cache"] is True
        )

    def test_no_gc_control(self, monkeypatch):
        assert _parse_serve(monkeypatch, ["--no-gc-control"])["gc_control"] is False

    def test_gc_control_defaults_on(self, monkeypatch):
        assert _parse_serve(monkeypatch, [])["gc_control"] is True

    def test_no_continuous_batching_is_parsed(self, monkeypatch):
        # Accepted for compatibility; serve_command warns and ignores it.
        assert (
            _parse_serve(monkeypatch, ["--no-continuous-batching"])[
                "continuous_batching"
            ]
            is False
        )


class TestBenchBooleanOptionalFlags:
    def test_bench_prefix_cache_pair(self, monkeypatch):
        captured = {}

        def fake_bench(args):
            captured.update(vars(args))
            return 0

        for extra, expected in (
            ([], True),
            (["--no-prefix-cache"], False),
            (["--disable-prefix-cache"], False),
            (["--enable-prefix-cache"], True),
        ):
            captured.clear()
            argv = ["fusion-mlx", "bench", _ALIAS] + extra
            with (
                patch.object(cli_mod, "bench_command", fake_bench),
                patch.object(sys, "argv", argv),
            ):
                try:
                    cli_mod.main()
                except SystemExit as exc:
                    pytest.fail(f"bench {extra} exited: {exc}")
            assert captured["prefix_cache"] is expected, extra


class TestNoDeadFlagConsumers:
    def test_consumers_read_the_new_destination(self):
        from pathlib import Path

        root = Path(__file__).parent.parent.parent / "fusion_mlx"
        serve = (root / "cli_serve" / "serve_command.py").read_text()
        bench = (root / "cli_serve" / "bench_command.py").read_text()
        resolve = (root / "cli_serve" / "config_resolve.py").read_text()

        assert "args.enable_prefix_cache" not in serve
        assert "args.disable_prefix_cache" not in serve
        assert "enable_prefix_cache = args.prefix_cache" in serve
        assert "args.disable_prefix_cache" not in bench
        assert "enable_prefix_cache = args.prefix_cache" in bench
        assert "args.no_gc_control" not in resolve
        assert "gc_control = args.gc_control" in resolve


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
