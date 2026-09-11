# SPDX-License-Identifier: Apache-2.0
"""D3.2/N2: --beginner preset CLI wiring tests.

Pure argparse — no model load, no server boot. Monkeypatches serve_command
to capture the parsed args, then verifies the flag/choices compose correctly.
"""

import sys


def _capture_serve_args(monkeypatch, argv):
    captured = {}

    def _fake_serve_command(args):
        captured["args"] = args
        raise SystemExit(0)

    import fusion_mlx.cli_serve as cli_serve

    monkeypatch.setattr(cli_serve, "serve_command", _fake_serve_command)
    # Also patch the reference held in cli.py namespace if it imported the name.
    import fusion_mlx.cli as cli

    if hasattr(cli, "serve_command"):
        monkeypatch.setattr(cli, "serve_command", _fake_serve_command)

    old = sys.argv
    sys.argv = ["fusion-mlx"] + argv
    try:
        cli.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old
    return captured.get("args")


def test_beginner_flag_registered_and_default_false(monkeypatch):
    args = _capture_serve_args(monkeypatch, ["serve", "--model-dir", "/tmp/x"])
    assert args is not None, "serve_command was not invoked"
    assert args.beginner is False


def test_beginner_flag_parses_true(monkeypatch):
    args = _capture_serve_args(
        monkeypatch, ["serve", "--beginner", "--model-dir", "/tmp/x"]
    )
    assert args is not None
    assert args.beginner is True


def test_profile_choices_include_turbo(monkeypatch):
    args = _capture_serve_args(
        monkeypatch, ["serve", "--profile", "turbo", "--model-dir", "/tmp/x"]
    )
    assert args is not None
    assert args.profile == "turbo"


def test_profile_rejects_unknown_choice(monkeypatch):
    # argparse should exit 2 before serve_command is reached.
    captured = {}
    args = _capture_serve_args(
        monkeypatch, ["serve", "--profile", "bogus", "--model-dir", "/tmp/x"]
    )
    assert args is None, "invalid profile choice must be rejected by argparse"
