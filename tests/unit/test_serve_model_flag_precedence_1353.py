# SPDX-License-Identifier: Apache-2.0
"""#1353 (H8): an explicit ``serve --model X`` must win over settings.json
``default_model`` instead of being overwritten and then rejected as
"--model and a positional <model> are mutually exclusive"."""

import sys


def _capture_serve_args(monkeypatch, argv):
    captured = {}

    def _fake_serve_command(args):
        captured["args"] = args
        raise SystemExit(0)

    import fusion_mlx.cli as cli
    import fusion_mlx.cli_serve as cli_serve

    monkeypatch.setattr(cli_serve, "serve_command", _fake_serve_command)
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


def test_explicit_model_flag_skips_settings_default_model(monkeypatch):
    import fusion_mlx.cli as cli

    monkeypatch.setattr(cli, "_settings_default_model", lambda: "gemma-4-4b-4bit")

    args = _capture_serve_args(monkeypatch, ["serve", "--model", "qwen3.5-9b-4bit"])

    assert args is not None, "serve_command was not invoked"
    # Pre-fix the zero-arg block wrote default_model into args.model, which
    # serve_command then saw alongside model_flag -> hard exit.
    assert args.model is None
    assert args.model_flag == "qwen3.5-9b-4bit"


def test_no_model_flag_still_uses_settings_default_model(monkeypatch, tmp_path):
    import fusion_mlx.cli as cli

    default_dir = tmp_path / "default-model"
    default_dir.mkdir()
    monkeypatch.setattr(cli, "_settings_default_model", lambda: str(default_dir))
    monkeypatch.setattr("fusion_mlx.model_aliases.resolve_model", lambda name: name)

    args = _capture_serve_args(monkeypatch, ["serve"])

    assert args is not None, "serve_command was not invoked"
    assert args.model == str(default_dir)
