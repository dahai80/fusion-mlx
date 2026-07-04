# SPDX-License-Identifier: Apache-2.0
import importlib
import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)
    import fusion_mlx.telemetry.state as state

    importlib.reload(state)
    return tmp_path


def _stub_tty(monkeypatch, *, in_=True, out=True):
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: in_)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: out)


def test_skips_when_consent_already_recorded(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import record_consent

    record_consent(False, rapid_mlx_version="0.6.33")
    _stub_tty(monkeypatch)
    maybe_prompt_for_consent("serve")
    assert capsys.readouterr().out == ""


def test_skips_when_env_var_set(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent

    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    _stub_tty(monkeypatch)
    maybe_prompt_for_consent("serve")
    assert capsys.readouterr().out == ""


def test_skips_when_cli_no_telemetry(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent

    _stub_tty(monkeypatch)
    maybe_prompt_for_consent("serve", cli_no_telemetry=True)
    assert capsys.readouterr().out == ""


def test_skips_when_stdin_not_tty(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent

    _stub_tty(monkeypatch, in_=False)
    maybe_prompt_for_consent("serve")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "subcommand",
    ["version", "help", "models", "ps", "info", "telemetry"],
)
def test_skips_for_non_interactive_subcommands(
    fake_home, monkeypatch, capsys, subcommand
):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent

    _stub_tty(monkeypatch)
    maybe_prompt_for_consent(subcommand)
    assert capsys.readouterr().out == ""


def test_skips_when_subcommand_none(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent

    _stub_tty(monkeypatch)
    maybe_prompt_for_consent(None)
    assert capsys.readouterr().out == ""


def test_yes_records_consent_true(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "y")
    assert maybe_prompt_for_consent("serve") is True

    state = get_consent_state()
    assert state is not None
    assert state.consent is True
    out = capsys.readouterr().out
    lower = out.lower()
    assert "thank you" in lower
    assert "rapid-mlx telemetry disable" in lower


def test_no_records_consent_false(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "n")
    assert maybe_prompt_for_consent("serve") is True

    state = get_consent_state()
    assert state is not None
    assert state.consent is False
    out = capsys.readouterr().out
    assert "stays off" in out.lower()


def test_cached_consent_skip_returns_false(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import record_consent

    _stub_tty(monkeypatch)

    record_consent(True, rapid_mlx_version="0.0.0+test")
    assert maybe_prompt_for_consent("serve") is False


def test_cli_no_telemetry_skip_returns_false(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)

    def _no_prompt():
        raise AssertionError("input() was called -- cli_no_telemetry guard is broken")

    monkeypatch.setattr("builtins.input", _no_prompt)
    assert get_consent_state() is None
    assert maybe_prompt_for_consent("serve", cli_no_telemetry=True) is False
    assert get_consent_state() is None


def test_non_interactive_subcommand_skip_returns_false(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)

    def _no_prompt():
        raise AssertionError(
            "input() was called -- non-interactive subcommand guard is broken"
        )

    monkeypatch.setattr("builtins.input", _no_prompt)
    assert get_consent_state() is None
    assert maybe_prompt_for_consent("version") is False
    assert get_consent_state() is None


def test_empty_answer_defaults_to_no(fake_home, monkeypatch):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "")
    maybe_prompt_for_consent("serve")

    state = get_consent_state()
    assert state is not None
    assert state.consent is False


def test_eof_during_prompt_does_not_record(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)

    def boom():
        raise EOFError

    monkeypatch.setattr("builtins.input", boom)
    maybe_prompt_for_consent("serve")
    assert get_consent_state() is None


def test_records_prompted_version_correctly(fake_home, monkeypatch):
    from fusion_mlx import __version__ as actual_version
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "y")
    maybe_prompt_for_consent("serve")

    state = get_consent_state()
    assert state is not None
    assert state.prompted_version == actual_version


def test_disclosure_is_ascii_encodable():
    from fusion_mlx.telemetry.consent import _DISCLOSURE

    rendered = _DISCLOSURE.format(env="RAPID_MLX_TELEMETRY", client_id_path="/tmp/x")
    rendered.encode("ascii")


def test_disclosure_unicodeerror_is_caught_safely(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry import consent as consent_mod
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)

    def _boom():
        raise UnicodeEncodeError("ascii", "x", 0, 1, "simulated stdout encoding")

    monkeypatch.setattr(consent_mod, "client_id_path", _boom)

    assert maybe_prompt_for_consent("serve") is False
    assert get_consent_state() is None


def test_post_record_oserror_still_reports_just_collected(
    fake_home, monkeypatch, capsys
):
    from fusion_mlx.telemetry import consent as consent_mod
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "n")

    def _explode():
        raise OSError("simulated SIGPIPE from closed parent pipe")

    monkeypatch.setattr(consent_mod, "consent_path", _explode)

    assert maybe_prompt_for_consent("serve") is True

    state = get_consent_state()
    assert state is not None
    assert state.consent is False


def test_pre_record_oserror_returns_false(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry import consent as consent_mod
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
    from fusion_mlx.telemetry.state import get_consent_state

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "y")

    def _explode():
        raise OSError("simulated stdout-closed during disclosure")

    monkeypatch.setattr(consent_mod, "client_id_path", _explode)

    assert maybe_prompt_for_consent("serve") is False
    assert get_consent_state() is None


def test_unwritable_home_does_not_crash_cli(fake_home, monkeypatch, capsys):
    from fusion_mlx.telemetry.consent import maybe_prompt_for_consent

    _stub_tty(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "y")

    def boom(*a, **kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(
        "fusion_mlx.telemetry.consent.record_consent",
        lambda *a, **kw: (_ for _ in ()).throw(boom()),
    )
    maybe_prompt_for_consent("serve")
