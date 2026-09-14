"""ps_command detects fusion-mlx servers whether or not setproctitle
has renamed the process. setproctitle collapses argv into one string
element (with trailing empties); ps_command must re-split via shlex and
still match the ``serve`` token."""

from __future__ import annotations


def test_merged_cmdline_detected(monkeypatch):
    """setproctitle-style merged single-string argv is detected."""
    import psutil

    import fusion_mlx.cli_commands as cc

    class _Proc:
        def __init__(self, pid, cmdline):
            self.info = {"pid": pid, "cmdline": cmdline, "create_time": 0}

    procs = [
        # Renamed: single collapsed string + trailing empties
        _Proc(
            111,
            [
                "fusion-mlx-server serve --model-dir /x --port 11434",
                "",
                "",
            ],
        ),
        # Original: token list
        _Proc(222, ["fusion-mlx", "serve", "--model-dir", "/y"]),
        # Unrelated: chat command (must NOT match)
        _Proc(333, ["fusion-mlx", "chat"]),
        # caffeinate wrapper (must NOT match)
        _Proc(
            444,
            ["/usr/bin/caffeinate", "-is", "fusion-mlx", "serve"],
        ),
    ]

    monkeypatch.setattr(psutil, "process_iter", lambda _attrs: procs)

    printed: list[str] = []

    def fake_print(*a, **k):
        printed.append(" ".join(str(x) for x in a))

    import builtins

    monkeypatch.setattr(builtins, "print", fake_print)
    try:
        import tabulate as _tab

        monkeypatch.setattr(
            _tab, "tabulate", lambda rows, **k: "\n".join(str(r) for r in rows)
        )
    except Exception:
        pass

    class _A:
        pass

    cc.ps_command(_A())
    blob = "\n".join(printed)
    assert "111" in blob, f"renamed server PID 111 not detected: {blob!r}"
    assert "222" in blob, f"original server PID 222 not detected: {blob!r}"
    assert "333" not in blob, "chat process should not match"
    assert "444" not in blob, "caffeinate wrapper should not match"
