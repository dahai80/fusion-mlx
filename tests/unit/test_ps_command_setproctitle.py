"""ps_command detects fusion-mlx servers whether or not setproctitle
has renamed the process. setproctitle collapses argv into one string
element (with trailing empties) or, on current builds, a bare
"fusion-mlx-server" title; ps_command must re-split via shlex, match
the ``serve`` token, and enrich bare titles from server.json."""

from __future__ import annotations


def _run_scan(monkeypatch, procs, server_info):
    """Run ps_command over fake psutil processes + server.json content."""
    import psutil

    import fusion_mlx.cli_commands as cc

    monkeypatch.setattr(psutil, "process_iter", lambda _attrs: procs)

    # ps_command reads server.json via open() — patch builtins.open for
    # that path only.
    import builtins
    import io
    import json as _json

    real_open = builtins.open

    def fake_open(path, *a, **k):
        if str(path).endswith("server.json"):
            return io.StringIO(_json.dumps(server_info or {}))
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)

    printed: list[str] = []

    def fake_print(*a, **k):
        printed.append(" ".join(str(x) for x in a))

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
    return "\n".join(printed)


def test_merged_cmdline_detected(monkeypatch):
    """setproctitle-style merged single-string argv is detected."""

    class _Proc:
        def __init__(self, pid, cmdline):
            self.info = {"pid": pid, "cmdline": cmdline, "create_time": 0}

    procs = [
        # Older build renamed with args: collapsed string + trailing empties
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

    blob = _run_scan(monkeypatch, procs, server_info={})
    assert "111" in blob, f"renamed server PID 111 not detected: {blob!r}"
    assert "222" in blob, f"original server PID 222 not detected: {blob!r}"
    assert "333" not in blob, "chat process should not match"
    assert "444" not in blob, "caffeinate wrapper should not match"


def test_bare_title_enriched_from_server_json(monkeypatch):
    """Bare fusion-mlx-server title detected + enriched via server.json."""

    class _Proc:
        def __init__(self, pid, cmdline):
            self.info = {"pid": pid, "cmdline": cmdline, "create_time": 0}

    procs = [
        # Current build: bare title + trailing empties
        _Proc(555, ["fusion-mlx-server", "", ""]),
        # Unrelated python process (must NOT match)
        _Proc(666, ["python3", "-c", "print(1)"]),
    ]

    blob = _run_scan(
        monkeypatch,
        procs,
        server_info={
            "pid": 555,
            "host": "127.0.0.1",
            "port": 11434,
            "model": "/Users/x/.fusion-mlx/models",
        },
    )
    assert "555" in blob, f"bare-title server PID 555 not detected: {blob!r}"
    assert "11434" in blob, f"port from server.json missing: {blob!r}"
    assert (
        "/Users/x/.fusion-mlx/models" in blob
    ), f"model from server.json missing: {blob!r}"
    assert "666" not in blob, "unrelated python process should not match"


def test_bare_title_stale_server_json_not_used(monkeypatch):
    """Stale server.json (pid of a dead server) shows (unknown) port."""

    class _Proc:
        def __init__(self, pid, cmdline):
            self.info = {"pid": pid, "cmdline": cmdline, "create_time": 0}

    procs = [_Proc(777, ["fusion-mlx-server", ""])]

    blob = _run_scan(
        monkeypatch,
        procs,
        server_info={"pid": 12345, "host": "127.0.0.1", "port": 9999},
    )
    assert "777" in blob, f"bare-title server PID 777 not detected: {blob!r}"
    assert "9999" not in blob, "stale server.json port must not be shown"
