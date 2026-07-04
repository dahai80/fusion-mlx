#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for fusion-mlx (models, pull, rm, ps, chat, info, agents, upgrade, telemetry)."""

import argparse
import os
import sys

from fusion_mlx._cli_base import (
    MIRROR_DEFAULT,
    _has_seen_tip,
    _mark_tip_seen,
    _print_unknown_model_help,
    alias_completer,
)
from fusion_mlx.cli_serve import (
    _ensure_model_downloaded,
)

def _format_bytes(n: int) -> str:
    """Render a byte count as a 1-decimal IEC-suffixed string (GiB/MiB/KiB).

    Picks the largest unit where the value is >= 1; falls back to bytes.
    Returns ``"0 B"`` for zero / negative.

    Aligned with ``fusion_mlx._download_gate._format_size`` so the same
    byte count rendered by ``ls --cached`` and by the B2 confirmation
    prompt uses the same suffix convention (Codex/DeepSeek round-3 NIT:
    ``5.0 G`` vs ``5.0 GiB`` for the same model is the kind of paper-
    cut that makes users think two screens are talking about different
    sizes).
    """
    if n <= 0:
        return "0 B"
    for unit, factor in (
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    ):
        if n >= factor:
            return f"{n / factor:.1f} {unit}"
    return f"{n} B"


def _dir_size_bytes(path: str) -> int:
    """Recursive on-disk size of ``path`` (follows blob symlinks).

    HF cache stores model weights as ``blobs/<sha>`` files referenced via
    symlinks under ``snapshots/<rev>/<file>``. ``os.scandir`` recurses
    through both — we follow links so a snapshot's reported size matches
    the user's mental model of "how much disk this model uses".
    """
    total = 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_dir(follow_symlinks=False):
                    total += _dir_size_bytes(entry.path)
                else:
                    # follow_symlinks=True so blob symlinks count their
                    # underlying file size, matching ``du -sL``.
                    total += entry.stat(follow_symlinks=True).st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _scan_hf_cache_models() -> list[tuple[str, int, float]]:
    """Return ``[(hf_repo, size_bytes, last_modified_epoch), ...]`` for every
    ``models--<org>--<name>`` directory in the HF cache.

    Empty list when the cache dir doesn't exist (fresh install) or has no
    model entries (e.g. only datasets were downloaded). Datasets/spaces
    (``datasets--*``, ``spaces--*``) are deliberately skipped.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        HF_HUB_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(HF_HUB_CACHE):
        return []
    out: list[tuple[str, int, float]] = []
    for name in os.listdir(HF_HUB_CACHE):
        if not name.startswith("models--"):
            continue
        # ``models--org--name`` → ``org/name``. Some legacy entries are
        # ``models--name`` (no org) for single-segment repos; pass those
        # through unchanged so the user still sees them in the listing.
        parts = name[len("models--") :].split("--", 1)
        repo = "/".join(parts) if len(parts) == 2 else parts[0]
        full = os.path.join(HF_HUB_CACHE, name)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0.0
        size = _dir_size_bytes(full)
        out.append((repo, size, mtime))
    return out


def _print_cached_models() -> None:
    """Render the ``--cached`` view: locally-downloaded HF cache entries
    cross-referenced against the alias registry.

    Each row: ``Alias | HF repo | Size on disk | Last modified``. Models
    not in the alias registry are shown with alias=``(unmapped)`` so the
    user still sees what's eating disk space. Empty cache prints a hint
    pointing at ``pull`` / ``chat``.
    """
    import time as _time

    from fusion_mlx.model_aliases import list_profiles

    rows = _scan_hf_cache_models()
    print()
    if not rows:
        print(
            "  No models cached yet. Run 'fusion-mlx pull <alias>' or "
            "'fusion-mlx chat <alias>' to download one."
        )
        print()
        return

    # Reverse-map HF repo path → alias name so the alias column matches the
    # user's mental model (``qwen3.5-4b-4bit`` not ``mlx-community/Qwen3.5-4B...``).
    # list_profiles() returns list[AliasProfile] (released contract), not a
    # dict — iterate the list directly to build the HF-path→alias reverse map.
    profiles = list_profiles()
    hf_to_alias: dict[str, str] = {}
    for p in profiles:
        hf_to_alias.setdefault(p.hf_path, p.name)

    cols = (
        ("Alias", 22),
        ("HF repo", 50),
        ("Size", 9),
        ("Modified", 12),
    )
    width = sum(w for _, w in cols) + len(cols) - 1
    sep = "  " + "─" * width
    header = "  " + " ".join(f"{name:<{w}}" for name, w in cols)
    print(f"  Cached models ({len(rows)} on disk)")
    print(sep)
    print(header)
    print(sep)

    now = _time.time()
    total_bytes = 0
    # Sort by size descending so the biggest-disk-hog row is first — the
    # most useful ordering for "what do I rm to free space?".
    for repo, size, mtime in sorted(rows, key=lambda r: -r[1]):
        total_bytes += size
        alias = hf_to_alias.get(repo, "(unmapped)")
        # Render modified as a human delta: "2 days ago" beats raw epoch.
        if mtime <= 0:
            mod = "?"
        else:
            delta = max(0, int(now - mtime))
            if delta < 3600:
                mod = f"{delta // 60}m ago"
            elif delta < 86400:
                mod = f"{delta // 3600}h ago"
            else:
                mod = f"{delta // 86400}d ago"
        # Truncate over-long HF paths so the row doesn't wrap on a
        # narrow terminal; the alias column carries the canonical name.
        repo_disp = repo if len(repo) <= 50 else (repo[:47] + "...")
        print(f"  {alias:<22} {repo_disp:<50} {_format_bytes(size):<9} {mod:<12}")
    print(sep)
    print(f"  Total: {_format_bytes(total_bytes)}")
    print()
    print("  Tip: `fusion-mlx rm <hf-repo>` to free disk space")
    print()


def models_command(args):
    # Released 1.0/2.0/3.0 contract (docs/cli-reference.md "models"): query the
    # running server's /v1/models and list discovered model IDs + types, then
    # print DEFAULT_ALIASES. Restored after the Rapid-MLX migration replaced
    # this with a local alias-capability table that depended on a missing
    # aliases.json and a non-existent AliasProfile.suffix_decoding_tier field
    # (it crashed on `profiles.keys()` because list_profiles() returns a list,
    # not a dict). The `--cached` / top-level `ls` path (Rapid-MLX additions,
    # absent from the released guide) scans the local HuggingFace cache via
    # _print_cached_models and is kept — it does not conflict with the released
    # `models` server query.
    import json as _json
    import sys
    from pathlib import Path

    import requests

    from fusion_mlx._version_check import print_staleness_warning_if_any
    from fusion_mlx.config import DEFAULT_ALIASES

    print_staleness_warning_if_any()

    if getattr(args, "cached", False):
        _print_cached_models()
        return

    host = getattr(args, "host", None) or "localhost"
    port = int(getattr(args, "port", None) or 8000)
    base = f"http://{host}:{port}"

    data = None
    try:
        r = requests.get(base + "/v1/models", timeout=5)
        if r.status_code == 200:
            data = r.json()
    except requests.RequestException:
        data = None

    if not data:
        # Released _discover_server fallback: saved server info first, then
        # common ports (incl. 11434/11435 for Ollama-style setups) so
        # `fusion-mlx models` finds a server started on a non-default port
        # without requiring --port.
        candidates = []
        info_path = Path.home() / ".fusion-mlx" / "server.json"
        if info_path.exists():
            try:
                info = _json.loads(info_path.read_text())
                s_host = info.get("host", "localhost")
                s_port = info.get("port")
                if s_port:
                    candidates.append(f"http://{s_host}:{s_port}")
            except (ValueError, OSError):
                pass
        for p in (8000, 11434, 11435, 8001, 3000):
            if p != port:
                candidates.append(f"http://localhost:{p}")
        for cand in candidates:
            try:
                rc = requests.get(cand + "/v1/models", timeout=2)
                if rc.status_code == 200:
                    print(f"Auto-detected server at {cand}")
                    data = rc.json()
                    break
            except requests.RequestException:
                continue

    if not data:
        print("Error: cannot reach server. Is fusion-mlx running?")
        print("  Start it with: fusion-mlx serve --model-dir <dir> --port <port>")
        sys.exit(1)

    models = data.get("data", [])
    if not models:
        print("No models found.")
        return

    print(f"{'MODEL ID':<50} {'TYPE':<12}")
    print("-" * 65)
    for m in models:
        mid = m.get("id", "?")[:47]
        mtype = m.get("type", "llm")
        print(f"{mid:<50} {mtype:<12}")

    if DEFAULT_ALIASES:
        print()
        print("Default aliases:")
        for alias, real in DEFAULT_ALIASES.items():
            print(f"     {alias:<25} -> {real}")
    print()


def _format_pull_duration(seconds: float) -> str:
    """Render a duration as ``Xs`` (< 60s) or ``Xm Ys`` (>= 60s).

    Sub-minute keeps one decimal so a 4.2s pull doesn't read as ``4s``;
    once we cross a minute the decimals are noise. ``round`` (not
    ``int``) on the whole-second branch means ``119.9s`` reads as
    ``2m 0s`` instead of ``1m 59s``.
    """
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}m {secs}s"


def _snapshot_size_bytes(path) -> int:
    """Sum file sizes under ``path`` (recursively, following symlinks).

    The HF cache stores ``snapshots/<rev>/<file>`` as symlinks into
    ``blobs/<sha>``; ``stat()`` follows the link so the byte count is
    the real on-disk weight, matching what the user just downloaded.
    Quietly tolerates partial / missing trees so the summary line is
    a print, not a crash, in degenerate cache states.
    """
    from pathlib import Path

    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    try:
        for entry in root.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def _print_pull_summary(repo_id: str, snapshot_dir, elapsed: float) -> None:
    """Emit the one-line ``Downloaded ... — <size> in <duration>`` summary."""
    size = _snapshot_size_bytes(snapshot_dir)
    print(
        f"  Downloaded {repo_id} — {_format_bytes(size)} in "
        f"{_format_pull_duration(elapsed)}"
    )


def pull_command(args):
    """Download a model to the HuggingFace cache without serving."""
    import time

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import RepositoryNotFoundError

    repo_id = args.model  # already alias-resolved by main()

    # Guard against path traversal in repo_id (e.g. "foo/../bar")
    if ".." in repo_id or repo_id.startswith("/"):
        print(f"\n  Error: invalid model identifier: {repo_id}")
        return

    t0 = time.monotonic()

    # R2-first / HuggingFace-fallback per file. Default mirror is
    # ``https://models.fusionmlx.com``; set ``FUSION_MLX_MODEL_MIRROR=""``
    # to force HF only. The function prints its own progress + summary.
    if _try_mirror_prefetch(repo_id):
        from pathlib import Path

        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            cache_root = Path(HF_HUB_CACHE)
        except Exception:
            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        owner, _, repo = repo_id.partition("/")
        repo_root = cache_root / f"models--{owner}--{repo}"
        try:
            rev = (repo_root / "refs" / "main").read_text().strip()
            snapshot_dir = repo_root / "snapshots" / rev
            print(f"  Cached at: {snapshot_dir}")
        except OSError:
            snapshot_dir = repo_root
            print(f"  Cached at: {repo_root}")
        _print_pull_summary(repo_id, snapshot_dir, time.monotonic() - t0)
        return
    # Mirror returned False — fall through to plain snapshot_download.
    # Either the catalog was unreachable, the alias isn't catalog-listed,
    # or one or more files failed both R2 and HF in the per-file pool.
    # snapshot_download will retry from HF with its own (more robust)
    # error reporting.
    print(f"\n  Pulling {repo_id} from HuggingFace ...")
    try:
        path = snapshot_download(repo_id)
    except HFValidationError:
        # Malformed HF repo id (e.g. ``foo/bar/baz``) — surface the same
        # friendly "unknown model" hint the alias path uses instead of a
        # raw stack trace.
        shown = getattr(args, "_original_alias", repo_id)
        print(
            f"\n  Error: '{shown}' is not a valid HuggingFace repo id "
            "(expected ``namespace/name``)."
        )
        _print_unknown_model_help(
            shown, full_path_example="mlx-community/Qwen3.5-9B-4bit"
        )
        sys.exit(1)
    except Exception as e:
        is_404 = isinstance(e, RepositoryNotFoundError) or (
            "404" in str(e) or "not found" in str(e).lower()
        )
        if is_404:
            shown = getattr(args, "_original_alias", repo_id)
            print(f"\n  Error: Model '{shown}' not found on HuggingFace.")
            _print_unknown_model_help(
                shown, full_path_example="mlx-community/Qwen3.5-9B-4bit"
            )
            sys.exit(1)
        raise
    print(f"  Cached at: {path}")
    _print_pull_summary(repo_id, path, time.monotonic() - t0)


def rm_command(args):
    """Remove a model from the HuggingFace cache.

    Default flow prompts for confirmation, defaulting to N — a real user
    typo (``fusion-mlx rm qwn3.5-9b-4bit`` → matches a 6 GB model) could
    silently nuke gigabytes of weights pre-0.9.7. EOF (non-TTY pipe,
    ctrl-D) also cancels rather than being treated as accept-by-default.
    ``-y/--yes`` skips the prompt for scripts.
    """
    from huggingface_hub import scan_cache_dir

    repo_id = args.model
    cache = scan_cache_dir()
    # Filter by repo_type=="model" — same repo_id can refer to a dataset or
    # space, and we don't want ``fusion-mlx rm foo`` deleting a dataset.
    matching = [
        r for r in cache.repos if r.repo_id == repo_id and r.repo_type == "model"
    ]
    if not matching:
        print(f"\n  '{repo_id}' is not in the HuggingFace cache.")
        print("  Nothing to remove.")
        sys.exit(1)

    repo = matching[0]
    size_str = _format_bytes(repo.size_on_disk)

    if not getattr(args, "yes", False):
        try:
            response = input(f"Remove {repo_id} ({size_str})? [y/N] ").strip().lower()
        except EOFError:
            # Non-TTY (piped stdin, ctrl-D) — treat as cancel, never as
            # silent-yes. Matches ``apt`` / ``brew`` muscle memory.
            print("Aborted.")
            sys.exit(0)
        if response not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    revisions = [rev.commit_hash for rev in repo.revisions]
    strategy = cache.delete_revisions(*revisions)
    strategy.execute()
    print(f"Freed {size_str}")


def ps_command(_args):
    """List running fusion-mlx servers (process scan)."""
    import time

    import psutil

    rows: list[tuple[int, str, str, str]] = []
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            cmd = proc.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not any(
            ("fusion-mlx" in c or "fusion_mlx" in c) and "serve" in cmd for c in cmd
        ):
            continue

        # 0.9.0 dogfood: ``fusion-mlx serve`` runs under a ``caffeinate
        # -is fusion-mlx serve ...`` wrapper on macOS to prevent sleep.
        # The wrapper's argv carries the same ``fusion-mlx`` / ``serve``
        # tokens as the real server, so the substring match above
        # double-counts it as a second row (same port, different PID).
        # The wrapper is never the actual server — its argv[0] basename
        # is the only reliable way to filter it out without missing the
        # case where caffeinate is launched via an absolute path.
        if cmd and os.path.basename(cmd[0]) == "caffeinate":
            continue

        # Extract model arg and --port flag. argparse accepts options
        # before positionals, so the model is the first non-flag token
        # after `serve` whose prior token isn't a value-taking flag.
        # The small list of flags here is conservative; unknown flags
        # are assumed to NOT take a value.
        VALUE_FLAGS = {
            "--host",
            "--port",
            "--api-key",
            "--tool-call-parser",
            "--reasoning-parser",
            "--log-level",
            "--mcp-config",
            "--cors-origins",
            "--cloud-model",
            "--cloud-api-base",
            "--cloud-api-key",
            "--served-model-name",
            "--max-tokens",
            "--gpu-memory-utilization",
        }
        model = "(unknown)"
        port = "8000"  # serve's default
        try:
            i = cmd.index("serve") + 1
            # Pre-PR this loop ``break``ed on the first positional, so a
            # ``fusion-mlx serve qwen3.5-4b-4bit --port 8005`` ended with
            # port="8000" because the positional model token came before
            # ``--port``. Keep scanning for flags after we've captured the
            # model — argparse accepts them on either side.
            model_seen = False
            while i < len(cmd):
                tok = cmd[i]
                if tok.startswith("--"):
                    if "=" in tok:
                        key, val = tok.split("=", 1)
                        if key == "--port":
                            port = val
                        i += 1
                    elif tok in VALUE_FLAGS:
                        if tok == "--port" and i + 1 < len(cmd):
                            port = cmd[i + 1]
                        i += 2
                    else:
                        i += 1
                else:
                    if not model_seen:
                        model = tok
                        model_seen = True
                    i += 1
        except ValueError:
            pass

        uptime_s = max(0, int(time.time() - proc.info["create_time"]))
        h, m = uptime_s // 3600, (uptime_s % 3600) // 60
        uptime = f"{h}h{m:02d}m" if h else f"{m}m{uptime_s % 60:02d}s"
        rows.append((proc.info["pid"], port, model, uptime))

    if not rows:
        print("\n  No fusion-mlx servers running.")
        return

    print()
    print(f"  {'PID':<8}{'PORT':<8}{'MODEL':<40}{'UPTIME':<10}")
    print(f"  {'-' * 66}")
    # Sort numerically by port — string sort would put "10000" before "8000".
    for pid, port, model, uptime in sorted(rows, key=lambda r: int(r[1])):
        print(f"  {pid:<8}{port:<8}{model:<40}{uptime:<10}")
    print()



def _spawn_chat_server(
    model: str,
    log_path: str,
    served_name: str | None = None,
    *,
    register_in: list | None = None,
    log_handle=None,
) -> tuple[object, str]:
    """Spawn a `serve` subprocess on an ephemeral port for chat REPL use.

    Returns (Popen handle, base_url).

    ``register_in`` is an optional list (typically the chat REPL's
    ``_active_procs``). When provided, the new ``Popen`` is appended to it
    *immediately* after construction — narrowing the SIGTERM-orphan race
    that exists between ``Popen()`` returning and the caller registering
    the handle. Caller-side ``register_in.append(proc)`` would still leave
    one Python statement of unprotected window; doing it inside this
    function closes that window for the caller.

    ``log_handle`` is the ``managed_tempfile_path`` context-manager handle
    that owns ``log_path``. When provided, ownership is transferred to the
    proc inside a SIGTERM/SIGINT-masked critical section that also performs
    the ``register_in`` append and the ``_fusion_mlx_log*`` attribute set.
    Without the mask + atomic transfer, a signal landing between
    ``_active_procs.append`` and the caller's later ``handle.release()``
    could fire ``_teardown_proc`` (which intentionally keeps non-empty
    logs for post-mortem) — then the ``with`` block's ``finally`` would
    unlink the kept log anyway, violating the keep-non-empty-log policy
    documented on ``_teardown_proc``. Codex round-1 BLOCKING #1.

    If ``served_name`` is given, it is passed via ``--served-model-name`` so
    the spawned server exposes the alias as the API model name (e.g. user
    typed ``qwen3.5-4b-4bit`` → API requests use ``qwen3.5-4b-4bit`` rather than the
    expanded HF path).
    """
    import signal as _signal
    import socket
    import subprocess

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "fusion_mlx.cli",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "WARNING",
    ]
    if served_name and served_name != model:
        cmd.extend(["--served-model-name", served_name])
    log = open(log_path, "w")  # noqa: SIM115 — kept open for proc lifetime
    # Tell the child main() that the parent already gated (or that this is
    # an internal spawn, where prompting would deadlock anyway because the
    # child stdin is not a TTY). Without this, the child's B2 gate would
    # see a stdin pipe and re-evaluate against a potentially-stale cache.
    child_env = os.environ.copy()
    child_env["FUSION_MLX_CHAT_SPAWN"] = "1"
    # Parent-PID watchdog (rapid-desktop #449 sibling fix). The
    # SIGTERM-handler + atexit pair installed below cannot fire under
    # SIGKILL of the chat REPL — the spawned ``serve`` would otherwise
    # outlive ``fusion-mlx chat`` and keep the model + port locked. The
    # watchdog inside the child polls ``os.getppid()`` every 2 s and
    # self-terminates the moment the live PPID stops matching this
    # stamp.
    #
    # Direct assignment (NOT setdefault). Codex r2 MAJOR: if the chat
    # REPL itself was launched under a supervisor that already exported
    # ``FUSION_MLX_WATCHDOG_PPID=<grandparent_pid>``, ``setdefault``
    # would carry the grandparent's PID into the child env. The
    # watchdog would then compare ``os.getppid()`` (= chat REPL's PID,
    # the IMMEDIATE parent) against the grandparent PID, mismatch on
    # first poll, and self-terminate the freshly-booted server. The
    # spawner owns the watchdog relationship for the spawn it just
    # created — overwrite is correct.
    child_env["FUSION_MLX_WATCHDOG_PPID"] = str(os.getpid())
    # Atomic critical section: block SIGTERM/SIGINT delivery around
    # the whole ``Popen()`` + register + attribute-set + ``release()``
    # sequence. We use ``pthread_sigmask(SIG_BLOCK, ...)`` so the
    # parent thread's mask blocks the signals (queued, delivered when
    # restored).
    #
    # POSIX caveat (codex pr_validate round-3 BLOCKING): both the
    # signal mask AND the signal disposition are inherited across
    # ``fork`` + ``execve``. If we Popen() while the mask blocks
    # SIGTERM/SIGINT, the child server inherits the block and won't
    # honour normal shutdown. The fix is a ``preexec_fn`` that
    # explicitly UNBLOCKS the signals in the child between ``fork``
    # and ``exec`` so the child starts with a clean mask.
    #
    # ``preexec_fn`` runs in the child after fork, before exec, and
    # is exactly the right hook for this. There is no async-signal-
    # safety concern because we are still pre-exec; the child has
    # not yet been replaced with a new image.
    #
    # On platforms without ``pthread_sigmask`` (Windows), fall back
    # to the ``SIG_IGN`` shape — Windows ``subprocess`` doesn't have
    # the same fork/exec model, and the chat REPL is not a Windows
    # feature anyway.
    has_pthread_sigmask = hasattr(_signal, "pthread_sigmask")
    sigset = {_signal.SIGTERM, _signal.SIGINT}
    _prev_mask = None
    _prev_term = _prev_int = None

    def _child_unblock_signals():
        """preexec_fn: clear inherited SIGTERM/SIGINT mask in the child
        so it starts with default mask + default disposition.
        """
        try:
            _signal.pthread_sigmask(_signal.SIG_UNBLOCK, sigset)
        except (ValueError, OSError):
            pass

    try:
        if has_pthread_sigmask:
            try:
                _prev_mask = _signal.pthread_sigmask(_signal.SIG_BLOCK, sigset)
            except (ValueError, OSError):
                _prev_mask = None
        else:
            try:
                _prev_term = _signal.signal(_signal.SIGTERM, _signal.SIG_IGN)
            except (ValueError, OSError):
                pass
            try:
                _prev_int = _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        try:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=child_env,
                # Codex pr_validate r3 BLOCKING: clear the inherited
                # signal mask in the child so it can be terminated
                # normally. ``preexec_fn`` is the documented hook for
                # post-fork / pre-exec setup; the child cannot reach
                # ``exec`` until this runs.
                preexec_fn=_child_unblock_signals if has_pthread_sigmask else None,
            )
        except (OSError, ValueError):
            # Popen raised before constructing the child — the log handle
            # would otherwise leak. Re-raise after closing. The ``finally``
            # below still restores the signal mask / handlers.
            log.close()
            raise
        # Register first so a SIGTERM landing between here and the caller's
        # next statement still tears the child down.
        if register_in is not None:
            register_in.append(proc)
        # Stash the log handle and path on the proc object so the chat REPL
        # can close+unlink them when the proc is torn down (fixes the file
        # descriptor + tempfile leak across `/model` swaps).
        proc._fusion_mlx_log = log
        proc._fusion_mlx_log_path = log_path
        # Hand the tempfile path off to ``_teardown_proc`` BEFORE we
        # leave the masked section. Once released, the ``with`` block's
        # ``finally`` in the caller is a no-op for this path.
        if log_handle is not None:
            log_handle.release()
    finally:
        # Best-effort restore so post-spawn signals route normally. Any
        # SIGTERM/SIGINT that landed while blocked is delivered HERE
        # (kernel-queued, exactly the desired behaviour: the chat's
        # installed handler now sees the proc in ``_active_procs``).
        if has_pthread_sigmask:
            if _prev_mask is not None:
                try:
                    _signal.pthread_sigmask(_signal.SIG_SETMASK, _prev_mask)
                except (ValueError, OSError):
                    pass
        else:
            for signum, prev in (
                (_signal.SIGTERM, _prev_term),
                (_signal.SIGINT, _prev_int),
            ):
                if prev is not None:
                    try:
                        _signal.signal(signum, prev)
                    except (ValueError, OSError):
                        pass
    return proc, base_url


def _wait_for_chat_server(base_url: str, proc, timeout_s: int = 600) -> None:
    """Block until /health/ready returns 200, the proc exits, or timeout.

    On a TTY, draws a spinner + elapsed-seconds counter to stderr so the
    user can see the chat REPL is alive while the spawned server loads
    weights (typically 20-90 s for 4-30 B models on Apple Silicon). The
    line is erased before this function returns so the caller's next
    print lands on a clean line.
    """
    import time

    import requests

    is_tty = sys.stderr.isatty()
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    cyan = "\x1b[36m" if is_tty else ""
    dim = "\x1b[2m" if is_tty else ""
    reset = "\x1b[0m" if is_tty else ""
    start = time.monotonic()
    deadline = start + timeout_s
    tick = 0

    def _draw():
        if not is_tty:
            return
        elapsed = int(time.monotonic() - start)
        ch = spinner[tick % len(spinner)]
        sys.stderr.write(
            f"\r  {cyan}{ch}{reset} loading model ... {dim}{elapsed}s{reset}"
        )
        sys.stderr.flush()

    def _clear():
        if not is_tty:
            return
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()

    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (code {proc.returncode}); "
                    "see chat-server.log for details"
                )
            # Animate the spinner at 10 fps; only poll /health once a
            # second to keep the spinner smooth and the network polite.
            if tick % 10 == 0:
                try:
                    r = requests.get(f"{base_url}/health/ready", timeout=2)
                    if r.status_code == 200:
                        return
                except requests.RequestException:
                    pass
            _draw()
            time.sleep(0.1)
            tick += 1
    finally:
        _clear()
    raise TimeoutError(
        f"server did not become ready within {timeout_s}s "
        "(large models can take longer — pass --ready-timeout)"
    )


def _has_short_pattern_dominating_suffix(
    text: str,
    *,
    window: int = 600,
    max_period: int = 300,
) -> bool:
    """Return True if the trailing ``window`` chars of ``text`` are
    periodic with a cycle length ≤``max_period``.

    Catches the degenerate-model cases the rolling whitespace-token
    counter in ``_stream_chat_response`` misses:

    - ``"BarleyBarleyBarley..."`` (no whitespace separator) — the entire
      suffix collapses to a single ``str.split()`` token whose count
      never increments. Real qwen3.5-4b-4bit regression surfaced in the
      0.6.28 onboarding test.
    - Long-cycle phrase loops, e.g. a ~280-char clause that repeats
      verbatim until ``max_tokens``. Surfaced when asked "describe the
      entire history of the Roman Empire in one long unbroken sentence".

    Implementation: compute the KMP failure function over the trailing
    window. The smallest period of the *entire* window is
    ``len(s) - fail[-1]``; a short period (≤``max_period``) means the
    window is dominated by that repetition starting from offset 0.

    Note: KMP itself does NOT detect periods that begin mid-window
    (rotated patterns). Mid-window degeneracy gets caught because this
    helper is invoked after every streaming chunk — once the model has
    been looping long enough to fill the window, the rolling 600-char
    suffix aligns with the pattern and the smallest-period check fires.
    A pure end-of-stream check would miss rotated cases.

    Cost is ``O(window)`` time and memory per call regardless of
    pattern length (the failure-function array is allocated each
    invocation) — much cheaper than the prior
    ``O(window * pattern_max_len)`` anchored scan, and cheap enough
    to run on every streaming chunk.

    The defaults (window=600, max_period=300) leave room for legitimate
    repetitive content like ``[0, 0, 0, ...]`` lists shorter than the
    window. *Long* lists of truly identical values the user explicitly
    asked for will get cut — a user hitting that false positive can
    ``/reset`` and rephrase. The cost of NOT cutting genuine model
    degeneracy (2000+ tokens of garbage) is far higher.
    """
    if len(text) < window:
        return False
    tail = text[-window:]
    n = len(tail)
    # KMP failure function: ``fail[i]`` = longest proper prefix of
    # ``tail[: i + 1]`` that is also a suffix.
    fail = [0] * n
    for i in range(1, n):
        j = fail[i - 1]
        while j > 0 and tail[i] != tail[j]:
            j = fail[j - 1]
        if tail[i] == tail[j]:
            j += 1
        fail[i] = j
    # Smallest period of ``tail``. Always >= 1 (fail[-1] <= n-1, since
    # ``fail`` is the longest *proper* prefix-suffix). ``period == n``
    # means no nontrivial period — the entire window is its own only
    # period and content is aperiodic. Defaults guarantee
    # ``max_period < window`` so this case never trips, but a caller
    # with ``max_period >= window`` would otherwise see aperiodic
    # strings flagged. Explicit ``period < n`` guard locks the contract.
    period = n - fail[-1]
    return period < n and period <= max_period


def _stream_chat_response(
    base_url: str,
    payload: dict,
    timeout_s: int,
    metrics: dict | None = None,
) -> str:
    """POST /v1/chat/completions with stream=True and print tokens as they
    arrive. Returns the full assistant content (concatenated content deltas).

    Reasoning-content deltas (Qwen3, DeepSeek-R1, etc.) are streamed to stdout
    in dim ANSI so the user sees thinking, but excluded from the returned
    string — chat history stores only the final answer, matching the
    OpenAI-compat split between ``content`` and ``reasoning_content``.

    Plain streaming: tokens land directly in the user's terminal as they
    arrive. We deliberately do NOT use ``rich.Live`` + ``Markdown`` here:
    Live re-renders the panel on every refresh and, when the console's
    cursor-overwrite path is unreliable (recordings, some terminal
    multiplexers), each refresh appends rather than overwrites — turning
    a 200-token response into a wall of repeated text. Live markdown
    rendering deserves a separate, more careful effort with explicit
    fallback detection; for now correctness wins over formatting.
    """
    import json

    import requests

    DIM = "\x1b[2m"
    BOLD = "\x1b[1m"
    RESET = "\x1b[0m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    in_reasoning = False
    full = ""

    # ----- Streaming markdown colorer ------------------------------------
    # Body text streams in the terminal's default color (Claude-Code-style
    # — accents only on chrome). Inline coloring handles the markers users
    # see most often: ``\`code\``` (cyan), ``\`\`\`fence\`\`\``` (dim cyan
    # block), ``**bold**`` (ANSI bold), and ATX headers (``#`` … ``####``)
    # at line start. Lists / italic stay raw so the parser stays small.
    HEADING_STYLE = {
        1: BOLD + CYAN,  # `# h1`     — most prominent
        2: BOLD + MAGENTA,  # `## h2`    — secondary
        3: BOLD,  # `### h3`   — bold only
        4: CYAN,  # `#### h4`  — cyan
        5: MAGENTA,  # `##### h5` — magenta
        6: DIM,  # `###### h6`— dim
    }
    _state = {
        "in_fence": False,  # inside a ``` block
        "in_inline_code": False,  # inside a `code` span
        "in_bold": False,  # inside **bold**
        "in_heading": False,  # inside an ATX heading line
        "at_line_start": True,  # cursor is at start of a logical line
        "pending": "",  # buffered chars awaiting lookahead
    }

    def _emit_with_inline_md(piece: str) -> None:
        if not is_tty:
            sys.stdout.write(piece)
            sys.stdout.flush()
            return
        text = _state["pending"] + piece
        _state["pending"] = ""
        out: list[str] = []
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            # Newline closes any line-scoped span (heading) and resets the
            # line-start anchor so the next `#`/`*`/etc. is interpreted in
            # the right context.
            if c == "\n":
                if _state["in_heading"]:
                    out.append(RESET)
                    _state["in_heading"] = False
                out.append("\n")
                _state["at_line_start"] = True
                i += 1
                continue
            # ATX heading: `#`..`######` followed by space at line start.
            # We skip this inside fences (a `#` at line start there is
            # almost always a comment, not a heading).
            if _state["at_line_start"] and c == "#" and not _state["in_fence"]:
                # Count consecutive `#` (1..6).
                j = i
                while j < n and j - i < 6 and text[j] == "#":
                    j += 1
                # Need to see one more char after the hashes to decide
                # heading vs literal "###foo" — buffer if we don't have it.
                if j == n:
                    _state["pending"] = text[i:]
                    break
                hashes = j - i
                if 1 <= hashes <= 6 and text[j] == " ":
                    style = HEADING_STYLE.get(hashes, BOLD)
                    out.append(style)
                    out.append(text[i : j + 1])  # emit "## "
                    _state["in_heading"] = True
                    _state["at_line_start"] = False
                    i = j + 1
                    continue
                # Not a heading — fall through to literal emission below.
            if c == "`":
                # Need 2 chars of lookahead to disambiguate ``` vs `.
                if i + 2 >= n:
                    _state["pending"] = text[i:]
                    break
                if text[i : i + 3] == "```":
                    if _state["in_fence"]:
                        out.append("```" + RESET)
                        _state["in_fence"] = False
                    else:
                        out.append(DIM + CYAN + "```")
                        _state["in_fence"] = True
                    _state["at_line_start"] = False
                    i += 3
                    continue
                # Single backtick.
                if _state["in_fence"]:
                    out.append("`")
                elif _state["in_inline_code"]:
                    out.append("`" + RESET)
                    _state["in_inline_code"] = False
                else:
                    out.append(CYAN + "`")
                    _state["in_inline_code"] = True
                _state["at_line_start"] = False
                i += 1
                continue
            if c == "*" and not _state["in_fence"] and not _state["in_inline_code"]:
                if i + 1 >= n:
                    _state["pending"] = text[i:]
                    break
                if text[i : i + 2] == "**":
                    if _state["in_bold"]:
                        out.append("**" + RESET)
                        _state["in_bold"] = False
                    else:
                        out.append(BOLD + "**")
                        _state["in_bold"] = True
                    _state["at_line_start"] = False
                    i += 2
                    continue
            out.append(c)
            # Whitespace (other than newline, handled above) keeps the
            # line-start anchor true so leading-indent headings still
            # parse — e.g., a list item's child paragraph is rare here.
            if c not in " \t":
                _state["at_line_start"] = False
            i += 1
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def _close_open_md_spans() -> None:
        if is_tty and (
            _state["in_fence"]
            or _state["in_inline_code"]
            or _state["in_bold"]
            or _state["in_heading"]
        ):
            sys.stdout.write(RESET)
            sys.stdout.flush()
        if _state["pending"]:
            sys.stdout.write(_state["pending"])
            sys.stdout.flush()
            _state["pending"] = ""

    # ----- Repetition guard ----------------------------------------------
    # Models occasionally degenerate into the same token repeated until
    # max_tokens — filling the screen with "Barley Barley Barley...".
    # Two complementary checks run per delta:
    #
    # 1. Whitespace-token-consecutive: the SAME whitespace-split token
    #    repeats ≥``REPEAT_LIMIT`` times in a row. O(1) rolling counter.
    #    Catches the common form ``"Barley Barley Barley..."``. Earlier
    #    guards used "≤2 unique in last 30" but fired on legit content
    #    like ``[0, 0, 0, ...]`` and markdown table separators, so the
    #    bar is now stricter.
    #
    # 2. Character-level pattern check (``_has_short_pattern_dominating_
    #    suffix``): the trailing window is dominated by a short repeating
    #    pattern. Catches the form ``"BarleyBarleyBarley..."`` (no
    #    whitespace separator), where ``piece.split()`` produces one
    #    giant token whose count never increments — this was a real
    #    qwen3.5-4b-4bit regression in 0.6.28 (issue surfaced post-release).
    REPEAT_LIMIT = 25
    repeat_last: str | None = None
    repeat_run = 0
    repetition_aborted = False

    with requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout_s,
    ) as resp:
        if resp.status_code != 200:
            # With stream=True the body may still be partial / mid-chunk when
            # the server closed the socket; read defensively so we surface a
            # useful HTTP code instead of a ChunkedEncodingError.
            try:
                body = resp.text[:500]
            except Exception:
                body = "(no body)"
            raise RuntimeError(f"HTTP {resp.status_code}: {body}")
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            # When the caller passes ``stream_options.include_usage``,
            # the server emits a final chunk with empty choices and a
            # populated ``usage`` block. Capture it for the speed line.
            usage = chunk.get("usage")
            if usage and metrics is not None:
                metrics["completion_tokens"] = usage.get("completion_tokens")
                metrics["prompt_tokens"] = usage.get("prompt_tokens")
            # The usage-only final chunk has ``choices=[]``; guard
            # against an IndexError there.
            choices = chunk.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            # ``finish_reason`` arrives on the last token chunk (after
            # which the server may still emit a usage-only chunk).
            # Capture the most recent non-null value so the caller can
            # surface a "length" warning if the answer was truncated.
            if choices and metrics is not None:
                fr = choices[0].get("finish_reason")
                if fr is not None:
                    metrics["finish_reason"] = fr
            reasoning = delta.get("reasoning_content")
            piece = delta.get("content")
            if reasoning:
                if not in_reasoning:
                    if is_tty:
                        sys.stdout.write(f"{MAGENTA}[thinking]{RESET} {DIM}")
                    else:
                        sys.stdout.write("[thinking] ")
                    in_reasoning = True
                sys.stdout.write(reasoning)
                sys.stdout.flush()
            if piece:
                if in_reasoning:
                    sys.stdout.write(f"{RESET}\n  " if is_tty else "\n")
                    in_reasoning = False
                # Detect repetition BEFORE emitting. If a single coalesced
                # delta contains the cutoff inside it (server batched many
                # repeated tokens into one chunk), find the position and
                # only emit the prefix up to that token — otherwise the
                # user sees the full degenerate dump before the abort
                # message lands.
                #
                # Rolling counter: each new whitespace-separated token in
                # this delta either extends the current consecutive run
                # or resets it. Aborts only on a single token repeated
                # ``REPEAT_LIMIT`` times in a row, not on diverse-but-
                # repetitive content like ``[0, 0, 0, ...]`` or markdown
                # tables.
                cutoff_idx: int | None = None
                tokens = piece.split()
                for i, tok in enumerate(tokens):
                    if tok == repeat_last:
                        repeat_run += 1
                    else:
                        repeat_last = tok
                        repeat_run = 1
                    if repeat_run >= REPEAT_LIMIT:
                        repetition_aborted = True
                        cutoff_idx = i
                        break
                if cutoff_idx is not None:
                    # Find the byte position in ``piece`` corresponding to
                    # the start of the cutoff token, so we can emit only
                    # the prefix. ``str.split()`` collapses runs of
                    # whitespace, so we walk the original text token-by-
                    # token to recover the offset.
                    pos = 0
                    seen = 0
                    while seen < cutoff_idx and pos < len(piece):
                        # Skip leading whitespace.
                        while pos < len(piece) and piece[pos].isspace():
                            pos += 1
                        # Skip the token itself.
                        while pos < len(piece) and not piece[pos].isspace():
                            pos += 1
                        seen += 1
                    prefix = piece[:pos]
                    if prefix:
                        _emit_with_inline_md(prefix)
                        full += prefix
                else:
                    _emit_with_inline_md(piece)
                    full += piece
                # Char-level guard: catches no-whitespace degenerate
                # output like ``"BarleyBarleyBarley..."`` that the
                # whitespace-token counter misses (the entire chunk
                # collapses to one giant token whose consecutive count
                # never climbs). Cheap enough to run on every chunk.
                #
                # Trade-off: runs *after* the chunk is already emitted,
                # so the user sees one extra chunk of garbage before
                # the abort message lands. We accept this — slicing
                # mid-chunk would require re-running KMP per byte (or
                # binary search) on every delta, and degenerate chunks
                # are typically small (≤64 chars) since servers stream
                # token-by-token.
                if not repetition_aborted and _has_short_pattern_dominating_suffix(
                    full
                ):
                    repetition_aborted = True
                if repetition_aborted:
                    break
    _close_open_md_spans()
    if in_reasoning and is_tty:
        sys.stdout.write(RESET)
        sys.stdout.flush()
    if repetition_aborted:
        msg = (
            f"\n\n  {DIM}(response cut: model began repeating itself — "
            f"try /reset or a larger model){RESET}"
            if is_tty
            else "\n\n(response cut: repetition detected)"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()
    return full


def chat_command(args):
    """Interactive REPL chat with a model.

    Spawns a local `serve` on an ephemeral port (or connects to an existing
    server via --base-url / --port), then loops stdin → /v1/chat/completions
    (streaming) → stdout. Maintains multi-turn history; `/reset` clears it.
    Exits cleanly on Ctrl-D, Ctrl-C, or `exit` / `quit`.
    """
    import atexit
    import signal
    import subprocess

    from fusion_mlx._tempfile_safe import managed_tempfile_path

    base_url: str
    proc = None
    log_path: str | None = None
    # Tracks every spawned server (initial + every /model candidate) so
    # the SIGTERM/atexit cleanup tears down in-flight candidates too —
    # not just the bound ``proc``. A SIGTERM landing while a /model
    # swap is mid-spawn would otherwise orphan the candidate server.
    _active_procs: list[subprocess.Popen] = []

    # TTY-gated ANSI palette for the chat UI. NO_COLOR is honoured.
    _is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    BOLD = "\x1b[1m" if _is_tty else ""
    DIM = "\x1b[2m" if _is_tty else ""
    GREEN = "\x1b[32m" if _is_tty else ""
    CYAN = "\x1b[36m" if _is_tty else ""
    YELLOW = "\x1b[33m" if _is_tty else ""
    RED = "\x1b[31m" if _is_tty else ""
    RESET = "\x1b[0m" if _is_tty else ""

    def _teardown_proc(p) -> None:
        """Terminate a spawned chat server and free its log file.

        Used by `_cleanup` (process exit) and `_switch_model` (mid-
        session swap). Idempotent — safe to call when the proc has
        already exited or never existed. Also reaps the killed child
        with wait(timeout=1) so repeated /model swaps don't leave
        zombies until the parent exits.
        """
        if p is None:
            return
        try:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    try:
                        p.kill()
                        # Reap the SIGKILL'd child — without this,
                        # repeated /model swaps stack zombie entries.
                        try:
                            p.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                    except (ProcessLookupError, OSError):
                        pass
                except (ProcessLookupError, OSError):
                    pass
        finally:
            # Drop from the tracked set so a subsequent _cleanup walk
            # doesn't double-tear it down.
            try:
                _active_procs.remove(p)
            except ValueError:
                pass
            # Close the log handle and reap the tempfile so /model
            # swaps don't leak FDs. Both attributes set by
            # _spawn_chat_server.
            #
            # Log file unlink policy: zero-byte logs (no server output
            # ever flushed — typical for a clean spawn that never logged
            # a warning) are unlinked; non-empty logs are LEFT IN PLACE
            # so a user investigating a crash or post-mortem error still
            # has the server's stderr to look at. Previously every log
            # was unlinked, which scrubbed useful debugging breadcrumbs
            # along with the noise.
            fh = getattr(p, "_fusion_mlx_log", None)
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
            lp = getattr(p, "_fusion_mlx_log_path", None)
            if lp:
                try:
                    size = os.path.getsize(lp)
                except OSError:
                    size = -1  # treat unknown as "leave alone"
                if size == 0:
                    try:
                        os.unlink(lp)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass

    # Guard against re-entry: ``_cleanup`` is registered once with
    # ``atexit`` AND fired from the SIGTERM handler. Without an idempotent
    # check, a SIGTERM during shutdown would walk ``_active_procs``,
    # _teardown_proc would empty it, then atexit's invocation would walk
    # an empty list — harmless today, but the explicit flag keeps the
    # contract obvious and survives future helpers that read the list
    # before iterating.
    _cleanup_state = {"done": False}

    def _cleanup():
        # Walk every tracked proc — covers the active server and any
        # in-flight /model candidate. Iterate over a snapshot since
        # _teardown_proc mutates _active_procs. Idempotent: a second call
        # short-circuits so atexit + SIGTERM-handler ordering doesn't
        # matter.
        if _cleanup_state["done"]:
            return
        # Mask BOTH SIGTERM and SIGINT for the duration of the loop.
        # Codex round-3 BLOCKING #1: with only SIGTERM masked, a SIGINT
        # landing mid-teardown raises KeyboardInterrupt, unwinds the
        # for-loop, the surrounding ``finally`` issues ``sys.exit(143)``,
        # and atexit's later call sees ``done=True`` (set at function
        # entry, original implementation) → procs after the interrupted
        # one get orphaned. Move the ``done`` flag to AFTER the loop AND
        # mask SIGINT so a Ctrl-C-during-cleanup can't kill the unwind.
        _prev_term = _prev_int = None
        try:
            _prev_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
        try:
            _prev_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
        try:
            for p in list(_active_procs):
                _teardown_proc(p)
            _cleanup_state["done"] = True
        finally:
            # Best-effort restore so post-cleanup signals route normally.
            # If restore raises, swallow — we're about to exit anyway.
            for signum, prev in (
                (signal.SIGTERM, _prev_term),
                (signal.SIGINT, _prev_int),
            ):
                if prev is not None:
                    try:
                        signal.signal(signum, prev)
                    except (ValueError, OSError):
                        pass

    # Install SIGTERM handler + atexit BEFORE any spawn. Otherwise a
    # SIGTERM landing in the window between `Popen()` and `signal.signal`
    # uses Python's default handler (calls `_exit`, skips atexit) and
    # orphans the spawned server. SIGINT is *deliberately* left on the
    # default handler so Ctrl-C unblocks ``input()`` via the natural
    # KeyboardInterrupt path, the REPL loop's ``except
    # KeyboardInterrupt: break`` fires, and atexit runs ``_cleanup``.
    # On non-tty stdin (piped input) the SIGINT path is never exercised,
    # so the SIGTERM + atexit pair is what reaps the spawned server.
    #
    # Re-entry: a second SIGTERM landing mid-cleanup (common from process
    # supervisors that escalate after a short grace period) would
    # otherwise call _cleanup again — _teardown_proc's
    # ``proc.terminate() + proc.wait(timeout=5)`` would block while the
    # outer cleanup is still mid-wait, leaving the child orphaned.
    # ``_cleanup`` masks both SIGTERM and SIGINT internally for the
    # duration of its teardown loop (Codex round-3 BLOCKING #1), so the
    # handler here only needs to drive the lifecycle: cleanup → exit.
    # The try/finally guarantees sys.exit fires even if _teardown_proc
    # raises (rare — only on the secondary proc.kill() escalation).
    def _sigterm_handler(*_):
        try:
            _cleanup()
        finally:
            sys.exit(143)

    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError):
        pass
    atexit.register(_cleanup)

    if args.base_url:
        base_url = args.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
    elif args.port is not None:
        # Pre-flight probe: a valid-range but unbound port previously
        # dropped the user into the REPL and only failed on the first
        # message with a raw HTTPConnectionPool stack trace. Probe once
        # with a 1 s timeout so the failure is friendly + actionable.
        import socket as _socket

        try:
            with _socket.create_connection(("127.0.0.1", args.port), timeout=1):
                pass
        except OSError:
            # OSError covers ConnectionRefusedError + socket.timeout
            # (which is an alias for ``TimeoutError`` in Python 3.10+).
            print(
                f"\n  {RED}Error:{RESET} no fusion-mlx server reachable at "
                f"127.0.0.1:{args.port}."
            )
            print(f"    Start one with: fusion-mlx serve <alias> --port {args.port}")
            print("    Or omit --port to spawn one automatically.")
            sys.exit(1)
        base_url = f"http://127.0.0.1:{args.port}"
    else:
        # Pre-download in the foreground so the HF tqdm progress bar lands
        # in the user's terminal. Otherwise the serve subprocess swallows
        # the bar into the log file and `fusion-mlx chat` looks frozen for
        # several minutes on first run with a fresh model.
        _ensure_model_downloaded(args.model)

        # GH #719: ``NamedTemporaryFile(...).name`` leaked one zero-byte
        # log per invocation if ANYTHING raised between path creation
        # and the proc being appended to ``_active_procs`` (where
        # ``_teardown_proc`` would otherwise reap it). The
        # ``managed_tempfile_path`` helper registers an atexit unlink
        # the moment the path exists, so the race window is closed:
        # cleanup runs on context exit, on ``sys.exit``, or via atexit
        # if the body propagates. The handle is passed through to
        # ``_spawn_chat_server`` which performs the
        # register/attribute-set/release as a single SIGTERM-masked
        # critical section, so ``_teardown_proc``'s keep-non-empty-log
        # policy cannot be undone by a signal during the handoff
        # (codex round-1 BLOCKING #1).
        with managed_tempfile_path(
            prefix="fusion-mlx-chat-", suffix=".log"
        ) as _log_handle:
            log_path = _log_handle.path
            print(f"\n  Starting server {DIM}(log: {log_path}){RESET} ...")
            # If main() resolved an alias, expose the alias as the API model name
            # so the chat request body matches what the user typed.
            original = getattr(args, "_original_alias", None)
            proc, base_url = _spawn_chat_server(
                args.model,
                log_path,
                served_name=original,
                register_in=_active_procs,
                log_handle=_log_handle,
            )

        try:
            _wait_for_chat_server(base_url, proc, timeout_s=args.ready_timeout)
        except (RuntimeError, TimeoutError) as e:
            print(f"\n  {RED}Failed to start server:{RESET} {e}")
            sys.exit(1)
        print(f"  {GREEN}✓ Ready.{RESET}\n")

    from fusion_mlx._version_check import print_staleness_warning_if_any

    print_staleness_warning_if_any()

    # Resolve ``--max-tokens``. Default is None at the argparse layer so
    # we can distinguish "user did not pass it" from "user passed 2048
    # explicitly". When ``--think`` is set and the user did not supply a
    # value, raise the default from 2048 to 4096 so the reasoning trace +
    # final answer both fit (the round-1 finding: ``chat qwen3.5-4b-4bit
    # --think`` filled the 2048 budget with reasoning and emitted an
    # empty answer with ``finish_reason='length'``).
    user_passed_max_tokens = args.max_tokens is not None
    if args.max_tokens is None:
        args.max_tokens = 4096 if args.think else 2048
    if args.think and not user_passed_max_tokens:
        print(
            f"  {DIM}(--think on; raised --max-tokens to {args.max_tokens} — "
            f"pass --max-tokens to override){RESET}"
        )

    print(
        f"  🐆 {BOLD}Chat{RESET} — "
        f"{DIM}type {RESET}{BOLD}/help{RESET}{DIM} for commands, "
        f"Ctrl-D to exit.{RESET}"
    )
    # First-launch-only banner for the agents/codex tip. The marker-file
    # gate keeps the tip from re-appearing on every chat launch (persona-3
    # finding: irritating by launch #50). Marker logic is skipped entirely
    # when stdout is not a TTY or NO_COLOR is set — pipe/CI runs shouldn't
    # pollute the user's config dir, and the banner is fluff there anyway.
    _is_pipe_or_no_color = (not sys.stdout.isatty()) or ("NO_COLOR" in os.environ)
    if not _is_pipe_or_no_color and not _has_seen_tip("chat_intro_codex"):
        print(
            f"  {DIM}For a Claude Code-like TUI: `fusion-mlx agents codex --setup`, "
            f"then run `codex` in any project.{RESET}\n"
        )
        _mark_tip_seen("chat_intro_codex")
    else:
        # Maintain the existing blank-line spacing the banner used to
        # provide — keeps the prompt layout consistent across runs.
        print()

    served_name = getattr(args, "_original_alias", args.model)
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    # The fusion-mlx server's ChatCompletionRequest exposes a top-level
    # ``enable_thinking`` field — ``chat_template_kwargs`` is not a recognized
    # request field and would be silently dropped.
    #
    # Default thinking OFF in the REPL. Reasoning models (Qwen3.5/3.6, etc.)
    # otherwise emit raw chain-of-thought to stdout AND, on the default
    # qwen3.5-4b-4bit model, degenerate into infinite repetition until max-tokens
    # truncates the response — producing zero usable output for a brand-new
    # user. ``--think`` opts back in for users who explicitly want to see
    # reasoning traces; ``--no-think`` is preserved as the legacy form.
    extra: dict = {}
    if not args.think:
        extra["enable_thinking"] = False

    import time

    import requests

    # Importing ``readline`` upgrades the built-in ``input()`` so that
    # the arrow keys recall earlier prompts (and Ctrl-A/E/U/R work).
    # The module is stdlib on macOS/Linux; on Windows it doesn't exist
    # and we fall back to plain input(). When readline IS available we
    # need to wrap the colored prompt's ANSI escapes in \001/\002 so
    # readline's column counter doesn't include the invisible bytes —
    # otherwise long history entries wrap incorrectly and Ctrl-A jumps
    # to the wrong column (especially on libedit-backed Apple system
    # python). The wrappers are no-op on a terminal, so it's safe to
    # always emit them when readline is loaded.
    have_readline = False
    try:
        import readline  # noqa: F401 — side-effect import

        have_readline = True
    except ImportError:
        pass

    def _wrap_invisible(esc: str) -> str:
        if have_readline and esc:
            return "\001" + esc + "\002"
        return esc

    if _is_tty:
        prompt = _wrap_invisible(BOLD + CYAN) + ">" + _wrap_invisible(RESET) + " "
        cont_prompt = _wrap_invisible(DIM) + "…" + _wrap_invisible(RESET) + " "
    else:
        prompt = "> "
        cont_prompt = "… "

    def _print_help():
        print(
            f"\n  {BOLD}Slash commands{RESET}\n"
            f"    {BOLD}/help{RESET}, {BOLD}/?{RESET}          show this help\n"
            f"    {BOLD}/reset{RESET}, {BOLD}/clear{RESET}     clear conversation history\n"
            f"    {BOLD}/model <alias>{RESET}     switch model "
            f"{DIM}(restarts the server, resets history){RESET}\n"
            f"    {BOLD}/save <path>{RESET}       save conversation to a markdown file\n"
            f"    {BOLD}/exit{RESET}, {BOLD}/quit{RESET}, {BOLD}/bye{RESET}    "
            f"exit chat {DIM}(or Ctrl-D){RESET}\n"
            f"\n  {BOLD}Multi-line input{RESET}\n"
            f'    type {BOLD}"""{RESET} on its own line to start, again to end '
            f"{DIM}(paste code blocks){RESET}\n"
            f"\n  {BOLD}Keys{RESET}\n"
            f"    {BOLD}Ctrl-C{RESET}             cancel the current response, "
            f"or exit at empty prompt\n"
            f"    {BOLD}Ctrl-D{RESET}             exit\n"
        )

    def _save_conversation(path_arg: str):
        # Refuse early on an empty conversation — otherwise we create a
        # near-empty file then lock the user out of the same path on
        # the next try (since exclusive-mode open refuses overwrite).
        non_system = [m for m in messages if m.get("role") != "system"]
        if not non_system:
            print(
                f"  {YELLOW}Nothing to save yet.{RESET} "
                f"{DIM}(send a chat turn first){RESET}\n"
            )
            return
        path = os.path.expanduser(path_arg)
        # Auto-create parent directories; otherwise users see a confusing
        # "No such file or directory" for /save logs/2026-05/convo.md.
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as exc:
                print(f"  {RED}Save failed:{RESET} cannot create {parent}: {exc}\n")
                return
        try:
            # Mode "x" (O_CREAT | O_EXCL) is atomic — refuses if the path
            # already exists, with no TOCTOU window between exists() and
            # open() that an exists()-then-open("w") check has. Also
            # naturally rejects existing symlinks pointing elsewhere.
            with open(path, "x", encoding="utf-8") as f:
                f.write(f"# fusion-mlx chat — {served_name}\n\n")
                for m in messages:
                    if m["role"] == "system":
                        continue
                    f.write(f"## {m['role'].capitalize()}\n\n{m['content']}\n\n")
            print(f"  {GREEN}✓{RESET} Saved {len(messages)} messages to {path}\n")
        except FileExistsError:
            print(
                f"  {YELLOW}{path} already exists.{RESET} "
                f"{DIM}(/save won't overwrite — pick a different path){RESET}\n"
            )
        except IsADirectoryError:
            print(
                f"  {RED}Save failed:{RESET} {path} is a directory — "
                f"{DIM}give a file path, not a directory{RESET}\n"
            )
        except OSError as exc:
            print(f"  {RED}Save failed:{RESET} {exc}\n")

    def _read_multiline() -> str:
        lines: list[str] = []
        while True:
            try:
                more = input(cont_prompt)
            except (EOFError, KeyboardInterrupt):
                # Tell the user how many lines they're losing — silent
                # discard on Ctrl-C/Ctrl-D mid-paste is hostile.
                if lines:
                    print(
                        f"\n  {YELLOW}(multi-line cancelled — "
                        f"{len(lines)} line{'' if len(lines) == 1 else 's'} "
                        f"discarded){RESET}\n"
                    )
                else:
                    print(f"\n  {YELLOW}(multi-line cancelled){RESET}\n")
                return ""
            if more.rstrip() == '"""':
                # Preserve leading/trailing whitespace verbatim — the
                # heredoc is meant for code paste, where stripping
                # indentation actively corrupts the input.
                return "\n".join(lines)
            lines.append(more)

    def _switch_model(new_alias: str) -> None:
        """Hot-swap the spawned chat server to a new model alias.

        Order matters: validate + pre-download the new model BEFORE
        terminating the old one. If anything fails (bogus alias, disk
        gate, network), the old server stays running and the REPL is
        usable. Only when the new model is on-disk and the new server is
        spawn-ready do we tear down the old proc and rebind.
        """
        nonlocal proc, base_url, log_path, served_name, messages
        if proc is None:
            print(
                f"  {YELLOW}/model is only available when chat spawns its "
                f"own server (not with --base-url / --port).{RESET}\n"
            )
            return
        from fusion_mlx.model_aliases import resolve_model

        resolved = resolve_model(new_alias) or new_alias
        print(f"  {DIM}Preparing {new_alias} → {resolved} ...{RESET}")

        # 1a. Gate before download: the main() entry-point gate only
        #     fires on the CLI invocation, so an uncached /model swap
        #     would otherwise start a 40+ GB pull with no prompt.
        #     Mirror main()'s cheap env/TTY short-circuit so we don't
        #     pay the 5-second HF metadata round-trip on every /model
        #     swap when the user opted into AUTO_PULL or is on non-TTY
        #     stdin. ``confirm_or_abort`` self-skips again internally
        #     but skipping ``estimate_repo_size_bytes`` saves the wait.
        if "/" in resolved and not os.path.exists(resolved):
            _env_val = os.environ.get("FUSION_MLX_AUTO_PULL", "").strip().lower()
            _auto_yes = _env_val in {"1", "true", "yes"}
            _interactive = sys.stdin.isatty()
            if not _auto_yes and _interactive:
                from fusion_mlx._download_gate import (
                    confirm_or_abort,
                    estimate_repo_size_bytes,
                    is_repo_cached,
                )

                if not is_repo_cached(resolved):
                    try:
                        confirm_or_abort(
                            resolved,
                            estimate_repo_size_bytes(resolved),
                        )
                    except SystemExit:
                        # User said no — keep the current server up.
                        print(
                            f"  {YELLOW}Model switch cancelled{RESET} "
                            f"{DIM}(previous server still running).{RESET}\n"
                        )
                        return

        # 1. Pre-download the new model (this also runs the disk-space
        #    gate). The current server keeps running while we do this so
        #    a download failure leaves the user where they were.
        try:
            _ensure_model_downloaded(resolved)
        except SystemExit:
            # Disk gate aborted via sys.exit(1); old server is untouched.
            print(
                f"  {RED}Model switch aborted{RESET} "
                f"{DIM}(disk gate); previous server still running.{RESET}\n"
            )
            return
        except RuntimeError as exc:
            # Definitive 404 from HF; old server stays.
            print(
                f"  {RED}Model switch aborted:{RESET} {exc}  "
                f"{DIM}(previous server still running){RESET}\n"
            )
            return

        # 2. Allocate a new log file and spawn the new server. We don't
        #    tear down the old one yet; we want a working candidate
        #    before we commit. ``managed_tempfile_path`` (GH #719)
        #    guarantees the log path is unlinked if the spawn raises
        #    before the proc is registered onto ``_active_procs`` —
        #    the leak window in the original ``NamedTemporaryFile(...).name``
        #    pattern. The handle is passed into ``_spawn_chat_server``
        #    so the register/attribute-set/release happens under one
        #    SIGTERM/SIGINT mask, preserving ``_teardown_proc``'s
        #    keep-non-empty-log policy on signal-during-handoff (codex
        #    round-1 BLOCKING #1).
        with managed_tempfile_path(
            prefix="fusion-mlx-chat-", suffix=".log"
        ) as _new_log_handle:
            new_log_path = _new_log_handle.path
            print(f"  Starting server {DIM}(log: {new_log_path}){RESET} ...")
            # ``register_in=_active_procs`` makes the candidate visible to
            # ``_cleanup`` *inside* ``_spawn_chat_server`` — before the
            # readiness wait, before any further Python statement runs in
            # this scope. A SIGTERM/Ctrl-C during the (possibly multi-second)
            # load tears the child down via the cleanup walk.
            new_proc, new_base_url = _spawn_chat_server(
                resolved,
                new_log_path,
                served_name=new_alias,
                register_in=_active_procs,
                log_handle=_new_log_handle,
            )
        try:
            _wait_for_chat_server(new_base_url, new_proc, timeout_s=args.ready_timeout)
        except (RuntimeError, TimeoutError) as exc:
            print(
                f"  {RED}Failed to start new server:{RESET} {exc}  "
                f"{DIM}(previous server still running){RESET}\n"
            )
            # Roll back: tear down the half-spawned new proc + free its
            # log file. The old proc/base_url/log_path stay bound.
            _teardown_proc(new_proc)
            return

        # 3. New server is healthy — commit. Rebind ``proc`` BEFORE
        #    tearing down the old one so a SIGTERM during teardown
        #    walks the new (still-running) proc, not just a freshly
        #    killed corpse.
        old_proc = proc
        proc = new_proc
        base_url = new_base_url
        log_path = new_log_path
        served_name = new_alias
        messages = [{"role": "system", "content": args.system}] if args.system else []
        _teardown_proc(old_proc)
        print(
            f"  {GREEN}✓ Switched to {new_alias}.{RESET} "
            f"{DIM}(history cleared){RESET}\n"
        )

    while True:
        try:
            line = input(prompt).rstrip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        # Heredoc-pasted content must NEVER be dispatched as a slash
        # command — a markdown doc whose first line starts with `/path`
        # or whose content includes `/save` would otherwise be silently
        # eaten by the slash dispatcher. Track the source so we know.
        is_heredoc = False
        if line == '"""':
            line = _read_multiline()
            if not line:
                continue
            is_heredoc = True
        if not is_heredoc:
            # Parse the leading word as the command and dispatch on
            # *exact* match. ``startswith("/save")`` would otherwise treat
            # ``/savefoo`` as ``/save`` (with arg ``foo``), silently
            # writing a file from a typo. Same for ``/modelfoo``.
            # ``str.split(maxsplit=1)`` (no separator arg) splits on any
            # whitespace, so ``/save\tpath.md`` works the same as
            # ``/save path.md``.
            parts = line.split(maxsplit=1)
            cmd = parts[0] if parts else ""
            rest = parts[1].strip() if len(parts) > 1 else ""
            # ``/bye`` is an Ollama-muscle-memory alias for ``/exit`` /
            # ``/quit``. ``/?`` mirrors ``/help`` and was already
            # supported; both alias sets are advertised in ``/help``.
            if cmd in ("exit", "quit", "/exit", "/quit", "/bye"):
                break
            if cmd in ("/help", "/?"):
                _print_help()
                continue
            if cmd in ("/reset", "/clear"):
                messages = (
                    [{"role": "system", "content": args.system}] if args.system else []
                )
                print(f"  {DIM}(history cleared){RESET}\n")
                continue
            if cmd == "/save":
                if not rest:
                    print(f"  {YELLOW}Usage: /save <path>{RESET}\n")
                else:
                    _save_conversation(rest)
                continue
            if cmd == "/model":
                if not rest:
                    print(
                        f"  {YELLOW}Usage: /model <alias>{RESET}  "
                        f"{DIM}(see `fusion-mlx models`){RESET}\n"
                    )
                else:
                    _switch_model(rest)
                continue
            if cmd.startswith("/"):
                print(
                    f"  {YELLOW}Unknown command: {cmd}{RESET}  "
                    f"{DIM}(type /help){RESET}\n"
                )
                continue

        messages.append({"role": "user", "content": line})
        payload = {
            "model": served_name,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            **extra,
        }
        # Claude-Code-style turn marker: a colored bullet introduces the
        # assistant's response so the user can visually scan turn
        # boundaries when scrolling back through long conversations.
        sys.stdout.write(f"\n  {CYAN}●{RESET} ")
        sys.stdout.flush()
        metrics: dict = {}
        start_t = time.monotonic()
        try:
            assistant = _stream_chat_response(
                base_url,
                payload,
                timeout_s=args.response_timeout,
                metrics=metrics,
            )
        except KeyboardInterrupt:
            print(f"\n  {YELLOW}(response interrupted){RESET}\n")
            messages.pop()
            continue
        except RuntimeError as e:
            print(f"\n  {RED}{e}{RESET}\n")
            messages.pop()
            continue
        except requests.RequestException as e:
            # Connection refused, timeout, dropped midstream — keep the REPL
            # alive and roll back the failed user turn so the next request
            # doesn't carry a dangling user role with no assistant reply.
            print(f"\n  {RED}Request failed:{RESET} {e}\n")
            messages.pop()
            continue
        elapsed = time.monotonic() - start_t
        # Speed line: prefer server-reported usage, fall back to a rough
        # 4-chars-per-token estimate when the server doesn't ship usage
        # in the stream.
        tokens = metrics.get("completion_tokens")
        if not tokens:
            tokens = max(1, len(assistant) // 4)
            tokens_label = f"~{tokens}"
        else:
            tokens_label = str(tokens)
        if assistant and elapsed > 0:
            tps = tokens / elapsed
            print(
                f"\n  {DIM}{tokens_label} tok · {elapsed:.1f}s · "
                f"{tps:.0f} tok/s{RESET}\n"
            )
        else:
            print()
        # Length-cut + empty-content warning. When the server stops
        # because ``finish_reason == "length"`` AND no visible content
        # was streamed (only reasoning), the user otherwise sees an
        # empty bullet and has no signal that the budget was the
        # problem. This is the round-1 ``--think`` regression: 2048-
        # token budget filled by reasoning on small models, zero answer.
        if metrics.get("finish_reason") == "length" and not assistant:
            print(
                f"  {YELLOW}(reasoning consumed the full --max-tokens "
                f"budget; bump --max-tokens for a final answer){RESET}\n"
            )
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        else:
            messages.pop()


def info_command(args):
    """Print the per-model profile for a model name or alias.

    Stage 1 (regex match) only — does NOT load the model, so this is fast
    and works without weights. Stage 2 (ArraysCache probe) is skipped.
    """
    from fusion_mlx.model_aliases import resolve_model, resolve_profile
    from fusion_mlx.model_auto_config import (
        detect_model_config,
        format_profile_table,
    )

    # ``main()`` (cli.py:~3400) pre-resolves ``args.model`` from alias →
    # HF path before dispatch, stashing the user-typed alias on
    # ``args._original_alias``. Pull from that first so DFlash
    # eligibility (alias-keyed) and the start-command hint render with
    # the alias the user actually typed, not the resolved HF repo.
    original_alias = getattr(args, "_original_alias", None) or args.model
    name = args.model
    resolved = (
        resolve_model(name) if not getattr(args, "_original_alias", None) else None
    )
    if resolved and resolved != name:
        print(f"  alias: {name} → {resolved}")
        name = resolved

    cfg = detect_model_config(name)
    print()
    print(format_profile_table(name, cfg))
    print()

    # DFlash eligibility — render the report so users can see which
    # gates pass/fail without consulting the docs. Skipped for unknown
    # models since AliasProfile is alias-keyed.
    profile = resolve_profile(original_alias)
    if profile is not None:
        _print_dflash_status(original_alias, profile)

    if cfg is None:
        print("  No pattern matched — runtime probe will run when the model loads.")
        print()


def _print_dflash_status(alias: str, profile) -> None:
    """Render a 3-row DFlash status block for ``fusion-mlx info <alias>``.

    Shows each gate (declared support / not MoE / not 4-bit / drafter
    present) so a user who tried ``--enable-dflash`` and got a vague
    error can see exactly which gate they're tripping.
    """
    from fusion_mlx.speculative.dflash.eligibility import (
        _looks_like_4bit,
        have_runtime,
        report,
    )

    r = report(profile, alias=alias)
    inner = 60
    sep = "─" * inner

    def _row(text: str) -> str:
        return f"│ {text:<{inner}} │"

    def _yes(ok: bool, msg_ok: str, msg_no: str) -> str:
        return ("✓ " + msg_ok) if ok else ("✗ " + msg_no)

    rows = [
        (
            "Declared support",
            _yes(profile.supports_dflash, "yes (supports_dflash=true)", "no"),
        ),
        ("Not MoE", _yes(not profile.is_moe, "yes (dense)", "no (MoE)")),
        (
            "Precision ≥8-bit",
            _yes(
                not _looks_like_4bit(profile.hf_path),
                "yes",
                "no (4-bit/mxfp4/nvfp4)",
            ),
        ),
        (
            "Drafter declared",
            _yes(
                bool(profile.dflash_draft_model),
                profile.dflash_draft_model or "yes",
                "no (dflash_draft_model unset)",
            ),
        ),
        (
            "mlx-vlm 0.5.0+",
            _yes(have_runtime(), "installed", "missing (need fusion-mlx[dflash])"),
        ),
    ]

    eligible = not r.reasons and have_runtime()
    summary = "✓ eligible" if eligible else "✗ ineligible"

    top = "┌" + "─" * (inner + 2) + "┐"
    bot = "└" + "─" * (inner + 2) + "┘"

    body = [top, _row(f"DFlash eligibility: {summary}"), _row(sep)]
    for k, v in rows:
        body.append(_row(f"{k:<18}: {v}"))
    body.append(bot)
    print("\n".join(body))
    print()
    if eligible:
        print(f"  Start with: fusion-mlx serve {alias} --enable-dflash")
        print()


def agents_command(args):
    """List, configure, and test agent integrations."""
    from fusion_mlx.agents import get_profile, list_profiles
    from fusion_mlx.agents.adapter import get_setup_instructions, setup_agent_config

    agent_name = args.agent_name
    base_url = args.base_url

    # No agent specified → list all profiles
    if not agent_name:
        profiles = list_profiles()
        print()
        print("  Supported AI Agents")
        print("  " + "─" * 56)
        for p in profiles:
            fc = "FC" if p.needs_function_calling else "  "
            stars = f"{p.stars // 1000}K" if p.stars and p.stars >= 1000 else ""
            if p.recommended_models:
                shown = p.recommended_models[:3]
                models = ", ".join(shown)
                if len(p.recommended_models) > 3:
                    models += f" +{len(p.recommended_models) - 3}"
            else:
                models = ""
            print(f"  {p.name:<15} {p.display_name:<20} {stars:>5}  [{fc}]  {models}")
        print()
        print(f"  {len(profiles)} agents supported")
        print("  Usage: fusion-mlx agents <name>          Show setup guide")
        print("         fusion-mlx agents <name> --setup   Auto-configure")
        print("         fusion-mlx agents <name> --test    Run integration tests")
        print()
        return

    # Get profile
    profile = get_profile(agent_name)
    if not profile:
        print(f"  Unknown agent: {agent_name}")
        print("  Run 'fusion-mlx agents' to see available agents.")
        sys.exit(1)

    # --test: run integration tests
    if args.test:
        from fusion_mlx.agents.testing import AgentTestRunner

        model_id = args.model or None
        runner = AgentTestRunner(
            profile,
            base_url=base_url,
            model_id=model_id,
            agent_version=args.agent_version,
        )
        if not runner._server_available():
            print(f"\n  Server not running at {base_url}")
            print("  Start it first: fusion-mlx serve <model>")
            sys.exit(1)

        report = runner.run()
        success = report.print_summary()
        sys.exit(0 if success else 1)

    # --setup: auto-configure agent
    if args.setup:
        # Detect model from running server
        model_id = args.model or "default"
        if model_id == "default":
            try:
                import httpx

                resp = httpx.get(f"{base_url}/models", timeout=3)
                model_id = resp.json()["data"][0]["id"]
            except Exception:
                pass

        summary = setup_agent_config(
            profile, base_url, model_id, agent_version=args.agent_version
        )
        print(f"\n  {profile.display_name} configured!")
        print(f"  {summary}")
        print()
        return

    # Default: show setup instructions
    # Pass "default" to trigger auto-detection of running model
    model_id = args.model or "default"
    instructions = get_setup_instructions(
        profile, base_url, model_id, agent_version=args.agent_version
    )
    print()
    print(instructions)
    print()


def upgrade_command(args):
    """Detect install method and (optionally) run the right upgrade command."""
    import subprocess

    from fusion_mlx._version_check import (
        _installed_version,
        _parse_version,
        detect_install_method,
        get_latest_version,
    )

    current = _installed_version() or "dev"
    print()
    print(f"  Current:  fusion-mlx {current}")

    latest = get_latest_version(force_refresh=True)
    if latest is None:
        print("  Latest:   (could not reach GitHub — check your network)\n")
        sys.exit(1)
    print(f"  Latest:   fusion-mlx {latest}")

    cur = _parse_version(current)
    lat = _parse_version(latest)
    if cur is not None and lat is not None and cur >= lat:
        print("\n  ✓ Already up to date.\n")
        return

    info = detect_install_method()
    print(f"  Install:  {info.method} ({info.binary_path or 'unknown path'})")
    print(f"  Command:  {info.upgrade_command}")
    print()

    if info.method == "unknown":
        print(
            "  Could not auto-detect install method — run the command above manually.\n"
        )
        return

    if getattr(args, "dry_run", False):
        print("  (dry-run — not executed; rerun without --dry-run to apply.)\n")
        return

    if args.yes:
        confirmed = True
    else:
        # Default Y — the user already typed the upgrade command;
        # punishing the Enter key with a no-op skip is bad UX. EOF on
        # stdin is treated as Enter (proceed), mirroring the download
        # gate. Ctrl-C is the only "skip" path — it returns silently
        # without ``sys.exit`` because upgrade is a leaf operation, so
        # there's nothing downstream to abort; cf. the gate, which
        # exits 1 because it's gatekeeping a multi-GB download.
        try:
            answer = input("  Run now? [Y/n] ").strip().lower()
        except EOFError:
            answer = ""
        except KeyboardInterrupt:
            print()
            return
        confirmed = answer not in {"n", "no"}

    if not confirmed:
        print("  Skipped — run the command above when ready.\n")
        return

    print()
    try:
        # Use argv form (shell=False) so paths with spaces in
        # ``sys.executable`` (or any other argv entry) can't be reinterpreted
        # as shell separators. install.sh's pipe is wrapped as ``bash -c``
        # in upgrade_argv, so we still get the pipe semantics it needs.
        result = subprocess.run(info.upgrade_argv, check=False)
    except KeyboardInterrupt:
        print("\n  Interrupted.\n")
        sys.exit(130)
    print()
    sys.exit(result.returncode)


def telemetry_command(args) -> None:
    """Manage anonymous usage telemetry — see Issue #236.

    Five actions: ``status`` / ``enable`` / ``disable`` / ``preview`` /
    ``reset``. Defaults to ``status`` when no action given so users can
    type ``fusion-mlx telemetry`` and immediately see what's set up.
    """
    # Imports kept inside the function so the telemetry package is only
    # loaded when actually needed — keeps `--help` and unrelated
    # subcommands cheap.
    import json

    from fusion_mlx import __version__ as fusion_mlx_version
    from fusion_mlx.telemetry import (
        consent_source,
        get_consent_state,
        get_or_create_client_id,
        is_enabled,
        record_consent,
        reset_state,
    )
    from fusion_mlx.telemetry.schema import sample_preview_payload
    from fusion_mlx.telemetry.state import client_id_path, consent_path

    action = getattr(args, "telemetry_action", None) or "status"
    cli_no = getattr(args, "no_telemetry", False)

    if action == "status":
        state = get_consent_state()
        print()
        print(
            f"  Telemetry: {'ENABLED' if is_enabled(cli_no_telemetry=cli_no) else 'disabled'}"
        )
        print(f"  Source:    {consent_source(cli_no_telemetry=cli_no)}")
        if state is not None:
            print(
                f"  Consent:   {state.consent} (recorded {state.prompted_at}, "
                f"by fusion-mlx {state.prompted_version})"
            )
        else:
            print("  Consent:   never prompted")
        print(f"  Files:     {consent_path()}")
        print(f"             {client_id_path()}")
        print()
        print("  Subcommands:  enable | disable | preview | reset")
        print()
        return

    if action == "enable":
        record_consent(True, fusion_mlx_version=fusion_mlx_version)
        # Generate the client_id eagerly so `preview` immediately after
        # has a real id to show.
        get_or_create_client_id()
        print()
        print("  Telemetry: ENABLED. Thanks for helping us prioritise.")
        print("  Disable anytime with `fusion-mlx telemetry disable`.")
        print("  Preview what we'd send: `fusion-mlx telemetry preview`.")
        print()
        return

    if action == "disable":
        record_consent(False, fusion_mlx_version=fusion_mlx_version)
        print()
        print("  Telemetry: disabled. No data will be sent.")
        print("  Re-enable anytime with `fusion-mlx telemetry enable`.")
        print()
        return

    if action == "preview":
        cid = get_or_create_client_id()
        payload = sample_preview_payload(
            client_id=cid, fusion_mlx_version=fusion_mlx_version
        )
        print()
        print("  Sample payload (this is exactly the shape we send):")
        print()
        print(json.dumps(payload.to_dict(), indent=2))
        print()
        if not is_enabled(cli_no_telemetry=cli_no):
            print("  Telemetry is currently disabled — nothing is actually sent.")
            print()
        return

    if action == "reset":
        reset_state()
        print()
        print("  Removed consent + client-id files. Next interactive run re-prompts.")
        print()
        return

    # Unknown action — argparse choices=[] would have caught this earlier
    # in normal flow; defensive guard for future maintainers.
    print(f"  Unknown telemetry action: {action!r}")
    sys.exit(1)


def doctor_command(args):
    """Run environment health checks via the doctor module."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info("running doctor env-health checks")
    from fusion_mlx.doctor.cli import doctor_command as _doctor_command
    _doctor_command(args)
