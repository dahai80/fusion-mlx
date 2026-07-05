"""Shared CLI helpers and constants for fusion-mlx."""

import argparse
import os
import sys

MIRROR_DEFAULT = "https://models.fusionmlx.com"


def _log_level_choice(value: str) -> str:
    """Argparse ``type`` callable: normalize to upper-case so
    ``--log-level info`` is accepted as ``INFO``. Named (not a lambda)
    so argparse's error messages read sensibly instead of
    ``invalid <lambda> value``.
    """
    return value.upper()


def _auth_feature_str(argv_api_key: str | None) -> str | None:
    """Banner-side renderer for the ``auth: on`` feature line.

    Returns ``"auth: on"`` when the effective API key (argv or env)
    is non-empty, else ``None`` so the banner omits the feature.

    Lives at module scope (not inline in ``serve_command``) so the
    banner gate is directly unit-testable without booting a model.
    Routes through ``server._resolve_api_key`` — the same SSOT the
    server-side enforcement reads — so a refactor of the env-var
    policy cannot drift the banner from the actual auth state.
    Pre-fix the gate was ``if args.api_key`` directly, which printed
    ``auth: off`` for env-only sidecars even though
    ``verify_api_key`` was enforcing (dogfood-v0.8.2 finding #3).
    """
    from fusion_mlx import server as _server

    if _server._resolve_api_key(argv_api_key):
        return "auth: on"
    return None


def _port_arg(value: str) -> int:
    """Argparse ``type`` callable: validate ``--port`` is in [1, 65535].

    Without this, ``fusion-mlx chat --port 99999`` parsed successfully and
    dropped the user into a REPL whose first turn failed with a confusing
    ``Failed to parse: http://127.0.0.1:99999/...``. Validate early so the
    user sees a one-line argparse error instead.
    """
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"port must be an integer, got {value!r}"
        ) from None
    if not (1 <= port <= 65535):
        raise argparse.ArgumentTypeError(
            f"port must be between 1 and 65535, got {port}"
        )
    return port


def _listen_fd_arg(value: str) -> int:
    """Argparse ``type`` callable: validate ``--listen-fd`` is a sane fd.

    ``--listen-fd`` enables socket activation — the supervisor (launchd,
    systemd, an external parent process) binds the listening socket
    itself and execve's into ``fusion-mlx serve`` with the pre-bound fd.
    This closes the bind→auth TOCTOU window: by the time fusion-mlx
    runs, the socket is already bound but no requests can be accepted
    until ``uvicorn.run`` calls ``accept()`` — at which point the
    FastAPI app (with all route auth dependencies wired) is already
    constructed. See ``fusion_mlx/server.py`` and the regression test
    pinning the bind→auth invariant.

    Accept integers in ``[3, 1023]``:

    * 0/1/2 are stdin/stdout/stderr — never a listening socket.
    * 3 is the conventional "first non-stdio fd" (systemd's
      ``LISTEN_FDS_START`` and launchd both follow this convention).
    * 1023 is the SysV soft-limit ceiling — anything higher is almost
      certainly a typo, not a real fd.
    """
    try:
        fd = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--listen-fd must be an integer, got {value!r}"
        ) from None
    if not (3 <= fd <= 1023):
        raise argparse.ArgumentTypeError(
            f"--listen-fd must be between 3 and 1023, got {fd}"
        )
    return fd


def _apply_body_receive_timeout_env(server_mod, *, logger=None) -> None:
    """Resolve ``FUSION_MLX_BODY_RECEIVE_TIMEOUT_SECONDS`` onto
    ``server_mod._body_receive_timeout_seconds`` (H-14 / F-072
    slow-DoS gate).

    Extracted from ``serve_command`` so the
    ``tests/test_body_receive_timeout.py::test_h14_env_var_override_reduces_timeout``
    case exercises the same code path the production binary runs —
    codex round-2 BLOCKING on PR #786 spotted that an inline-only
    resolver couldn't be unit-tested without duplicating its logic,
    which would silently mask a regression that deleted the wire-up.

    Behaviour:
      * No env var (or empty after strip) → leave the existing
        ``server_mod._body_receive_timeout_seconds`` untouched (the
        module's documented 15 s default).
      * Numeric env value → clamp via ``max(0.0, float(...))`` so
        negative numbers disable the gate without crashing.
      * Non-numeric env value → log a warning and explicitly write
        the 15 s default back to ``server_mod`` (an inherited
        non-default from a prior call would otherwise leak).

    ``server_mod`` is passed in so the test can hand a fresh
    ``fusion_mlx.server`` reference each call without dragging the
    whole import-time CLI prologue along.
    """
    import os

    if logger is None:  # pragma: no cover — tests always pass a logger
        import logging as _logging

        logger = _logging.getLogger(__name__)

    _brt_env_name = "FUSION_MLX_BODY_RECEIVE_TIMEOUT_SECONDS"
    _brt_env = os.environ.get(_brt_env_name, "").strip()
    if not _brt_env:
        return
    try:
        server_mod._body_receive_timeout_seconds = max(0.0, float(_brt_env))
    except ValueError:
        # Interpolate the env-var name via ``%s`` instead of baking it
        # into the format string — same false-positive avoidance
        # pattern as the SSE-keepalive block above.
        logger.warning(
            "%s=%r is not a number; falling back to the 15 s default",
            _brt_env_name,
            _brt_env,
        )
        server_mod._body_receive_timeout_seconds = 15.0


def _wildcard_host_aliases() -> frozenset[str]:
    """Strings that name "bind on every interface" rather than a single
    address. Python's ``socket.bind(("", N))`` and ``socket.bind(("0.0.0.0",
    N))`` are equivalent for IPv4; uvicorn historically treats both the
    empty string and ``0.0.0.0`` the same way. We treat them as a single
    class for the loopback-collision pre-flight (codex round-1 MAJOR on
    PR #848: original gate only matched ``"0.0.0.0"`` so ``--host ""``
    could still re-open the dual-bind ambiguity).

    Kept as a function rather than a module constant so the test suite
    can monkey-patch it in case a future host alias (e.g. ``"::"`` once we
    grow IPv6 pre-flight) needs to land without touching every call site.
    """
    return frozenset({"0.0.0.0", ""})


def _is_ipv6_host(host: str) -> bool:
    """Detect IPv6 literal hosts (``::``, ``::1``, ``2001:db8::1`` ...).

    Codex round-1 MED #6 on PR #855: the IPv4-only preflight always
    created an ``AF_INET`` socket, so any valid uvicorn IPv6 bind
    (``--host ::1``, ``--host ::``, etc.) failed ``socket.bind`` and got
    misreported as "port already in use." Detection is colon-based:
    every IPv6 literal contains at least one ``:``, no IPv4 literal /
    DNS name does (``localhost`` is the canonical non-IPv6 with no
    colon). We deliberately keep this purely lexical — a stricter
    ``ipaddress.ip_address`` parse would reject scoped literals
    (``fe80::1%en0``) that uvicorn happily accepts.
    """
    return ":" in host


def _port_preflight_or_die(host: str, port: int, *, model: str) -> None:
    """Probe ``(host, port)`` AND — when ``host`` is a wildcard alias —
    additionally probe ``("127.0.0.1", port)``. Print a friendly error
    and ``sys.exit(1)`` on the first collision.

    Why both: macOS / Linux let a wildcard listener (``0.0.0.0`` or
    ``""``) coexist with a more-specific loopback listener
    (``127.0.0.1``) on the same port. v0.8.2 dogfood finding #2
    reproduced the resulting PortSweep bypass: ``nc -l 127.0.0.1 11812``
    + ``fusion-mlx serve --port 11812`` BOTH succeed, and
    ``curl 127.0.0.1:11812/healthz`` returns HTTP 000 (kernel routes
    loopback to nc, not fusion-mlx). The fix is to explicitly probe the
    loopback address whenever the requested bind is wider than loopback.

    Extracted from ``serve_command`` so the legacy
    ``python -m fusion_mlx.server`` entrypoint can call it too without
    duplicating the wildcard-alias / probe-loop logic — codex round-1
    MAJOR on PR #848 (the dogfood-CLI fix had to land on both supported
    entrypoints to actually close the bypass).

    ``::1`` is intentionally NOT probed when the user binds an IPv4
    wildcard: macOS treats v4 and v6 loopback as distinct stacks, and
    uvicorn's IPv4 bind never collides with an IPv6 listener. When the
    user EXPLICITLY binds an IPv6 host, we switch the probe family to
    ``AF_INET6`` so the bind doesn't spuriously fail (codex round-1
    MED #6 on PR #855 — pre-fix ``--host ::1`` raised ``OSError`` from
    the ``AF_INET`` socket and was misreported as "port already in use").
    """
    import socket

    wildcards = _wildcard_host_aliases()
    if host in wildcards:
        # Probe the requested wildcard FIRST (so a LAN-side port
        # collision still surfaces the user-supplied host name in the
        # error), then probe 127.0.0.1 to catch the loopback shadow.
        hosts_to_probe: tuple[str, ...] = (host, "127.0.0.1")
    else:
        hosts_to_probe = (host,)

    for probe_host in hosts_to_probe:
        # Pick the address family that matches the host string. IPv6
        # literals (``::``, ``::1``, etc.) need ``AF_INET6`` or the bind
        # raises before we can detect a real collision (codex r1 MED #6
        # on PR #855). Everything else — IPv4 literals, wildcards
        # (``0.0.0.0``, ``""``), the loopback-shadow probe ``127.0.0.1``
        # — stays on ``AF_INET``.
        family = socket.AF_INET6 if _is_ipv6_host(probe_host) else socket.AF_INET
        # ``with`` guarantees the preflight socket is closed on every
        # exit path — including OSError during ``bind``. The previous
        # form called ``_sock.close()`` only on the success branch,
        # which leaked the fd whenever the bind raised (e.g. when
        # running under a test harness that catches ``SystemExit``).
        with socket.socket(family, socket.SOCK_STREAM) as _sock:
            _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                _sock.bind((probe_host, port))
            except OSError:
                # Surface the host we actually collided on so the user
                # can distinguish "LAN port busy" from "loopback port
                # already claimed by another fusion-mlx / nc / proxy".
                # Use the empty-string-friendly display name so
                # ``--host ""`` shows up as ``0.0.0.0`` rather than a
                # confusing bare quote.
                display_host = probe_host or "0.0.0.0"
                print(f"\n  Error: Port {port} is already in use on {display_host}.")
                print(
                    f"  Try a different port: fusion-mlx serve {model} --port {port + 1}"
                )
                sys.exit(1)


def _print_port_collision_and_exit(
    host: str, port: int, *, in_listen_fd_mode: bool
) -> None:
    """Print a Sven-style supervisor-friendly EADDRINUSE message to
    stderr and ``sys.exit(1)``. Single SSOT so both the host/port and
    ``--listen-fd`` failure paths emit a consistent operator-facing
    message and the exit code stays non-zero in both.

    In ``--listen-fd`` mode the ``host``/``port`` args don't describe
    the real bind (the supervisor owns it), so we omit the
    port-specific ``lsof -i :N`` hint and reference the inherited fd
    instead — otherwise the operator would chase a port the fusion-mlx
    process never tried to bind (codex round-1 NIT #3).
    """
    if in_listen_fd_mode:
        print(
            "\n  Error: bind() failed on the supervisor-provided "
            "--listen-fd. The inherited socket is unusable. Re-launch "
            "with a fresh socket activation or fall back to --host/--port.",
            file=sys.stderr,
        )
    else:
        display_host = host or "0.0.0.0"
        print(
            f"\n  Error: Port {port} already in use on {display_host}. "
            f"Choose a different --port or stop the existing server "
            f"(lsof -i :{port}).",
            file=sys.stderr,
        )
    sys.exit(1)


def _run_uvicorn(app, args, log_level: str) -> None:
    """Dispatch into ``uvicorn.run`` with the kwargs that match the
    current ``--listen-fd`` / ``--host``/``--port`` mode.

    Extracted so the call-site contract is unit-testable WITHOUT booting
    the heavy ``serve_command`` prologue (version check, model download,
    server import). The companion bytecode test in
    ``tests/test_serve_listen_fd.py`` pins that ``serve_command``
    actually references this helper so a future refactor that drops the
    dispatch silently is caught — that's the regression-detection codex
    round-1 PR #696 review was after.

    R13 Sven B1: also the single CLI-side chokepoint that converts a
    uvicorn-side bind failure into the friendly "Port N already in
    use…" message + ``sys.exit(1)`` the operator's supervisor (systemd,
    launchd, k8s) needs to detect failure. Three paths feed in:

      * ``OSError(EADDRINUSE)`` raised through uvicorn (older uvicorns,
        ``--listen-fd`` mode where ``socket.fromfd`` / ``create_server``
        fail before uvicorn's own except arms) — caught directly.
      * ``SystemExit(1)`` from uvicorn>=0.34: ``Server.startup`` catches
        the bind ``OSError``, ``logger.error(exc)``s it (raw
        ``ERROR: [Errno 48] …``), and ``sys.exit(1)``s before our
        ``except OSError`` can fire. The exit code is already non-zero,
        but the friendly hint is missing — so we re-detect by probing
        the same ``(host, port)`` ourselves and, if it's busy, re-emit
        the Sven-style message before propagating the same non-zero
        exit (codex round-1 BLOCKING #2).
      * Any other ``SystemExit`` from uvicorn (clean lifespan shutdown,
        TLS misconfig, etc.) is left untouched.

    ``_port_preflight_or_die`` (run earlier in ``serve_command``)
    handles the common pre-load case at zero cost — this layer is the
    TOCTOU-race / fd-mode safety net.
    """
    import errno

    import uvicorn

    listen_fd = getattr(args, "listen_fd", None)
    try:
        if listen_fd is not None:
            # ``fd=`` overrides ``host``/``port``: uvicorn skips its own
            # ``socket.bind()`` and adopts the inherited fd directly. This
            # is the close of the bind→auth TOCTOU window — the supervisor
            # bound + validated the auth secret BEFORE execve'ing, and the
            # FastAPI ``app`` (with route auth dependencies) is fully
            # constructed at module load before this call.
            uvicorn.run(
                app,
                fd=listen_fd,
                log_level=log_level,
                timeout_keep_alive=30,
            )
        else:
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                log_level=log_level,
                timeout_keep_alive=30,
            )
    except OSError as exc:
        # Direct EADDRINUSE — older uvicorn, ``--listen-fd`` mode bind
        # path. Translate to the friendly message; unrelated OSErrors
        # (e.g. EACCES on a low port) keep their original trace and
        # propagate so the failure is debuggable.
        if exc.errno == errno.EADDRINUSE:
            _print_port_collision_and_exit(
                args.host, args.port, in_listen_fd_mode=listen_fd is not None
            )
        raise
    except SystemExit as exc:
        # uvicorn>=0.34 catches the bind ``OSError`` in ``Server.startup``,
        # ``logger.error(exc)``s it (raw ``[Errno 48]`` line — not the
        # friendly hint a supervisor operator needs), and ``sys.exit(1)``s
        # before our ``except OSError`` can fire. The exit code is
        # already non-zero so the supervisor-failure-detection contract
        # holds, but we re-emit the Sven-style message on top so the
        # operator's grep for "already in use" still hits. Only override
        # the message when a probe confirms the port really IS in use —
        # other ``SystemExit(1)`` paths (TLS, lifespan, etc.) must keep
        # uvicorn's own diagnostic so we don't paper over them.
        #
        # Outer guard: codex round-2 BLOCKING — if the probe itself
        # raises (TypeError from a non-string host, gaierror, etc.) the
        # caller's ``SystemExit`` MUST still propagate. Wrap the
        # discriminator call so any probe-side exception is silently
        # absorbed and the original ``raise`` below re-delivers
        # uvicorn's exit. ``_port_is_busy`` ALSO defends internally,
        # but a future refactor that drops that guard (or a monkeypatch
        # in a test harness) must not corrupt the failure signal.
        if exc.code in (1, "1") and listen_fd is None:
            try:
                busy = _port_is_busy(args.host, args.port)
            except BaseException:
                busy = False
            if busy:
                _print_port_collision_and_exit(
                    args.host, args.port, in_listen_fd_mode=False
                )
        raise


def _port_is_busy(host: str, port: int) -> bool:
    """Best-effort probe: is ``(host, port)`` already bound by another
    process? Used by ``_run_uvicorn`` to disambiguate an uvicorn
    ``SystemExit(1)`` triggered by a bind collision from one triggered
    by an unrelated startup failure (TLS, lifespan, etc.).

    Returns True iff a fresh ``socket.bind`` fails with EADDRINUSE.
    Returns False on ANY other outcome (clean bind, ENETDOWN, EACCES,
    ``gaierror``, ``TypeError`` from a ``None`` host, etc.) so the
    caller's original ``SystemExit`` propagates untouched — codex
    round-2 BLOCKING was that a probe-side ``TypeError`` could replace
    uvicorn's ``SystemExit(1)`` with a misleading traceback. The probe
    is a HEURISTIC: a false-negative is acceptable (operator still
    gets uvicorn's diagnostic + non-zero exit, just no friendly hint),
    a probe-side raise that masks uvicorn's failure is not.
    """
    import errno
    import socket

    if not isinstance(host, str) or not host:
        # uvicorn accepts ``host=""`` as the wildcard alias, but
        # ``socket.bind(("", port))`` works on AF_INET — pre-normalize
        # to avoid a probe-side ``TypeError`` for non-string hosts
        # (sentinel values, configured-via-env edge cases).
        host = "0.0.0.0"

    try:
        family = socket.AF_INET6 if _is_ipv6_host(host) else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError as exc:
                return exc.errno == errno.EADDRINUSE
    except BaseException:
        # Outer guard: ANY probe-side failure (socket constructor,
        # gaierror, host normalization, etc.) MUST NOT mask the caller's
        # ``SystemExit``. Swallow and report "not busy" so the original
        # exception re-raises cleanly. ``BaseException`` is intentional
        # — even a stray ``KeyboardInterrupt`` during the probe should
        # not corrupt the supervisor-facing failure signal; the caller's
        # ``raise`` will re-deliver any interrupt on the next event loop.
        return False
    return False


def _chat_config_dir() -> str:
    """Directory for first-launch tip markers (and future per-user chat
    state). Honors ``FUSION_MLX_CONFIG_HOME`` override; otherwise falls back
    to ``~/.config/fusion-mlx``. The directory is created lazily by the
    writer; callers don't need to ensure it exists for reads.
    """
    override = os.environ.get("FUSION_MLX_CONFIG_HOME")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".config", "fusion-mlx")


def _seen_tips_path() -> str:
    return os.path.join(_chat_config_dir(), "seen-tips.json")


def _has_seen_tip(key: str) -> bool:
    """Return True iff the marker file records ``key: true``.

    Any IO/parse error is treated as "not seen" — better to show the tip
    one extra time than to hide it forever on a corrupt marker.
    """
    import json

    try:
        with open(_seen_tips_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and bool(data.get(key))


def _mark_tip_seen(key: str) -> None:
    """Persist ``key: true`` to the seen-tips marker. Best-effort —
    failures are swallowed so a read-only config dir never aborts chat.
    """
    import json

    path = _seen_tips_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        return
    try:
        existing: dict = {}
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, ValueError):
            existing = {}
        existing[key] = True
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh)
    except OSError:
        return


def _print_unknown_model_help(name: str, *, full_path_example: str) -> None:
    """Print fuzzy suggestions + a curated popular-models hint.

    Replaces the older "Did you mean: X?" + "Run `fusion-mlx models`" pattern
    that left users empty-handed when no close fuzzy match existed
    (e.g. ``fusion-mlx chat gemma4-27b`` returned zero suggestions, told the
    user to run another command, and gave no hint of what was actually
    supported). Now: always show *something* — fuzzy matches when we have
    them, curated popular aliases when we don't.
    """
    from fusion_mlx.model_aliases import POPULAR_ALIASES, list_aliases, suggest_similar

    suggestions = suggest_similar(name)
    if suggestions:
        print(f"  Did you mean: {', '.join(suggestions)}?")
    else:
        print(f"  Try one of: {', '.join(POPULAR_ALIASES)}")
    print(f"  Run `fusion-mlx models` to see all {len(list_aliases())} aliases,")
    print(f"  or pass a full path like: {full_path_example}")


def _embedding_not_found_exception_classes() -> tuple[type[BaseException], ...]:
    """Return the concrete exception classes the embedding loader raises
    for a missing model.

    pr_validate codex r1 NIT: matching ``"not found"`` as a substring of
    the exception text was too loose — a future ``ValueError("config
    field 'x' not found in tensor map")`` from a corrupt model could be
    mis-translated as the alias/HF-id hint, masking the real bug. Bind
    to the actual classes so the wrap-path only fires on the
    well-defined not-found shape.

    Lazy import so the base install (no ``[embeddings]`` extra, no
    ``huggingface_hub`` shadow) stays free of these imports until the
    code path actually runs. Missing classes are silently skipped — the
    caller's tuple-based ``except`` accepts an empty tuple as a no-op,
    so a sparse environment falls back to "re-raise everything", which
    is the safe default.
    """
    classes: list[type[BaseException]] = [FileNotFoundError]
    try:  # mlx_embeddings — installed via the [embeddings] extra
        from mlx_embeddings.utils import ModelNotFoundError

        classes.append(ModelNotFoundError)
    except Exception:  # pragma: no cover — defensive
        pass
    try:  # huggingface_hub — transitive of mlx_embeddings
        from huggingface_hub.errors import (
            EntryNotFoundError,
            RepositoryNotFoundError,
        )

        classes.append(RepositoryNotFoundError)
        classes.append(EntryNotFoundError)
    except Exception:  # pragma: no cover — defensive
        pass
    return tuple(classes)


def _resolve_embedding_alias(name: str) -> tuple[str, bool]:
    """Resolve a ``--embedding-model`` alias through the shared registry.

    D-EMBED-ALIAS: Sarah F-S2-1 — the positional chat-model arg goes
    through ``resolve_model`` at the CLI dispatch (cli.py ~5660), but
    the ``--embedding-model`` flag was passed verbatim to
    ``mlx_embeddings.load`` and crashed with ``ModelNotFoundError`` on
    any alias.

    Returns ``(resolved, did_resolve)``. ``did_resolve`` is True when
    the registry actually mapped ``name`` to a different HF path —
    used by the caller to log the alias hop.
    """
    from .model_aliases import resolve_model

    resolved = resolve_model(name)
    return resolved, resolved != name


def _resolve_audio_model_for_serve(model_name: str):
    """Resolve a model name to an audio registry entry, if it's audio.

    R10-C1: pre-fix ``serve_command`` had a boot guard (rc=2 when
    ``[audio]`` extra missing) but ZERO resolution logic for audio
    aliases. Short aliases like ``kokoro``/``whisper`` then fell into
    ``_ensure_model_downloaded`` and 404'd at HF, while full HF ids
    of audio models (``mlx-community/Kokoro-82M-bf16``) downloaded
    successfully but crashed in ``mlx_lm.load_model`` because they
    have no safetensors. Bo r10-R1: 0/8 audio aliases boot on 0.8.11.

    The fix routes audio names through a SEPARATE serve path that
    skips the text-model loader entirely. This helper returns the
    resolved registry entry (so the dispatcher knows the HF id, type,
    family, voice list) or ``None`` if the name isn't audio. ``None``
    falls through to the legacy text path unchanged — text-model boot
    paths must not regress.
    """
    from .audio.registry import resolve_audio_alias

    return resolve_audio_alias(model_name)
