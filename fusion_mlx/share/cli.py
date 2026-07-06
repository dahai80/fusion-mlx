# SPDX-License-Identifier: Apache-2.0
"""``fusion-mlx share <alias>`` — start a serve + open a public tunnel.

Orchestration shape:

  1. Validate alias (cheap fail-fast before booting the engine).
  2. Pick a free local port + generate a fresh 24-byte bearer key.
  3. Spawn ``fusion-mlx serve`` in a child process pointing at that port.
  4. Wait for /healthz to come back ready, then auth-gate /v1/models.
  5. Open a WebSocket to the rapidserver Worker (defaults to
     ``wss://rapidserver.quicksilverpro.io/up``). The Worker mints a
     Durable Object keyed on our tunnel id; inbound HTTPS requests at
     ``https://rapidserver.quicksilverpro.io/r/<id>/...`` are
     reverse-multiplexed back to us over the same WS frame.
  6. Probe ``<public_url>/v1/models`` to prove the tunnel ↔ serve
     round-trip works, then print the security banner + URL + key.
  7. Block until Ctrl-C, monitoring both the serve subprocess and the
     WS tunnel thread.
  8. On exit, close the WS first (cheap) then terminate serve.

State lives in ``~/.cache/fusion-mlx/share/`` — pid + serve log only.
Key + URL are NOT persisted: each invocation issues a new key
and a new session.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .._completion import alias_completer
from . import warning, ws_tunnel

logger = logging.getLogger(__name__)

_PORT_ENV_VAR = "FUSION_MLX_SHARE_PORT"
_CHAT_FRONTEND_ENV_VAR = "FUSION_MLX_CHAT_FRONTEND"
_DEFAULT_CHAT_FRONTEND = "https://rapid-pro.quicksilverpro.io"


def _resolve_chat_frontend(flag_value: str | None) -> str | None:
    if flag_value is not None:
        raw = flag_value
    else:
        raw = os.environ.get(_CHAT_FRONTEND_ENV_VAR)
        if raw is None:
            raw = _DEFAULT_CHAT_FRONTEND
    raw = raw.strip()
    if not raw:
        return None
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"--chat-frontend must use https:// or http:// (got {raw!r})")
    if (
        parsed.username is not None
        or parsed.password is not None
        or "@" in (parsed.netloc or "")
    ):
        raise ValueError(f"--chat-frontend must not include userinfo (got {raw!r})")
    host = parsed.hostname
    if not host:
        raise ValueError(f"--chat-frontend must include a host (got {raw!r})")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"--chat-frontend has an invalid port (got {raw!r})") from exc
    if parsed.scheme == "http":
        try:
            host_is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            host_is_loopback = host == "localhost"
        if not host_is_loopback:
            raise ValueError(
                f"--chat-frontend over plain http:// only allowed for "
                f"loopback hosts (got {raw!r})"
            )
    if parsed.path not in ("", "/"):
        raise ValueError(
            f"--chat-frontend must be an origin without a path (got {raw!r})"
        )
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"--chat-frontend must not include a query or fragment (got {raw!r})"
        )
    host_part = f"[{host}]" if ":" in host else host
    authority = f"{host_part}:{port}" if port is not None else host_part
    return f"{parsed.scheme}://{authority}"


_DEFAULT_CORS_ALLOWLIST: tuple[str, ...] = (
    "https://rapid-pro.pages.dev",
    "https://rapid-pro.quicksilverpro.io",
    "https://rapidmlx.com",
    "https://chat.rapidmlx.com",
)


def _resolve_cors_origins(
    flag_value: list[str] | None,
    chat_frontend: str | None,
) -> list[str]:
    if flag_value:
        return list(flag_value)
    origins = list(_DEFAULT_CORS_ALLOWLIST)
    if chat_frontend and chat_frontend not in origins:
        origins.append(chat_frontend)
    return origins


def _pick_port(preferred: int) -> int:
    for candidate in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", candidate))
                return s.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError("no free port available for share")


def _resolve_served_model_name(port: int, api_key: str) -> str | None:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            payload = json.load(r)
        data = payload.get("data") or []
        if data and isinstance(data[0], dict):
            return data[0].get("id")
    except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError):
        return None
    return None


def _wait_for_healthz(port: int, serve_proc: subprocess.Popen[bytes]) -> bool:
    url = f"http://127.0.0.1:{port}/healthz"
    while True:
        if serve_proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)


def _verify_auth_gate(port: int, api_key: str) -> bool:
    def _probe(bearer: str) -> int | None:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/models",
            headers={"Authorization": f"Bearer {bearer}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status
        except urllib.error.HTTPError as exc:
            return exc.code
        except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError):
            return None

    bad_status = _probe(secrets.token_hex(24))
    if bad_status == 200:
        return False
    return _probe(api_key) == 200


def _spawn_serve(
    *,
    alias: str,
    port: int,
    api_key: str,
    log_path: Path,
    extra_args: list[str],
) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "fusion_mlx.cli",
        "serve",
        alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "INFO",
        *extra_args,
    ]
    env = dict(os.environ)
    env["FUSION_MLX_API_KEY"] = api_key
    env["FUSION_MLX_WATCHDOG_PPID"] = str(os.getpid())
    log_fp = log_path.open("ab", buffering=0)
    try:
        Path(log_fp.name).chmod(0o600)
    except OSError:
        pass
    return subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )


def _state_dir() -> Path:
    d = Path.home() / ".cache" / "fusion-mlx" / "share"
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


def _maybe_confirm_download(alias: str) -> None:
    if "/" not in alias or os.path.exists(alias):
        return
    if os.environ.get("FUSION_MLX_CHAT_SPAWN", "") == "1":
        return
    env_val = os.environ.get("FUSION_MLX_AUTO_PULL", "").strip().lower()
    if env_val in {"1", "true", "yes"}:
        return
    if not sys.stdin.isatty():
        return
    from fusion_mlx._download_gate import (
        confirm_or_abort,
        estimate_repo_size_bytes,
        is_repo_cached,
    )

    if not is_repo_cached(alias):
        confirm_or_abort(alias, estimate_repo_size_bytes(alias))


def share_command(args: argparse.Namespace) -> None:
    alias: str = getattr(args, "_original_alias", None) or args.model
    _maybe_confirm_download(args.model)

    try:
        chat_frontend = _resolve_chat_frontend(args.chat_frontend)
    except ValueError as exc:
        print(f"share: {exc}", file=sys.stderr)
        sys.exit(2)

    extra_serve_args: list[str] = []
    if not args.thinking:
        extra_serve_args.append("--no-thinking")

    origins = _resolve_cors_origins(args.cors_origins, chat_frontend)
    extra_serve_args.append("--cors-origins")
    extra_serve_args.extend(origins)

    if args.rate_limit > 0:
        extra_serve_args.append("--rate-limit")
        extra_serve_args.append(str(args.rate_limit))

    api_key = secrets.token_hex(24)
    raw_port = os.environ.get(_PORT_ENV_VAR) if args.port is None else None
    try:
        if raw_port is not None:
            preferred_port = int(raw_port)
        elif args.port is not None:
            preferred_port = args.port
        else:
            preferred_port = 8765
    except ValueError:
        print(
            f"{_PORT_ENV_VAR} must be an integer (got {raw_port!r})",
            file=sys.stderr,
        )
        sys.exit(2)
    if not (1 <= preferred_port <= 65535):
        print(
            f"share port {preferred_port} is outside the valid range (1-65535)",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        port = _pick_port(preferred_port)
    except RuntimeError as exc:
        print(f"share: {exc}", file=sys.stderr)
        sys.exit(1)
    state_dir = _state_dir()
    serve_log = state_dir / "serve.log"

    relay_url = os.environ.get("FUSION_MLX_RELAY_URL", ws_tunnel.DEFAULT_RAPIDSERVER_WSS)
    if not (relay_url.startswith("wss://") or relay_url.startswith("ws://")):
        print(
            f"share: FUSION_MLX_RELAY_URL must start with wss:// or ws:// "
            f"(got {relay_url!r})",
            file=sys.stderr,
        )
        sys.exit(2)

    def _term_handler(signum, frame):
        raise KeyboardInterrupt

    original_sigterm = signal.signal(signal.SIGTERM, _term_handler)

    serve_proc: subprocess.Popen[bytes] | None = None
    tunnel: ws_tunnel.TunnelClient | None = None
    tunnel_thread = None
    serve_exit_code = 0
    try:
        logger.info("Starting fusion-mlx serve (%s on :%d)", alias, port)
        print(f"Starting fusion-mlx serve ({alias} on :{port})…", file=sys.stderr)
        serve_proc = _spawn_serve(
            alias=alias,
            port=port,
            api_key=api_key,
            log_path=serve_log,
            extra_args=extra_serve_args,
        )
        if not _wait_for_healthz(port, serve_proc):
            print(
                f"serve exited before becoming ready — see {serve_log}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not _verify_auth_gate(port, api_key):
            print(
                f"serve on :{port} did not answer authenticated /v1/models — "
                f"another process may be bound to the same port. Aborting "
                f"before opening a public tunnel.",
                file=sys.stderr,
            )
            sys.exit(1)

        logger.info("Connecting to relay %s", relay_url)
        print(f"Connecting to relay {relay_url}…", file=sys.stderr)
        tunnel = ws_tunnel.TunnelClient(local_port=port, relay_url=relay_url)
        tunnel_thread = tunnel.run_in_thread()
        if not tunnel.ready_event.wait(timeout=30):
            err = tunnel.error
            print(
                f"share: WS tunnel did not connect to {relay_url} within 30s",
                file=sys.stderr,
            )
            if err is not None:
                print(f"   reason: {err}", file=sys.stderr)
            sys.exit(1)
        if tunnel.error is not None:
            print(f"share: WS tunnel failed: {tunnel.error}", file=sys.stderr)
            sys.exit(1)

        if not ws_tunnel.wait_for_public_url(tunnel.public_url, api_key, timeout=30):
            print(
                f"share: public URL {tunnel.public_url} did not respond within 30s",
                file=sys.stderr,
            )
            sys.exit(1)

        display_model = _resolve_served_model_name(port, api_key) or alias
        print(
            warning.render(
                tunnel.public_url,
                api_key,
                display_model,
                tunnel.tunnel_id,
                chat_frontend,
            ),
            flush=True,
        )

        while True:
            serve_rc = serve_proc.poll()
            if serve_rc is not None:
                serve_exit_code = serve_rc if serve_rc != 0 else 1
                if serve_rc == 0:
                    print(
                        f"share: serve process exited cleanly but the "
                        f"public share is no longer live — see {serve_log}.",
                        file=sys.stderr,
                    )
                break
            if tunnel.closed_event.is_set():
                err = tunnel.error
                suffix = f": {err}" if err is not None else ""
                print(
                    f"share: WS tunnel disconnected{suffix}. Stopping serve.",
                    file=sys.stderr,
                )
                serve_exit_code = 1
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping share…", file=sys.stderr)
    finally:
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
        if tunnel is not None:
            tunnel.stop()
        if tunnel_thread is not None and tunnel_thread.is_alive():
            tunnel_thread.join(timeout=5)
        if serve_proc is not None and serve_proc.poll() is None:
            try:
                serve_proc.terminate()
                serve_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                serve_proc.kill()
            except OSError:
                pass
        try:
            signal.signal(signal.SIGTERM, original_sigterm)
        except (ValueError, OSError, TypeError):
            pass

    if serve_exit_code:
        sys.exit(1)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "share",
        help="Expose a local model behind a public URL via rapidmlx.com",
        description=(
            "Start fusion-mlx serve and open a public Cloudflare-fronted "
            "URL on rapidmlx.com so you can use the model from a different "
            "device — or share it with a friend. Press Ctrl-C to stop."
        ),
    )
    p.add_argument(
        "model",
        help="Alias to serve (same names as `fusion-mlx serve`, e.g. qwen3.5-4b-4bit)",
    ).completer = alias_completer
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Local port to bind serve to (default: 8765, or "
            "$FUSION_MLX_SHARE_PORT if set)"
        ),
    )
    p.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward thinking-mode behavior to serve. Default off "
            "(``--no-thinking``) so chat UIs see content immediately "
            "instead of waiting on a thinking prelude. Pass ``--thinking`` "
            "to keep upstream defaults."
        ),
    )
    p.add_argument(
        "--cors-origins",
        nargs="+",
        default=None,
        metavar="ORIGIN",
        help=(
            "Pass --cors-origins to serve. Accepts multiple values, same "
            "shape as ``fusion-mlx serve --cors-origins``. Default: the "
            "rapidmlx chat-frontend allowlist (rapid-pro.pages.dev, "
            "rapid-pro.quicksilverpro.io, rapidmlx.com, chat.rapidmlx.com) "
            "plus whatever ``--chat-frontend`` resolves to. Pass '*' to "
            "relax for browser chat UIs you host elsewhere (e.g. local "
            "Open WebUI). Example: --cors-origins http://localhost:3000."
        ),
    )
    p.add_argument(
        "--rate-limit",
        type=int,
        default=120,
        metavar="RPM",
        help=(
            "Per-client requests/minute cap forwarded to the spawned "
            "``fusion-mlx serve``. Default: 120 (2/sec) — high enough for "
            "tool-using power users and Beam-mode parallel completions, "
            "low enough that a leaked share key can't burst-DoS the "
            "publisher's machine. Set 0 to disable the cap entirely."
        ),
    )
    p.add_argument(
        "--chat-frontend",
        type=str,
        default=None,
        metavar="URL",
        help=(
            "Override the one-click chat link printed in the share banner. "
            "Default: https://rapid-pro.quicksilverpro.io (or $FUSION_MLX_CHAT_FRONTEND "
            "if set). The frontend must implement the rapidmlx splash "
            "share-key protocol — point this at your own fork if you host "
            "one. Pass an empty string ('') to suppress the chat link "
            "entirely (useful for OpenWebUI and other frontends that don't "
            "speak the splash protocol; the URL+Key lines below still let "
            "you wire it up by hand)."
        ),
    )
