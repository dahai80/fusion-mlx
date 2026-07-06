# SPDX-License-Identifier: Apache-2.0
"""Pure-Python WebSocket reverse tunnel for ``fusion-mlx share``.

Connects to the rapidserver Worker (defaults to
``wss://rapidserver.quicksilverpro.io/up``), receives HTTP requests
reverse-multiplexed over a single WebSocket, forwards them to the
local ``fusion-mlx serve`` on ``127.0.0.1``, and streams responses back.

Protocol (JSON text frames):

    worker → client:
        {"t":"req", "id":<reqId>, "method":<str>, "path":<str>,
         "headers":<obj>, "body":<base64>}
        {"t":"abort", "id":<reqId>}

    client → worker:
        {"t":"ready", "v":1}                              (sent once on connect)
        {"t":"head", "id":<reqId>, "status":<int>, "headers":<obj>}
        {"t":"chunk", "id":<reqId>, "data":<base64>}
        {"t":"end", "id":<reqId>}
        {"t":"err", "id":<reqId>, "msg":<str>}

One WS connection multiplexes many concurrent HTTP requests via the
``id`` field.
"""

from __future__ import annotations

import asyncio
import base64
import http.client
import json
import logging
import secrets
import threading
import time
import urllib.parse
from collections.abc import Callable
from typing import Any

try:
    import websockets
except ImportError as exc:
    raise ImportError(
        "fusion-mlx share requires the ``websockets`` package "
        "(pip install websockets if missing)"
    ) from exc

log = logging.getLogger(__name__)


DEFAULT_RAPIDSERVER_WSS = "wss://rapidserver.quicksilverpro.io/up"

LOCAL_FETCH_TIMEOUT_SECONDS = 1800

CHUNK_SIZE = 4096


def new_tunnel_id() -> str:
    return secrets.token_urlsafe(16)


def public_url_for(tunnel_id: str, relay_url: str = DEFAULT_RAPIDSERVER_WSS) -> str:
    parsed = urllib.parse.urlparse(relay_url)
    scheme = {"wss": "https", "ws": "http"}.get(parsed.scheme, "https")
    return f"{scheme}://{parsed.netloc}/r/{tunnel_id}"


class TunnelClient:

    def __init__(
        self,
        *,
        local_port: int,
        tunnel_id: str | None = None,
        relay_url: str = DEFAULT_RAPIDSERVER_WSS,
        ready_event: threading.Event | None = None,
    ) -> None:
        self.local_port = local_port
        self.tunnel_id = tunnel_id or new_tunnel_id()
        self.relay_url = relay_url
        self.ready_event = ready_event or threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._send_queue: asyncio.Queue[str] | None = None
        self._closed = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()
        self.error: BaseException | None = None
        self.closed_event = threading.Event()

    @property
    def public_url(self) -> str:
        return public_url_for(self.tunnel_id, self.relay_url)

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._send_queue = asyncio.Queue()
        uri = f"{self.relay_url}?id={self.tunnel_id}"
        try:
            async with websockets.connect(uri, max_size=None) as ws:
                await ws.send(json.dumps({"t": "ready", "v": 1}))
                self.ready_event.set()
                sender = asyncio.create_task(self._sender_loop(ws))
                try:
                    async for raw in ws:
                        if not isinstance(raw, str):
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        self._dispatch_inbound(msg)
                finally:
                    sender.cancel()
                    self._closed.set()
                    for t in list(self._tasks):
                        t.cancel()
        except Exception as exc:
            self.error = exc
            raise
        finally:
            self.closed_event.set()

    def run_in_thread(self) -> threading.Thread:
        def _entry() -> None:
            try:
                asyncio.run(self.run())
            except Exception as exc:
                if self.error is None:
                    self.error = exc

        t = threading.Thread(
            target=_entry,
            name="fusion-mlx-share-ws-tunnel",
            daemon=True,
        )
        t.start()
        return t

    def stop(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._closed.set)
        except RuntimeError:
            pass

    async def _sender_loop(self, ws: Any) -> None:
        assert self._send_queue is not None
        while True:
            msg = await self._send_queue.get()
            try:
                await ws.send(msg)
            except Exception:
                return

    def _dispatch_inbound(self, msg: dict[str, Any]) -> None:
        t = msg.get("t")
        if t == "req":
            task = asyncio.create_task(self._handle_request(msg))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        elif t == "abort":
            pass

    async def _handle_request(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        if not isinstance(req_id, str):
            return
        method = msg.get("method", "GET")
        path = msg.get("path", "/")
        headers: dict[str, str] = msg.get("headers") or {}
        body_b64: str = msg.get("body") or ""
        try:
            body = base64.b64decode(body_b64) if body_b64 else b""
        except (ValueError, TypeError) as exc:
            await self._send(
                {"t": "err", "id": req_id, "msg": f"bad body encoding: {exc}"}
            )
            return

        try:
            await asyncio.to_thread(
                self._perform_local_fetch, req_id, method, path, headers, body
            )
        except Exception as exc:
            await self._send({"t": "err", "id": req_id, "msg": str(exc)[:200]})

    def _perform_local_fetch(
        self,
        req_id: str,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.local_port, timeout=LOCAL_FETCH_TIMEOUT_SECONDS
        )
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            self._sync_send(
                {
                    "t": "head",
                    "id": req_id,
                    "status": resp.status,
                    "headers": dict(resp.getheaders()),
                }
            )
            while True:
                chunk = resp.read1(CHUNK_SIZE)
                if not chunk:
                    break
                self._sync_send(
                    {
                        "t": "chunk",
                        "id": req_id,
                        "data": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            self._sync_send({"t": "end", "id": req_id})
        finally:
            conn.close()

    async def _send(self, obj: Any) -> None:
        if self._send_queue is None:
            return
        await self._send_queue.put(json.dumps(obj))

    def _sync_send(self, obj: Any) -> None:
        loop = self._loop
        q = self._send_queue
        if loop is None or q is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(q.put_nowait, json.dumps(obj))
        except RuntimeError:
            pass


def wait_for_public_url(
    public_url: str,
    bearer: str,
    *,
    timeout: float = 30.0,
    log_fn: Callable[[str], None] = log.debug,
) -> bool:
    import urllib.error
    import urllib.request

    url = public_url.rstrip("/") + "/v1/models"
    deadline = time.monotonic() + timeout
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "fusion-mlx-share",
            "Authorization": f"Bearer {bearer}",
        },
    )
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except urllib.error.HTTPError as exc:
            log_fn(f"probe got HTTP {exc.code}")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    return False
