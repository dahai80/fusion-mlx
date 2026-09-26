# SPDX-License-Identifier: Apache-2.0
import base64
import json
import logging
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

_TTL_SECONDS = 3600
_MAX_ENTRIES = 1000


class _StoredResponse:
    __slots__ = (
        "response_id",
        "model",
        "input_items",
        "created_at",
        "status",
        "output_text",
    )

    def __init__(self, response_id, model, input_items):
        self.response_id = response_id
        self.model = model
        self.input_items = input_items
        self.created_at = int(time.time())
        self.status = "in_progress"
        self.output_text = ""


class ResponseStore:
    def __init__(
        self, ttl_seconds: int = _TTL_SECONDS, max_entries: int = _MAX_ENTRIES
    ):
        self._store: dict[str, _StoredResponse] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max = max_entries

    def put(
        self, response_id: str, model: str, input_items: list[Any]
    ) -> _StoredResponse:
        with self._lock:
            self._evict_locked()
            entry = _StoredResponse(response_id, model, input_items)
            self._store[response_id] = entry
            logger.info(
                "responses_store put id=%s model=%s items=%d",
                response_id,
                model,
                len(input_items),
            )
            return entry

    def get(self, response_id: str) -> _StoredResponse | None:
        with self._lock:
            entry = self._store.get(response_id)
            if entry is None:
                return None
            if time.time() - entry.created_at > self._ttl:
                del self._store[response_id]
                logger.info("responses_store expired id=%s", response_id)
                return None
            return entry

    def complete(self, response_id: str, output_text: str = "") -> None:
        with self._lock:
            entry = self._store.get(response_id)
            if entry is not None:
                entry.status = "completed"
                entry.output_text = output_text
                # append the assistant turn so a follow-up via
                # previous_response_id sees the full dialogue, not just
                # the user side (mlx-serve chain parity).
                if output_text:
                    entry.input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": output_text,
                            "status": "completed",
                        }
                    )

    def delete(self, response_id: str) -> bool:
        with self._lock:
            if response_id in self._store:
                del self._store[response_id]
                logger.info("responses_store delete id=%s", response_id)
                return True
            return False

    def _evict_locked(self) -> None:
        now = time.time()
        expired = [
            rid for rid, e in self._store.items() if now - e.created_at > self._ttl
        ]
        for rid in expired:
            del self._store[rid]
        if len(self._store) >= self._max:
            for rid, _ in sorted(self._store.items(), key=lambda kv: kv[1].created_at)[
                : len(self._store) - self._max + 1
            ]:
                del self._store[rid]

    def size(self) -> int:
        with self._lock:
            return len(self._store)


_store = ResponseStore()


def get_response_store() -> ResponseStore:
    return _store


def store_response_input(response_id: str, model: str, input_items: list[Any]) -> None:
    _store.put(response_id, model, input_items)


def complete_response(response_id: str, output_text: str = "") -> None:
    _store.complete(response_id, output_text)


def load_previous_input(response_id: str) -> list[Any] | None:
    entry = _store.get(response_id)
    if entry is None:
        return None
    return entry.input_items


class CompactRequest(BaseModel):
    model: str | None = Field(None, description="model id (informational)")
    input: list[Any] | str = Field(..., description="input items or text to compact")
    instructions: str | None = None


@router.post("/v1/responses/compact")
async def compact_responses(req: CompactRequest, _auth: bool = Depends(verify_api_key)):
    if isinstance(req.input, str):
        items = [{"type": "message", "role": "user", "content": req.input}]
    else:
        items = req.input
    blob = base64.b64encode(
        json.dumps(items, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    logger.info(
        "responses/compact model=%s items=%d blob_len=%d",
        req.model,
        len(items),
        len(blob),
    )
    return JSONResponse(
        {
            "object": "response.compaction",
            "model": req.model,
            "compaction": blob,
            "item_count": len(items),
        }
    )


@router.get("/v1/responses/{response_id}")
async def get_response(response_id: str, _auth: bool = Depends(verify_api_key)):
    entry = _store.get(response_id)
    if entry is None:
        raise HTTPException(404, f"response '{response_id}' not found or expired")
    return JSONResponse(
        {
            "id": entry.response_id,
            "object": "response",
            "model": entry.model,
            "created_at": entry.created_at,
            "status": entry.status,
            "input": entry.input_items,
            "output_text": entry.output_text,
        }
    )


@router.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str, _auth: bool = Depends(verify_api_key)):
    removed = _store.delete(response_id)
    if not removed:
        raise HTTPException(404, f"response '{response_id}' not found")
    return JSONResponse({"id": response_id, "object": "response", "deleted": True})


@router.websocket("/v1/responses/ws")
async def responses_websocket(ws: WebSocket):
    """Responses API over WebSocket (mlx-serve parity).

    Each inbound text frame is a ``response.create`` JSON message (the same
    body POST /v1/responses accepts). The handler delegates to the local
    HTTP SSE endpoint and forwards each SSE chunk as one outbound WS text
    frame, so per-event streaming survives the WS transport. Auth is the
    ``Authorization: Bearer <key>`` or ``X-Fusion-Route`` header echoed by
    the client in the initial query params / subprotocol, checked here.
    """
    from ..config import get_config

    cfg = get_config()
    api_key = getattr(cfg, "api_key", None) or ""
    # accept first, then auth via subprotocol or query param
    token = ws.query_params.get("token") or ""
    sub = ws.headers.get("sec-websocket-protocol", "")
    if api_key and token != api_key and not sub.startswith(api_key):
        await ws.close(code=4401, reason="unauthorized")
        return
    await ws.accept()
    host = getattr(cfg, "bind_host", "127.0.0.1") or "127.0.0.1"
    port = getattr(cfg, "bind_port", 11434)
    base = f"http://{host}:{port}"
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    body = json.loads(raw)
                except Exception:
                    await ws.send_text(json.dumps({"error": "invalid JSON frame"}))
                    continue
                body.setdefault("stream", True)
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                headers["X-Fusion-Route"] = "gateway-decision"
                try:
                    async with client.stream(
                        "POST", f"{base}/v1/responses", json=body, headers=headers
                    ) as r:
                        async for chunk in r.aiter_text():
                            if chunk:
                                await ws.send_text(chunk)
                except Exception as e:
                    logger.warning("responses ws delegate failed: %s", e)
                    await ws.send_text(json.dumps({"error": str(e)}))
        except WebSocketDisconnect:
            logger.info("responses ws client disconnected")
        except Exception as e:
            logger.warning("responses ws loop error: %s", e)
