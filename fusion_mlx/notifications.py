# P1-5 (catch-up G6): event notification hub.
# Fan-out bus for server events: SSE subscriber queues plus webhook POST
# targets (HMAC-SHA256 signed). Failures only log — never break inference.
import asyncio
import hashlib
import hmac
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

logger = logging.getLogger(__name__)

WEBHOOK_SIGNATURE_HEADER = "X-Fusion-Signature"

_notifier: "EventNotifier | None" = None
_notifier_lock = threading.Lock()


def configure_notifier(
    webhook_urls: list[str] | None = None,
    webhook_secret: str = "",
    webhook_timeout_s: float = 3.0,
    sse_queue_size: int = 256,
    webhook_workers: int = 4,
) -> "EventNotifier":
    global _notifier
    with _notifier_lock:
        _notifier = EventNotifier(
            webhook_urls=webhook_urls or [],
            webhook_secret=webhook_secret,
            webhook_timeout_s=webhook_timeout_s,
            sse_queue_size=sse_queue_size,
            webhook_workers=webhook_workers,
        )
    return _notifier


def get_notifier() -> "EventNotifier":
    global _notifier
    with _notifier_lock:
        if _notifier is None:
            _notifier = EventNotifier()
        return _notifier


class EventNotifier:
    def __init__(
        self,
        webhook_urls: list[str] | None = None,
        webhook_secret: str = "",
        webhook_timeout_s: float = 3.0,
        sse_queue_size: int = 256,
        webhook_workers: int = 4,
    ):
        self.webhook_urls = list(webhook_urls or [])
        self.webhook_secret = webhook_secret
        self.webhook_timeout_s = webhook_timeout_s
        self._sse_queues: list[asyncio.Queue] = []
        self._sse_queue_size = sse_queue_size
        self._dropped_events = 0
        self._webhook_pool = ThreadPoolExecutor(
            max_workers=max(1, webhook_workers),
            thread_name_prefix="webhook",
        )

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._sse_queue_size)
        self._sse_queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._sse_queues.remove(q)
        except ValueError:
            pass

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    def emit(self, event_type: str, payload: dict) -> None:
        event = {
            "type": event_type,
            "ts": time.time(),
            "data": payload,
        }
        self._fanout_sse(event)
        self._fanout_webhooks(event)

    def _fanout_sse(self, event: dict) -> None:
        for q in list(self._sse_queues):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                self._dropped_events += 1
                logger.warning(
                    "SSE subscriber queue full, event dropped (total dropped: %d)",
                    self._dropped_events,
                )
            except Exception:
                logger.warning("SSE fan-out failed", exc_info=True)

    def _fanout_webhooks(self, event: dict) -> None:
        if not self.webhook_urls:
            return
        body = json.dumps(event).encode("utf-8")
        signature = ""
        if self.webhook_secret:
            signature = hmac.new(
                self.webhook_secret.encode("utf-8"),
                body,
                hashlib.sha256,
            ).hexdigest()
        for url in self.webhook_urls:
            self._webhook_pool.submit(self._post_webhook, url, body, signature)

    def _post_webhook(self, url: str, body: bytes, signature: str) -> None:
        headers = {"Content-Type": "application/json"}
        if signature:
            headers[WEBHOOK_SIGNATURE_HEADER] = signature
        for attempt in (1, 2):
            try:
                resp = httpx.post(
                    url,
                    content=body,
                    headers=headers,
                    timeout=self.webhook_timeout_s,
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "Webhook POST %s returned HTTP %d (attempt %d)",
                        url,
                        resp.status_code,
                        attempt,
                    )
                else:
                    logger.debug("Webhook POST %s -> %d", url, resp.status_code)
                    return
            except Exception as exc:
                logger.warning(
                    "Webhook POST %s failed (attempt %d): %s",
                    url,
                    attempt,
                    exc,
                )
        logger.warning("Webhook POST %s dropped after retry", url)

    def close(self) -> None:
        self._webhook_pool.shutdown(wait=False, cancel_futures=True)
