import hashlib
import hmac
import json
import logging

import httpx

from fusion_mlx.notifications import (
    WEBHOOK_SIGNATURE_HEADER,
    EventNotifier,
    configure_notifier,
    get_notifier,
)


class TestSSEFanout:
    def test_subscriber_receives_event(self):
        notifier = EventNotifier()
        q = notifier.subscribe()
        notifier.emit("request.completed", {"request_id": "r1"})
        event = q.get_nowait()
        assert event["type"] == "request.completed"
        assert event["data"]["request_id"] == "r1"
        assert "ts" in event

    def test_unsubscribe_stops_delivery(self):
        notifier = EventNotifier()
        q = notifier.subscribe()
        notifier.unsubscribe(q)
        notifier.emit("batch.status", {"running": 1})
        assert q.empty()

    def test_drop_on_full_increments_counter(self):
        notifier = EventNotifier(sse_queue_size=1)
        q = notifier.subscribe()
        notifier.emit("e", {"n": 1})
        notifier.emit("e", {"n": 2})
        assert notifier.dropped_events >= 1
        assert q.qsize() == 1


class TestWebhookFanout:
    def test_signed_post(self, monkeypatch):
        received = []

        def fake_post(url, content=None, headers=None, timeout=None):
            received.append((url, content, headers, timeout))
            return httpx.Response(200)

        monkeypatch.setattr("fusion_mlx.notifications.httpx.post", fake_post)
        notifier = EventNotifier(
            webhook_urls=["http://hook.test/a"],
            webhook_secret="s3cret",
        )
        notifier.emit("request.completed", {"request_id": "r1"})
        notifier._webhook_pool.shutdown(wait=True)
        notifier.close()
        assert len(received) == 1
        url, body, headers, _ = received[0]
        assert url == "http://hook.test/a"
        event = json.loads(body)
        assert event["type"] == "request.completed"
        expected = hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        assert headers[WEBHOOK_SIGNATURE_HEADER] == expected

    def test_signature_is_hmac_sha256(self):
        body = json.dumps({"type": "e", "data": {}}).encode()
        sig = hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        assert len(sig) == 64

    def test_webhook_failure_does_not_raise(self, caplog):
        notifier = EventNotifier(webhook_urls=["http://127.0.0.1:1/nope"])
        with caplog.at_level(logging.WARNING):
            notifier.emit("e", {})
            notifier._webhook_pool.shutdown(wait=True)
        assert any("Webhook POST" in r.message for r in caplog.records)
        notifier.close()


class TestSingleton:
    def test_configure_then_get(self):
        n = configure_notifier(webhook_urls=["http://x/"], webhook_secret="k")
        assert get_notifier() is n
        assert n.webhook_urls == ["http://x/"]
        configure_notifier()

    def test_default_singleton_has_no_webhooks(self):
        n = get_notifier()
        assert n.webhook_urls == []


class TestEmitPointIntegration:
    def test_scheduler_batch_status_throttled(self):
        from fusion_mlx.scheduler.sched_response import _emit_batch_status

        class FakeSched:
            running = {"a": 1}
            waiting = []

        notifier = EventNotifier()
        q = notifier.subscribe()
        import fusion_mlx.notifications as notif_mod

        original = notif_mod.get_notifier
        notif_mod.get_notifier = lambda: notifier
        try:
            sched = FakeSched()
            _emit_batch_status(sched)
            _emit_batch_status(sched)
        finally:
            notif_mod.get_notifier = original
        assert q.qsize() == 1
        event = q.get_nowait()
        assert event["type"] == "batch.status"
        assert event["data"] == {"running": 1, "queued": 0}
