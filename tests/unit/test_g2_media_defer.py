# SPDX-License-Identifier: Apache-2.0
"""G2-SR: watchdog media-awareness + SR to_thread tests.

Verifies the G2 llm watchdog defers poison while a media job is active
(super-resolution / in-process image gen starves the shared Metal command
queue — the LLM worker is not hung, just starved). Without this the watchdog
false-poisons at 120s, killing a healthy engine and 500ing co-resident LLM
requests (#901-class production incident).
"""

import threading
import time

import pytest

from fusion_mlx import engine_core as ec
from fusion_mlx.engine_core import (
    is_llm_executor_poisoned,
    is_media_job_active,
    reset_llm_executor_poison,
    set_media_job_active,
    start_llm_watchdog,
    stop_llm_watchdog,
)


class _FastEvent:
    def __init__(self):
        self._flag = threading.Event()

    def set(self):
        self._flag.set()

    def clear(self):
        self._flag.clear()

    def is_set(self):
        return self._flag.is_set()

    def wait(self, timeout=None):
        return self._flag.wait(0.05)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    reset_llm_executor_poison()
    stop_llm_watchdog()
    set_media_job_active(False)
    with ec._llm_heartbeat_lock:
        ec._llm_heartbeat = 0.0
        ec._llm_step_deadline = 0.0
    monkeypatch.setattr(ec, "_llm_watchdog_stop", _FastEvent())
    yield
    stop_llm_watchdog()
    reset_llm_executor_poison()
    set_media_job_active(False)
    with ec._llm_heartbeat_lock:
        ec._llm_step_deadline = 0.0


class TestMediaFlag:
    def test_default_inactive(self):
        assert is_media_job_active() is False

    def test_set_active(self):
        set_media_job_active(True)
        assert is_media_job_active() is True

    def test_clear_resets_defer_state(self):
        set_media_job_active(True)
        assert ec._media_defer_start == 0.0
        set_media_job_active(False)
        assert ec._media_defer_start == 0.0
        assert ec._media_defer_logged is False


class TestWatchdogMediaDefer:
    def test_media_active_defers_poison(self):
        # Stale heartbeat + media active → NO poison (GPU starvation, not hang).
        ec._LLM_WATCHDOG_TIMEOUT_S = 0.2
        set_media_job_active(True)
        start_llm_watchdog()
        with ec._llm_heartbeat_lock:
            ec._llm_step_deadline = time.monotonic() + 10.0
            ec._llm_heartbeat = time.monotonic() - 5.0
        time.sleep(0.6)
        assert is_llm_executor_poisoned() is False
        stop_llm_watchdog()

    def test_media_cleared_then_poison_fires(self):
        # While media active: deferred. After clearing: poison fires.
        ec._LLM_WATCHDOG_TIMEOUT_S = 0.2
        set_media_job_active(True)
        start_llm_watchdog()
        with ec._llm_heartbeat_lock:
            ec._llm_step_deadline = time.monotonic() + 10.0
            ec._llm_heartbeat = time.monotonic() - 5.0
        time.sleep(0.4)
        assert is_llm_executor_poisoned() is False
        # Media job ends — watchdog resumes normal policing.
        set_media_job_active(False)
        time.sleep(0.4)
        assert is_llm_executor_poisoned() is True
        stop_llm_watchdog()

    def test_media_defer_exceeded_poisons(self):
        # Media active but exceeded the defer budget → poison (stuck media job).
        ec._LLM_WATCHDOG_TIMEOUT_S = 0.2
        ec._LLM_WATCHDOG_MEDIA_DEFER_S = 0.3
        set_media_job_active(True)
        start_llm_watchdog()
        with ec._llm_heartbeat_lock:
            ec._llm_step_deadline = time.monotonic() + 10.0
            ec._llm_heartbeat = time.monotonic() - 5.0
        # 0.4s > 0.3s defer budget → poison despite media active
        time.sleep(0.6)
        assert is_llm_executor_poisoned() is True
        stop_llm_watchdog()
        ec._LLM_WATCHDOG_MEDIA_DEFER_S = 3600.0

    def test_no_defer_without_pending_step(self):
        # No pending step (deadline=0) → watchdog never arms, media or not.
        ec._LLM_WATCHDOG_TIMEOUT_S = 0.2
        set_media_job_active(True)
        with ec._llm_heartbeat_lock:
            ec._llm_step_deadline = 0.0
            ec._llm_heartbeat = time.monotonic() - 999.0
        start_llm_watchdog()
        time.sleep(0.4)
        assert is_llm_executor_poisoned() is False
        stop_llm_watchdog()


class TestSRRouteMediaFlag:
    def test_sr_route_sets_and_clears_media_flag(self, monkeypatch):
        import numpy as np
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from fusion_mlx.api import images_sr

        def _stub_resolve(
            images,
            model_path=None,
            scale=4,
            tile_size=512,
            tile_overlap=64,
            config=None,
        ):
            # Assert the media flag was set before entering Metal work.
            assert is_media_job_active() is True
            n, h, w, c = images.shape
            return np.repeat(np.repeat(images, scale, axis=1), scale, axis=2).astype(
                np.float32
            )

        monkeypatch.setattr(images_sr, "super_resolve", _stub_resolve)
        monkeypatch.setattr(images_sr.os.path, "exists", lambda p: True)
        app = FastAPI()
        app.include_router(images_sr.router)
        client = TestClient(app)

        import io

        from PIL import Image

        pil = Image.fromarray((np.random.rand(32, 40, 3) * 255).astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        resp = client.post(
            "/v1/images/super-resolution",
            files={"image": ("frame.png", buf.getvalue(), "image/png")},
            data={"scale": "4", "tile_size": "512"},
        )
        assert resp.status_code == 200
        # Flag must be cleared after the route returns.
        assert is_media_job_active() is False

    def test_sr_route_clears_flag_on_exception(self, monkeypatch):
        import numpy as np
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from fusion_mlx.api import images_sr

        def _boom(*a, **kw):
            assert is_media_job_active() is True
            raise RuntimeError("metal blew up")

        monkeypatch.setattr(images_sr, "super_resolve", _boom)
        monkeypatch.setattr(images_sr.os.path, "exists", lambda p: True)
        app = FastAPI()
        app.include_router(images_sr.router)
        client = TestClient(app, raise_server_exceptions=False)

        import io

        from PIL import Image

        pil = Image.fromarray((np.random.rand(16, 16, 3) * 255).astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        resp = client.post(
            "/v1/images/super-resolution",
            files={"image": ("frame.png", buf.getvalue(), "image/png")},
            data={"scale": "2", "tile_size": "512"},
        )
        assert resp.status_code == 500
        # Flag MUST be cleared even on exception (finally block).
        assert is_media_job_active() is False
