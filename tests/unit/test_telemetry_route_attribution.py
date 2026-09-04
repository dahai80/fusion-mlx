# SPDX-License-Identifier: Apache-2.0
import logging
from collections import deque

from fusion_mlx.telemetry import emit, state
from fusion_mlx.telemetry.activation_spec import (
    ACTIVATION_FIRST_INFERENCE,
    CHAT_SPAWN_ENV,
    SURFACE_API,
    SURFACE_CLI,
)

logger = logging.getLogger(__name__)


class _FakeQueue:
    def __init__(self):
        self.events = deque()

    def enqueue(self, payload):
        logger.info(
            "fake-queue enqueue surface=%s",
            payload.get("activation", {}).get("surface"),
        )
        self.events.append(payload)


def test_server_surface_cli_when_chat_spawn_set(monkeypatch):
    monkeypatch.setenv(CHAT_SPAWN_ENV, "1")
    surface = emit.server_surface()
    logger.info("chat_spawn=1 -> surface=%s", surface)
    assert surface == SURFACE_CLI


def test_server_surface_api_when_chat_spawn_unset(monkeypatch):
    monkeypatch.delenv(CHAT_SPAWN_ENV, raising=False)
    surface = emit.server_surface()
    logger.info("chat_spawn unset -> surface=%s", surface)
    assert surface == SURFACE_API


def test_activation_surface_matches_server_surface(monkeypatch, tmp_path):
    monkeypatch.setenv(CHAT_SPAWN_ENV, "1")
    monkeypatch.setattr(
        state, "_default_telemetry_dir", lambda: tmp_path, raising=False
    )
    monkeypatch.setattr(state, "_activation_latch", set(), raising=False)
    monkeypatch.setattr(emit, "is_enabled", lambda *a, **kw: True, raising=False)
    monkeypatch.setattr(
        emit, "claim_activation_marker", lambda kind: True, raising=False
    )
    fake = _FakeQueue()
    monkeypatch.setattr(emit, "get_queue", lambda: fake, raising=False)

    emit.activation(
        activation_kind=ACTIVATION_FIRST_INFERENCE, surface=emit.server_surface()
    )
    assert fake.events, "activation payload was not enqueued"
    payload = fake.events[-1]
    assert payload["activation"]["surface"] == SURFACE_CLI
    logger.info("activation payload surface=%s", payload["activation"]["surface"])
