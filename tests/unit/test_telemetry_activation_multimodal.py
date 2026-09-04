# SPDX-License-Identifier: Apache-2.0
import glob
import importlib
import logging
import os

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FUSION_MLX_TELEMETRY", raising=False)
    monkeypatch.setenv("FUSION_MLX_TELEMETRY_REQUEST_SAMPLE", "1.0")

    import fusion_mlx.telemetry.state as state

    importlib.reload(state)

    import fusion_mlx.telemetry.emit as emit

    importlib.reload(emit)
    emit._reset_for_tests()
    return tmp_path


@pytest.fixture
def opted_in(fake_home):
    from fusion_mlx.telemetry.state import record_consent

    record_consent(True, fusion_mlx_version="0.0.0+test")
    return fake_home


@pytest.fixture
def stub_queue(monkeypatch):
    from fusion_mlx.telemetry import emit

    captured: list[dict] = []

    class _StubQueue:
        def enqueue(self, payload):
            captured.append(payload)

    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue())
    return captured


def _reset_markers():
    from fusion_mlx.telemetry import state

    for marker in glob.glob(str(state._default_telemetry_dir() / "activation_seen_*")):
        try:
            os.remove(marker)
        except OSError:
            pass
    state._activation_latch.clear()


def test_first_dictation_api_pair_allowed():
    from fusion_mlx.telemetry.activation_spec import (
        ACTIVATION_FIRST_DICTATION,
        ACTIVATION_KIND_SURFACE_PAIRS,
        SURFACE_API,
        is_allowed_activation,
    )

    assert (ACTIVATION_FIRST_DICTATION, SURFACE_API) in ACTIVATION_KIND_SURFACE_PAIRS
    assert is_allowed_activation(ACTIVATION_FIRST_DICTATION, SURFACE_API) is True


def test_image_generation_activation_fires_once(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.activation_spec import (
        ACTIVATION_FIRST_IMAGE_GENERATION,
        SURFACE_API,
    )

    _reset_markers()
    emit.activation(
        activation_kind=ACTIVATION_FIRST_IMAGE_GENERATION, surface=SURFACE_API
    )
    assert any(
        p.get("event") == "activation"
        and p.get("activation", {}).get("activation_kind") == "first_image_generation"
        for p in stub_queue
    ), f"first image-generation activation did not fire: {stub_queue!r}"
    first_len = len(stub_queue)
    emit.activation(
        activation_kind=ACTIVATION_FIRST_IMAGE_GENERATION, surface=SURFACE_API
    )
    assert len(stub_queue) == first_len, (
        f"second image-generation activation fired (should be once-per-install): "
        f"{stub_queue!r}"
    )
    _reset_markers()


def test_video_generation_activation_fires_once(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.activation_spec import (
        ACTIVATION_FIRST_VIDEO_GENERATION,
        SURFACE_API,
    )

    _reset_markers()
    emit.activation(
        activation_kind=ACTIVATION_FIRST_VIDEO_GENERATION, surface=SURFACE_API
    )
    assert any(
        p.get("event") == "activation"
        and p.get("activation", {}).get("activation_kind") == "first_video_generation"
        for p in stub_queue
    ), f"first video-generation activation did not fire: {stub_queue!r}"
    first_len = len(stub_queue)
    emit.activation(
        activation_kind=ACTIVATION_FIRST_VIDEO_GENERATION, surface=SURFACE_API
    )
    assert len(stub_queue) == first_len, (
        f"second video-generation activation fired (should be once-per-install): "
        f"{stub_queue!r}"
    )
    _reset_markers()


def test_dictation_activation_fires_on_api(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.activation_spec import (
        ACTIVATION_FIRST_DICTATION,
        SURFACE_API,
    )

    _reset_markers()
    emit.activation(activation_kind=ACTIVATION_FIRST_DICTATION, surface=SURFACE_API)
    assert any(
        p.get("event") == "activation"
        and p.get("activation", {}).get("activation_kind") == "first_dictation"
        for p in stub_queue
    ), f"first dictation activation did not fire on api surface: {stub_queue!r}"
    _reset_markers()


def test_rejected_pair_logs_and_does_not_fire(opted_in, stub_queue, caplog):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.activation_spec import SURFACE_API

    _reset_markers()
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.telemetry.emit"):
        emit.activation(activation_kind="first_image", surface=SURFACE_API)
    assert not any(
        p.get("event") == "activation"
        and p.get("activation", {}).get("activation_kind") == "first_image"
        for p in stub_queue
    ), f"rejected pair first_image/api unexpectedly fired: {stub_queue!r}"
    assert any(
        "rejected pair first_image/api" in rec.message for rec in caplog.records
    ), "expected rejection warning for first_image/api"
    _reset_markers()
