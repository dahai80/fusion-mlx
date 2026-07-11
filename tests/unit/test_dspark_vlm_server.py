# SPDX-License-Identifier: Apache-2.0
"""Tests for the DSpark server VLM dev path (.dev multimodal extension).

Covers the multimodal extraction + routing added behind ``--vlm-dev`` /
``DSPARK_VLM_DEV=1``. The generator is mocked (no real model load, no Metal),
and ``_render_prompt`` / ``_load_pil_image`` are patched so the suite never
touches real ``mx`` ops or PIL/http - it runs under the stub-mlx CI shadow.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")  # server.py does ``import mlx.core as mx``

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from fusion_mlx.speculative.dspark import server as dspark_server
from fusion_mlx.speculative.dspark.runtime import DSparkRuntime
from fusion_mlx.speculative.dspark.server import (
    _build_app,
    _extract_multimodal,
    _load_pil_image,
)


@pytest.fixture(autouse=True)
def _dspark_executor(monkeypatch):
    # /v1/chat/completions routes generate_* through the module-global
    # _dspark_executor (set only by run_dspark_server in production). _build_app
    # alone never initializes it, so provide a real 1-worker pool per test.
    ex = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(dspark_server, "_dspark_executor", ex)
    yield
    ex.shutdown(wait=False)


class _FakeResult:
    def __init__(self, text: str = "ok", prompt_tokens: int = 7):
        self.text = text
        self.generated_tokens = [1, 2, 3]
        self.metrics = {"num_input_tokens": prompt_tokens}


class _FakeEvent:
    def __init__(self, delta, finished=False, metrics=None, generated_tokens=None):
        self.delta = delta
        self.finished = finished
        self.metrics = metrics
        self.generated_tokens = generated_tokens or []


def _make_runtime(is_vlm: bool) -> DSparkRuntime:
    generator = MagicMock()
    generator._is_vlm.return_value = is_vlm
    generator.draft_quantization = None
    generator.generate_from_tokens.return_value = _FakeResult("plain", 5)
    generator.generate_multimodal.return_value = _FakeResult("caption", 42)
    return DSparkRuntime(
        generator=generator,
        target_repo="mlx-community/Qwen3-VL-8B",
        draft_path="/tmp/draft",
    )


def _client(is_vlm: bool, vlm_dev: bool):
    runtime = _make_runtime(is_vlm)
    app = _build_app(
        runtime=runtime,
        served_model_name="qwen3-vl",
        default_max_tokens=64,
        cors_origins=["*"],
        enable_thinking_default=False,
        vlm_dev_enabled=vlm_dev,
    )
    return TestClient(app), runtime


_IMG_PART = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def test_extract_multimodal_string_content():
    text, imgs = _extract_multimodal("hello world")
    assert text == "hello world"
    assert imgs == []


def test_extract_multimodal_text_parts_concat():
    content = [
        {"type": "text", "text": "describe "},
        {"type": "text", "text": "this"},
    ]
    text, imgs = _extract_multimodal(content)
    assert text == "describe this"
    assert imgs == []


def test_extract_multimodal_drops_unknown_parts():
    content = [
        {"type": "text", "text": "hi"},
        {"type": "audio_url", "audio_url": {"url": "x"}},
        "raw-string",
    ]
    text, imgs = _extract_multimodal(content)
    assert text == "hiraw-string"
    assert imgs == []


def test_load_pil_image_data_uri(tmp_path):
    pytest.importorskip("PIL")
    import base64

    from PIL import Image

    buf = tmp_path / "p.png"
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buf)
    b64 = base64.b64encode(buf.read_bytes()).decode()
    img = _load_pil_image(f"data:image/png;base64,{b64}")
    assert img.size == (1, 1)


def test_load_pil_image_file_path(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    f = tmp_path / "img.png"
    Image.new("RGB", (2, 3), (0, 255, 0)).save(f)
    img = _load_pil_image(str(f))
    assert img.size == (2, 3)


def test_vlm_dev_routes_image_to_generate_multimodal():
    with (
        patch("fusion_mlx.speculative.dspark.server._load_pil_image") as m_img,
        patch("fusion_mlx.speculative.dspark.server._render_prompt") as m_render,
    ):
        m_img.return_value = "fake-pil"
        m_render.return_value = (None, 9)
        client, runtime = _client(is_vlm=True, vlm_dev=True)
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "describe"}, _IMG_PART],
                    }
                ]
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "caption"
    assert body["usage"]["prompt_tokens"] == 42
    runtime.generator.generate_multimodal.assert_called_once()
    runtime.generator.generate_from_tokens.assert_not_called()
    _, kwargs = runtime.generator.generate_multimodal.call_args
    assert kwargs["images"] == ["fake-pil"]
    m_render.assert_not_called()


def test_vlm_dev_off_drops_images_uses_text_path():
    with (
        patch("fusion_mlx.speculative.dspark.server._load_pil_image") as m_img,
        patch("fusion_mlx.speculative.dspark.server._render_prompt") as m_render,
    ):
        m_img.return_value = "fake-pil"
        m_render.return_value = (None, 9)
        client, runtime = _client(is_vlm=True, vlm_dev=False)
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hi"}, _IMG_PART],
                    }
                ]
            },
        )
    assert r.status_code == 200, r.text
    runtime.generator.generate_from_tokens.assert_called_once()
    runtime.generator.generate_multimodal.assert_not_called()


def test_vlm_dev_non_vlm_target_drops_images():
    with (
        patch("fusion_mlx.speculative.dspark.server._load_pil_image") as m_img,
        patch("fusion_mlx.speculative.dspark.server._render_prompt") as m_render,
    ):
        m_img.return_value = "fake-pil"
        m_render.return_value = (None, 9)
        client, runtime = _client(is_vlm=False, vlm_dev=True)
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hi"}, _IMG_PART],
                    }
                ]
            },
        )
    assert r.status_code == 200, r.text
    runtime.generator.generate_from_tokens.assert_called_once()
    runtime.generator.generate_multimodal.assert_not_called()


def test_vlm_dev_text_only_uses_text_path():
    with patch("fusion_mlx.speculative.dspark.server._render_prompt") as m_render:
        m_render.return_value = (None, 9)
        client, runtime = _client(is_vlm=True, vlm_dev=True)
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "just text"}]},
        )
    assert r.status_code == 200, r.text
    runtime.generator.generate_from_tokens.assert_called_once()
    runtime.generator.generate_multimodal.assert_not_called()


def test_vlm_dev_streaming_routes_to_stream_multimodal():
    events = [
        _FakeEvent("cap"),
        _FakeEvent(
            "tion",
            finished=True,
            metrics={"num_input_tokens": 42, "avg_acceptance_length": 2.0},
            generated_tokens=[1, 2, 3],
        ),
    ]
    with (
        patch("fusion_mlx.speculative.dspark.server._load_pil_image") as m_img,
        patch("fusion_mlx.speculative.dspark.server._render_prompt") as m_render,
    ):
        m_img.return_value = "fake-pil"
        m_render.return_value = (None, 9)
        client, runtime = _client(is_vlm=True, vlm_dev=True)
        runtime.generator.stream_multimodal.return_value = iter(events)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "describe"}, _IMG_PART],
                    }
                ],
            },
        ) as r:
            assert r.status_code == 200
            text = "".join(r.iter_lines())
    runtime.generator.stream_multimodal.assert_called_once()
    runtime.generator.stream_from_tokens.assert_not_called()
    assert "cap" in text
    assert "tion" in text
    assert "[DONE]" in text
    assert "prompt_tokens" in text
