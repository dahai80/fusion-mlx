#!/usr/bin/env python3
"""Test raw base64 image processing (CyberAI format)."""

import base64
from pathlib import Path

import pytest

# Skip if no icon.png available and no server running
pytestmark = pytest.mark.skipif(
    not Path("icon.png").exists(),
    reason="icon.png not available - integration test requiring running server",
)


def test_raw_base64_image():
    """Test raw base64 image processing via chat completions endpoint."""
    import requests

    with open("icon.png", "rb") as f:
        image_data = f.read()

    raw_base64 = base64.b64encode(image_data).decode("utf-8")

    payload = {
        "model": "gemma-3n-e4b-it-mlx-8bit",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image?"},
                    {"type": "image_url", "image_url": {"url": raw_base64}},
                ],
            }
        ],
        "max_tokens": 100,
    }

    response = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )

    assert response.status_code == 200
    result = response.json()
    message = result["choices"][0]["message"]["content"]
    assert isinstance(message, str)
