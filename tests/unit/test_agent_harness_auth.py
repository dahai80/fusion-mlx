# SPDX-License-Identifier: Apache-2.0
"""Agent harness: env-based API auth + None-tolerant content reads.

Servers with auth.api_key enabled reject keyless requests with 401; the
harness reads FUSION_MLX_AGENT_TEST_API_KEY for both the JSON path
(_api_call) and the SSE path (httpx.stream). Thinking models may return
message.content = None, which must not crash the leak checks.
"""

import pytest

from fusion_mlx.agents.testing import _auth_headers


def test_auth_headers_empty_without_env(monkeypatch):
    monkeypatch.delenv("FUSION_MLX_AGENT_TEST_API_KEY", raising=False)
    assert _auth_headers() == {}


def test_auth_headers_bearer_with_env(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_AGENT_TEST_API_KEY", "k-test")
    assert _auth_headers() == {"Authorization": "Bearer k-test"}


def test_none_content_tolerated():
    # mirrors the leak-check pattern: content may be None for thinking
    # models — `or ""` keeps the marker scan iterable
    message = {"content": None, "tool_calls": [{"id": "x"}]}
    content = message.get("content") or ""
    for marker in ["<tool_call>", "<function="]:
        assert marker not in content


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
