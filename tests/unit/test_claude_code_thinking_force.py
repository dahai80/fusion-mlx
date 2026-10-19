# SPDX-License-Identifier: Apache-2.0
# Tests for Claude Code request normalization on the OpenAI chat surface:
# system-prompt-signature detection + forced enable_thinking + temperature
# cap (converter drops Anthropic thinking intent; see #0913 companion fix).
import pytest

from fusion_mlx.api.models import ChatCompletionRequest
from fusion_mlx.api.openai._common import (
    _CLAUDE_CODE_TEMP_CAP,
    is_claude_code_prompt,
    normalize_claude_code_request,
)


def _req(**kw):
    base = dict(
        model="mlx-community/Qwen3.8-27B-4bit",
        messages=[
            {
                "role": "system",
                "content": "You are Claude Code, Anthropic's official CLI for Claude.",
            },
            {"role": "user", "content": "hi"},
        ],
        temperature=1.0,
    )
    base.update(kw)
    return ChatCompletionRequest(**base)


@pytest.fixture(autouse=True)
def _force_env_on(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_CC_THINKING_FORCE", "1")


def test_detects_system_prompt_signature():
    assert is_claude_code_prompt(_req()) is True


def test_non_cc_request_not_detected():
    req = _req(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ]
    )
    assert is_claude_code_prompt(req) is False


def test_user_role_with_signature_not_detected():
    # Only system-role messages count — a user message quoting the phrase
    # must not trigger the force.
    req = _req(messages=[{"role": "user", "content": "You are Claude Code"}])
    assert is_claude_code_prompt(req) is False


def test_no_system_message_untouched():
    req = _req(
        messages=[{"role": "user", "content": "You are Claude Code"}],
        temperature=0.9,
    )
    changed = normalize_claude_code_request(req)
    assert changed is False
    assert req.temperature == 0.9
    assert req.enable_thinking is None


def test_forces_thinking_and_caps_temp():
    req = _req(temperature=0.7)
    assert normalize_claude_code_request(req) is True
    assert req.enable_thinking is True
    assert req.temperature == _CLAUDE_CODE_TEMP_CAP


@pytest.mark.parametrize(
    "inp,expected",
    [
        (1.0, _CLAUDE_CODE_TEMP_CAP),
        (0.7, _CLAUDE_CODE_TEMP_CAP),
        (0.2, 0.2),
        (0.0, 0.0),
        (None, _CLAUDE_CODE_TEMP_CAP),
    ],
)
def test_temp_cap_matrix(inp, expected):
    req = _req(temperature=inp)
    normalize_claude_code_request(req)
    assert req.temperature == expected
    assert req.enable_thinking is True


def test_explicit_disable_honored():
    req = _req(enable_thinking=False)
    changed = normalize_claude_code_request(req)
    assert changed is True
    assert req.temperature == _CLAUDE_CODE_TEMP_CAP
    assert req.enable_thinking is False


def test_explicit_enable_no_reforce():
    req = _req(enable_thinking=True)
    changed = normalize_claude_code_request(req)
    assert changed is True
    assert req.enable_thinking is True
    assert req.temperature == _CLAUDE_CODE_TEMP_CAP


def test_kill_switch_env_off(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_CC_THINKING_FORCE", "0")
    req = _req()
    changed = normalize_claude_code_request(req)
    assert changed is False
    assert req.temperature == 1.0
    assert req.enable_thinking is None


def test_multimodal_system_parts_detected():
    req = _req(
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                    },
                ],
            },
            {"role": "user", "content": "hi"},
        ]
    )
    assert is_claude_code_prompt(req) is True


def test_multimodal_parts_without_text_key_ignored():
    req = _req(
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            },
        ]
    )
    assert is_claude_code_prompt(req) is False


def test_wiring_resolve_inherits_force():
    # End-to-end seam contract: both _run_chat and _stream_chat build
    # ct_kwargs via resolve_enable_thinking_default(client_thinking=
    # client_thinking_from_request(request)). Post-normalization the force
    # must survive that seam.
    from fusion_mlx.api.utils import (
        client_thinking_from_request,
        resolve_enable_thinking_default,
    )

    req = _req()
    normalize_claude_code_request(req)
    ct_kwargs = {}
    resolve_enable_thinking_default(
        ct_kwargs, client_thinking=client_thinking_from_request(req)
    )
    assert ct_kwargs.get("enable_thinking") is True


def test_wiring_explicit_disable_survives_resolve():
    from fusion_mlx.api.utils import (
        client_thinking_from_request,
        resolve_enable_thinking_default,
    )

    req = _req(enable_thinking=False)
    normalize_claude_code_request(req)
    ct_kwargs = {}
    resolve_enable_thinking_default(
        ct_kwargs, client_thinking=client_thinking_from_request(req)
    )
    assert ct_kwargs.get("enable_thinking") is False
