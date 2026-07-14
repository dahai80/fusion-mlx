# SPDX-License-Identifier: Apache-2.0
# #71 Phase 2: runtime tool_choice enforcement on /v1/messages.
# Direct unit coverage for the ported enforcement logic in
# fusion_mlx/api/_anthropic_helpers.py. The live /v1/messages route
# previously ignored tool_choice entirely (0 refs); these pin the
# named-tool / required / auto contracts enforced post-generation.

from __future__ import annotations

from fusion_mlx.api._anthropic_helpers import (
    _inject_tool_use_required_suffix,
    enforce_tool_choice,
)

_SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
_SEARCH_REQ_ARGS = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {"type": "object", "required": ["q"]},
    },
}
_WEATHER = {
    "type": "function",
    "function": {
        "name": "weather",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_NAMED_SEARCH = {"type": "function", "function": {"name": "search"}}


def _tc(name: str) -> dict:
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


# ---------------------------------------------------------------------------
# enforce_tool_choice: pass-through cases (no forcing)
# ---------------------------------------------------------------------------


def test_no_tool_choice_is_noop():
    calls = [_tc("search")]
    out, err = enforce_tool_choice(calls, None, [_SEARCH])
    assert out == calls
    assert err is None


def test_auto_is_noop():
    calls = [_tc("search")]
    out, err = enforce_tool_choice(calls, "auto", [_SEARCH])
    assert out == calls
    assert err is None


# ---------------------------------------------------------------------------
# enforce_tool_choice: named-tool ({"type":"tool","name":X} -> OpenAI pin)
# ---------------------------------------------------------------------------


def test_named_model_complied_unchanged():
    calls = [_tc("search")]
    out, err = enforce_tool_choice(calls, _NAMED_SEARCH, [_SEARCH])
    assert out == calls
    assert err is None


def test_named_wrong_tool_filtered_then_synth():
    # Model defied the pin and called weather; filter drops it, synth adds
    # the pinned search call so the response still carries a tool_use.
    out, err = enforce_tool_choice([_tc("weather")], _NAMED_SEARCH, [_SEARCH, _WEATHER])
    assert err is None
    assert len(out) == 1
    assert out[0]["function"]["name"] == "search"
    assert out[0]["function"]["arguments"] == "{}"
    assert out[0]["type"] == "function"
    assert out[0]["id"].startswith("call_")


def test_named_no_calls_synth_pinned():
    out, err = enforce_tool_choice([], _NAMED_SEARCH, [_SEARCH])
    assert err is None
    assert len(out) == 1
    assert out[0]["function"]["name"] == "search"


def test_named_mix_keeps_only_pinned():
    calls = [_tc("search"), _tc("weather")]
    out, err = enforce_tool_choice(calls, _NAMED_SEARCH, [_SEARCH, _WEATHER])
    assert err is None
    assert len(out) == 1
    assert out[0]["function"]["name"] == "search"


def test_named_synth_with_required_args_returns_error():
    # Pinned tool needs arguments that cannot be synthesized safely -> 422.
    out, err = enforce_tool_choice([], _NAMED_SEARCH, [_SEARCH_REQ_ARGS])
    assert err is not None
    assert "requires arguments" in err


# ---------------------------------------------------------------------------
# enforce_tool_choice: required (Anthropic {"type":"any"} -> "required")
# ---------------------------------------------------------------------------


def test_required_single_tool_no_calls_synths_solo():
    out, err = enforce_tool_choice([], "required", [_SEARCH])
    assert err is None
    assert len(out) == 1
    assert out[0]["function"]["name"] == "search"


def test_required_multi_tool_no_calls_returns_error():
    out, err = enforce_tool_choice([], "required", [_SEARCH, _WEATHER])
    assert err is not None
    assert "no tool_calls" in err


def test_required_model_complied_unchanged():
    calls = [_tc("search")]
    out, err = enforce_tool_choice(calls, "required", [_SEARCH, _WEATHER])
    assert out == calls
    assert err is None


# ---------------------------------------------------------------------------
# _inject_tool_use_required_suffix: pre-generation prompt lever
# ---------------------------------------------------------------------------

_SUFFIX_MARKER = "You MUST use one of the provided tools"


def test_inject_required_appends_to_existing_system():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    _inject_tool_use_required_suffix(messages, "required", tools=[_SEARCH])
    assert messages[0]["role"] == "system"
    assert _SUFFIX_MARKER in messages[0]["content"]
    assert messages[0]["content"].startswith("You are helpful.")


def test_inject_named_appends_named_suffix():
    messages = [{"role": "system", "content": "base"}]
    _inject_tool_use_required_suffix(messages, _NAMED_SEARCH, tools=[_SEARCH])
    assert "search" in messages[0]["content"]
    assert messages[0]["content"] != "base"


def test_inject_auto_is_noop():
    messages = [{"role": "system", "content": "base"}]
    _inject_tool_use_required_suffix(messages, "auto", tools=[_SEARCH])
    assert messages[0]["content"] == "base"


def test_inject_no_tools_is_noop_even_when_required():
    messages = [{"role": "system", "content": "base"}]
    _inject_tool_use_required_suffix(messages, "required", tools=None)
    assert messages[0]["content"] == "base"


def test_inject_required_inserts_system_when_absent():
    messages = [{"role": "user", "content": "hi"}]
    _inject_tool_use_required_suffix(messages, "required", tools=[_SEARCH])
    assert messages[0]["role"] == "system"
    assert _SUFFIX_MARKER in messages[0]["content"]
    assert messages[1]["role"] == "user"
