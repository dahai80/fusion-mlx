import json
import logging

import pytest

from fusion_mlx.tool_parsers.gemma4_tool_parser import (
    Gemma4ToolParser,
    _parse_gemma4_args,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "args_str,expected",
    [
        ("a:3,b:4", {"a": 3, "b": 4}),
        ("flag:true,n:42", {"flag": True, "n": 42}),
        ("x:null", {"x": None}),
        ("rate:0.5", {"rate": 0.5}),
        ("flag:false", {"flag": False}),
        ('city:<|"|>Paris<|"|>', {"city": "Paris"}),
        ('a:3,b:<|"|>hi<|"|>', {"a": 3, "b": "hi"}),
        (
            'first:<|"|>Alice<|"|>,last:<|"|>Smith<|"|>',
            {"first": "Alice", "last": "Smith"},
        ),
        (
            'flag:true,n:42,name:<|"|>Bob<|"|>',
            {"flag": True, "n": 42, "name": "Bob"},
        ),
        ('msg:<|"|>hello, world<|"|>', {"msg": "hello, world"}),
        ("n:-5", {"n": -5}),
        ("", {}),
    ],
)
def test_parse_gemma4_args(args_str, expected):
    assert _parse_gemma4_args(args_str) == expected


def test_extract_bare_numeric_args():
    parser = Gemma4ToolParser()
    out = "<|tool_call>call:add{a:3,b:4}<tool_call|>"
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["name"] == "add"
    args = json.loads(tc["arguments"])
    assert args == {"a": 3, "b": 4}
    assert res.content is None


def test_extract_quoted_string_args():
    parser = Gemma4ToolParser()
    out = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {"city": "Paris"}
    assert res.content is None


def test_extract_mixed_args():
    parser = Gemma4ToolParser()
    out = '<|tool_call>call:mix{a:3,b:<|"|>hi<|"|>}<tool_call|>'
    res = parser.extract_tool_calls(out)
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {"a": 3, "b": "hi"}
    assert res.content is None


def test_no_tool_call_returns_content_unchanged():
    parser = Gemma4ToolParser()
    out = "Hello, the answer is 42."
    res = parser.extract_tool_calls(out)
    assert res.tools_called is False
    assert res.content == out
    assert res.tool_calls == []


def test_multiple_tool_calls():
    parser = Gemma4ToolParser()
    out = (
        "<|tool_call>call:add{a:1,b:2}<tool_call|>"
        "<|tool_call>call:multiply{a:3,b:4}<tool_call|>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0]["name"] == "add"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"a": 1, "b": 2}
    assert res.tool_calls[1]["name"] == "multiply"
    assert json.loads(res.tool_calls[1]["arguments"]) == {"a": 3, "b": 4}
    assert res.content is None


def test_tool_call_with_surrounding_content():
    parser = Gemma4ToolParser()
    out = (
        "Let me check the weather. "
        '<|tool_call>call:get_weather{city:<|"|>NYC<|"|>}<tool_call|>'
        " That should help."
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.content is not None
    assert "<|tool_call>" not in res.content
    assert "<tool_call|>" not in res.content
    assert "weather" in res.content
    assert "should help" in res.content


def test_streaming_emits_completed_tool_call_once():
    parser = Gemma4ToolParser()
    parser.reset()
    full = "<|tool_call>call:add{a:3,b:4}<tool_call|>"
    midpoint = len(full) // 2
    delta1 = full[:midpoint]
    delta2 = full[midpoint:]
    r1 = parser.extract_tool_calls_streaming("", delta1, delta1)
    assert r1 is None
    r2 = parser.extract_tool_calls_streaming(delta1, full, delta2)
    assert r2 is not None
    assert "tool_calls" in r2
    assert len(r2["tool_calls"]) == 1
    tc = r2["tool_calls"][0]
    assert tc["function"]["name"] == "add"
    assert json.loads(tc["function"]["arguments"]) == {"a": 3, "b": 4}
    r3 = parser.extract_tool_calls_streaming(full, full, "")
    assert r3 is None


def test_streaming_passthrough_when_no_markup():
    parser = Gemma4ToolParser()
    parser.reset()
    r = parser.extract_tool_calls_streaming("", "Hello world", "Hello world")
    assert r == {"content": "Hello world"}


def test_extract_stripped_form_bare_numeric():
    parser = Gemma4ToolParser()
    out = "call:add{a:432,b:1}"
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["name"] == "add"
    assert json.loads(tc["arguments"]) == {"a": 432, "b": 1}
    assert res.content is None


def test_extract_stripped_form_quoted_string():
    parser = Gemma4ToolParser()
    out = 'call:get_weather{location:<|"|>Palo Alto<|"|>}'
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {"location": "Palo Alto"}
    assert res.content is None


def test_extract_stripped_form_calculator_user_report():
    parser = Gemma4ToolParser()
    out = "call:calculator{expression:432+1}"
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "calculator"
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {"expression": "432+1"}
    assert res.content is None


def test_streaming_stripped_form_suppresses_then_emits():
    parser = Gemma4ToolParser()
    parser.reset()
    full = "call:add{a:3,b:4}"
    open_idx = full.index("{") + 1
    delta1 = full[:open_idx]
    delta2 = full[open_idx:]
    r1 = parser.extract_tool_calls_streaming("", delta1, delta1)
    assert r1 is None
    r2 = parser.extract_tool_calls_streaming(delta1, full, delta2)
    assert r2 is not None
    assert "tool_calls" in r2
    assert len(r2["tool_calls"]) == 1
    tc = r2["tool_calls"][0]
    assert tc["function"]["name"] == "add"
    assert json.loads(tc["function"]["arguments"]) == {"a": 3, "b": 4}


def test_streaming_stripped_form_natural_text_passes_through():
    parser = Gemma4ToolParser()
    parser.reset()
    text = "I will call you later: see you then."
    r = parser.extract_tool_calls_streaming("", text, text)
    assert r == {"content": text}


def test_has_pending_recognises_stripped_opener():
    parser = Gemma4ToolParser()
    assert parser.has_pending_tool_call("call:foo{x:1}") is True
    assert parser.has_pending_tool_call("call:foo{") is True
    assert parser.has_pending_tool_call("hello world") is False
    assert parser.has_pending_tool_call("call me later") is False
