# SPDX-License-Identifier: Apache-2.0
# #1412: some chat templates escape literal braces, so the model emits
# {{"name": ...}} instead of {"name": ...}. json.loads died at char 1, neither
# fallback recovered, and the tool call was silently dropped — agentic clients
# then got a bare-text assistant message and stalled.

from __future__ import annotations

import json
from types import SimpleNamespace

from fusion_mlx.api.tool_calling.parse import (
    _parse_xml_tool_calls,
    _undouble_braces,
    parse_tool_calls,
)

DOUBLED = (
    '\n{{"name": "Read", "arguments": '
    '{"file_path": "/private/tmp/fc-t223b/work/notes.txt", "limit": 2000}}}\n'
)


def _wrapped(body: str) -> str:
    return f"␝{body}␞"


def test_undouble_braces_collapses_nested_pairs():
    assert _undouble_braces('{{"a": {{"b": 1}}}}') == '{"a": {"b": 1}}'


def test_xml_fallback_recovers_doubled_brace_call():
    cleaned, calls = _parse_xml_tool_calls(_wrapped(DOUBLED))
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].function.name == "Read"
    assert json.loads(calls[0].function.arguments) == {
        "file_path": "/private/tmp/fc-t223b/work/notes.txt",
        "limit": 2000,
    }
    assert cleaned == ""


def test_xml_fallback_leaves_valid_nested_json_alone():
    body = '{"name": "Render", "arguments": {"template": "{{x}}", "depth": {"a": {"b": 1}}}}'
    _, calls = _parse_xml_tool_calls(_wrapped(body))
    assert calls is not None
    args = json.loads(calls[0].function.arguments)
    assert args["template"] == "{{x}}"
    assert args["depth"] == {"a": {"b": 1}}


def test_xml_fallback_still_rejects_unrecoverable_bodies():
    _, calls = _parse_xml_tool_calls(_wrapped('{{"name": "Read", "arguments": }}}'))
    assert calls is None


def test_bare_tool_call_tag_recovers_doubled_braces():
    text = f"<tool_call>{DOUBLED}</tool_call>"
    cleaned, calls = parse_tool_calls(text, tokenizer=None)
    assert calls is not None
    assert calls[0].function.name == "Read"
    assert cleaned == ""


def test_native_parser_retries_after_de_doubling():
    seen = []

    def tool_parser(body, tools):
        seen.append(body)
        return json.loads(body)

    tokenizer = SimpleNamespace(
        has_tool_calling=True,
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        tool_parser=tool_parser,
    )
    text = f"<tool_call>{DOUBLED}</tool_call>"

    cleaned, calls = parse_tool_calls(text, tokenizer=tokenizer)

    assert calls is not None
    assert calls[0].function.name == "Read"
    assert json.loads(calls[0].function.arguments)["limit"] == 2000
    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert cleaned == ""


def test_native_parser_keeps_valid_bodies_on_the_first_pass():
    seen = []

    def tool_parser(body, tools):
        seen.append(body)
        return json.loads(body)

    tokenizer = SimpleNamespace(
        has_tool_calling=True,
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        tool_parser=tool_parser,
    )
    text = '<tool_call>{"name": "Read", "arguments": {"file_path": "/a"}}</tool_call>'

    _, calls = parse_tool_calls(text, tokenizer=tokenizer)

    assert calls is not None
    assert len(seen) == 1
