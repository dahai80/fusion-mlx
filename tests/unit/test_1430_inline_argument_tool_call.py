# SPDX-License-Identifier: Apache-2.0
# #1430: a 27B checkpoint serving Claude Code emitted the argument object
# inside the Qwen tag — <function=Read({file_path: "/workspace/src/settings.py"})}
# — with no <parameter> children. The native parser raised
# "No function provided.", the XML fallback had no branch for the shape, and
# the call was dropped: the agent loop got a bare-text reply and stalled until
# the user typed "continue".

from __future__ import annotations

import json
from types import SimpleNamespace

from fusion_mlx.api.tool_calling.parse import (
    _loads_inline_arguments,
    _parse_xml_tool_calls,
    parse_tool_calls,
)

OBSERVED = (
    '\n<function=Read({file_path: "/workspace/src/settings.py"})}\n</parameter>\n'
)


def _wrapped(body: str) -> str:
    return f"␝{body}␞"


def test_loads_inline_arguments_quotes_bare_keys():
    assert _loads_inline_arguments('{file_path: "/a/b.txt"}') == {
        "file_path": "/a/b.txt"
    }


def test_loads_inline_arguments_keeps_valid_json():
    assert _loads_inline_arguments('{"limit": 20, "file_path": "/a"}') == {
        "limit": 20,
        "file_path": "/a",
    }


def test_loads_inline_arguments_rejects_non_object():
    assert _loads_inline_arguments("read the file") is None


def test_xml_fallback_recovers_inline_argument_call():
    cleaned, calls = _parse_xml_tool_calls(_wrapped(OBSERVED))
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].function.name == "Read"
    assert json.loads(calls[0].function.arguments) == {
        "file_path": "/workspace/src/settings.py"
    }
    assert cleaned == ""


def test_xml_fallback_recovers_inline_call_without_stray_brace():
    body = '<function=Bash({"command": "ls -la"})>'
    _, calls = _parse_xml_tool_calls(_wrapped(body))
    assert calls is not None
    assert calls[0].function.name == "Bash"
    assert json.loads(calls[0].function.arguments) == {"command": "ls -la"}


def test_parse_tool_calls_recovers_observed_body_end_to_end():
    seen = []

    def tool_parser(body, tools):
        seen.append(body)
        raise ValueError("No function provided.")

    tokenizer = SimpleNamespace(
        has_tool_calling=True,
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        tool_parser=tool_parser,
    )
    text = f"<tool_call>{OBSERVED}</tool_call>"

    _, calls = parse_tool_calls(text, tokenizer=tokenizer)

    assert seen, "native parser never ran"
    assert calls is not None
    assert calls[0].function.name == "Read"
    assert json.loads(calls[0].function.arguments)["file_path"].endswith("settings.py")


def test_unrecoverable_inline_body_is_still_dropped():
    _, calls = _parse_xml_tool_calls(_wrapped("<function=Read(not an object)>"))
    assert calls is None
