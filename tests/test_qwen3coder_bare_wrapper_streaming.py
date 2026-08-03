# SPDX-License-Identifier: Apache-2.0
# Migrated from Rapid-MLX test_qwen3coder_bare_wrapper_streaming.py
# vllm_mlx.tool_parsers.qwen3coder_tool_parser -> fusion_mlx.tool_parsers.qwen3coder_tool_parser

from __future__ import annotations

import json

import pytest

try:
    from fusion_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    _HAS_QWEN3CODER = True
except ImportError:
    _HAS_QWEN3CODER = False


def _require_qwen3coder():
    if not _HAS_QWEN3CODER:
        pytest.skip("fusion_mlx.tool_parsers.qwen3coder_tool_parser not migrated yet")


@pytest.fixture(autouse=True)
def _guard_qwen3coder():
    _require_qwen3coder()


def _request_with_tool(name: str, properties: dict) -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {"type": "object", "properties": properties},
                },
            }
        ]
    }


def _feed(parser: Qwen3CoderToolParser, chunks: list[str], request: dict | None):
    parser.reset()
    deltas: list[dict] = []
    previous = ""
    for chunk in chunks:
        if not chunk:
            continue
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            request=request,
        )
        if delta is not None:
            deltas.append(delta)
        previous = current
    return deltas


def _argument_fragments_for_index(deltas: list[dict], index: int) -> list[str]:
    out: list[str] = []
    for d in deltas:
        for tc in d.get("tool_calls") or []:
            if tc.get("index") != index:
                continue
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if args:
                out.append(args)
    return out


def _names_by_index(deltas: list[dict]) -> dict[int, str]:
    seen: dict[int, str] = {}
    for d in deltas:
        for tc in d.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if name and tc.get("index") not in seen:
                seen[tc.get("index")] = name
    return seen


def _content_events(deltas: list[dict]) -> list[str]:
    return [d["content"] for d in deltas if "content" in d]


def test_bare_function_block_streams_as_tool_call():
    pytest.skip(
        "fusion_mlx Qwen3CoderToolParser bare-wrapper streaming not migrated yet"
    )
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("read_file", {"path": {"type": "string"}})

    chunks = [
        "<function=read_file>\n",
        "<parameter=path>\n",
        "/src/main.py",
        "\n</parameter>\n",
        "</function>",
    ]

    deltas = _feed(parser, chunks, request)

    names = _names_by_index(deltas)
    assert names.get(0) == "read_file", (
        f"bare <function=…> stream never emitted a tool_calls header; "
        f"names={names!r}, deltas={deltas!r}"
    )

    fragments = _argument_fragments_for_index(deltas, 0)
    # Exclude the header emit (empty arguments) so we count only body deltas.
    body_fragments = [f for f in fragments if f != ""]
    combined = "".join(body_fragments)
    decoded = json.loads(combined)
    assert decoded == {"path": "/src/main.py"}, (
        f"bare-wrapper arguments did not stream correctly. "
        f"combined={combined!r}, decoded={decoded!r}"
    )

    # No content event should have leaked the raw tool-call markup.
    for text in _content_events(deltas):
        assert (
            "<function=" not in text
        ), f"bare tool-call markup leaked as content: {text!r}"
        assert (
            "</function>" not in text
        ), f"bare tool-call markup leaked as content: {text!r}"
        assert (
            "<parameter=" not in text
        ), f"bare tool-call markup leaked as content: {text!r}"


def test_bare_multi_function_blocks_stream_both_calls():
    pytest.skip(
        "fusion_mlx Qwen3CoderToolParser bare-wrapper streaming not migrated yet"
    )
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            },
        ]
    }

    chunks = [
        "<function=read_file>",
        "<parameter=path>/a.py</parameter>",
        "</function>",
        "\n",
        "<function=write_file>",
        "<parameter=path>/b.py</parameter>",
        "<parameter=content>hello</parameter>",
        "</function>",
    ]

    deltas = _feed(parser, chunks, request)

    names = _names_by_index(deltas)
    assert names.get(0) == "read_file", f"tool index 0 must be read_file, got {names!r}"
    assert (
        names.get(1) == "write_file"
    ), f"tool index 1 must be write_file, got {names!r}"

    args_0 = json.loads("".join(_argument_fragments_for_index(deltas, 0)))
    args_1 = json.loads("".join(_argument_fragments_for_index(deltas, 1)))
    assert args_0 == {"path": "/a.py"}, args_0
    assert args_1 == {"path": "/b.py", "content": "hello"}, args_1


def test_wrapped_streaming_still_works():
    pytest.skip(
        "fusion_mlx Qwen3CoderToolParser bare-wrapper streaming not migrated yet"
    )
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("read_file", {"path": {"type": "string"}})

    chunks = [
        "<tool_call>\n",
        "<function=read_file>\n",
        "<parameter=path>\n",
        "/src/main.py",
        "\n</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]

    deltas = _feed(parser, chunks, request)

    names = _names_by_index(deltas)
    assert names.get(0) == "read_file", names

    fragments = _argument_fragments_for_index(deltas, 0)
    combined = "".join(f for f in fragments if f != "")
    decoded = json.loads(combined)
    assert decoded == {
        "path": "/src/main.py"
    }, f"wrapped-mode regression: expected {{'path': '/src/main.py'}}, got {decoded!r}"

    for text in _content_events(deltas):
        assert (
            "<tool_call>" not in text
        ), f"wrapped tool-call markup leaked as content: {text!r}"
        assert (
            "<function=" not in text
        ), f"wrapped tool-call markup leaked as content: {text!r}"


def test_function_prefix_in_parameter_value_not_counted_as_tool_boundary():
    pytest.skip(
        "fusion_mlx Qwen3CoderToolParser._function_start_positions not migrated yet"
    )
    parser = Qwen3CoderToolParser(tokenizer=None)

    text = (
        "<function=echo>"
        "<parameter=text>call <function=inner> in a code snippet</parameter>"
        "</function>"
    )
    starts = parser._function_start_positions(text)
    assert starts == [
        0
    ], f"top-level ``<function=`` scanner miscounted: expected [0], got {starts!r}"


def test_function_end_in_parameter_value_not_counted_as_tool_close():
    pytest.skip(
        "fusion_mlx Qwen3CoderToolParser._top_level_function_close_count not migrated yet"
    )
    parser = Qwen3CoderToolParser(tokenizer=None)

    text = "<function=echo><parameter=code>foo</function>bar</parameter></function>"
    starts = parser._function_start_positions(text)
    close_count = parser._top_level_function_close_count(text, starts)
    assert close_count == 1, (
        f"top-level ``</function>`` scanner miscounted: expected 1, "
        f"got {close_count}. starts={starts!r}"
    )


def test_streaming_state_not_corrupted_by_function_prefix_in_value():
    pytest.skip(
        "fusion_mlx Qwen3CoderToolParser bare-wrapper streaming not migrated yet"
    )
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("echo", {"note": {"type": "string"}})

    chunks = [
        "<function=echo>",
        "<parameter=note>",
        # Full inner XML — the naive scanner would see 2 openers and 2
        # closers here and try to advance mid-argument.
        "see also <function=inner></function> below",
        "</parameter>",
        "</function>",
        # Trailing content after the real tool closes — the naive
        # scanner would still think ``is_tool_call_started`` is True
        # and try to slice a phantom second block instead of emitting
        # this as a content event.
        " done.",
    ]

    deltas = _feed(parser, chunks, request)
    names = _names_by_index(deltas)
    assert set(names.keys()) == {
        0
    }, f"phantom tool index emitted; names={names!r}, deltas={deltas!r}"
    assert names[0] == "echo"
    # Trailing content after the real tool close must reach the
    # content stream, not vanish into a phantom advance.
    contents = _content_events(deltas)
    assert any(
        "done" in c for c in contents
    ), f"trailing content swallowed by phantom advance; contents={contents!r}"


def test_content_before_bare_function_is_emitted_as_content():
    pytest.skip(
        "fusion_mlx Qwen3CoderToolParser bare-wrapper streaming not migrated yet"
    )
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("read_file", {"path": {"type": "string"}})

    # Fine-grained chunking so the streaming state machine has enough
    # deltas to reach the params-loop after (a) emitting content-before,
    # (b) parsing the header, and (c) emitting the JSON opener ``{``.
    # This mirrors the per-token deltas real streams deliver.
    chunks = [
        "Let me read that file. ",
        "<function=read_file>",
        "<parameter=path>",
        "/src/main.py",
        "</parameter>",
        "</function>",
    ]

    deltas = _feed(parser, chunks, request)

    contents = _content_events(deltas)
    assert any(
        "Let me read that file. " in c for c in contents
    ), f"prose before <function=…> was dropped. contents={contents!r}"
    # And the tool call still fired.
    names = _names_by_index(deltas)
    assert names.get(0) == "read_file", names
    combined = "".join(f for f in _argument_fragments_for_index(deltas, 0) if f != "")
    assert json.loads(combined) == {"path": "/src/main.py"}
