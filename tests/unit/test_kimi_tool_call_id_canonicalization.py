from fusion_mlx.engines.batched import (
    _KIMI_TOOL_CALL_START,
    _canonicalize_kimi_tool_call_ids,
)


def _assistant(id_, name, args='{"a": 1}'):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": id_,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    }


def test_openai_style_ids_rewritten_to_canonical():
    messages = [
        {"role": "user", "content": "hi"},
        _assistant("call_52ed92f8", "sort"),
        {"role": "tool", "tool_call_id": "call_52ed92f8", "content": "ok"},
    ]
    out = _canonicalize_kimi_tool_call_ids(messages)
    tc = out[1]["tool_calls"][0]
    assert tc["id"] == "functions.sort:0"
    assert out[2]["tool_call_id"] == "functions.sort:0"
    # Original dicts untouched (prompt-copy semantics).
    assert messages[1]["tool_calls"][0]["id"] == "call_52ed92f8"


def test_canonical_ids_left_alone():
    messages = [_assistant("functions.sort:0", "sort")]
    out = _canonicalize_kimi_tool_call_ids(messages)
    assert out[0]["tool_calls"][0]["id"] == "functions.sort:0"


def test_multiple_calls_get_stable_indices():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _assistant("call_aaa", "read")["tool_calls"][0],
                _assistant("call_bbb", "write")["tool_calls"][0],
            ],
        }
    ]
    out = _canonicalize_kimi_tool_call_ids(messages)
    ids = [tc["id"] for tc in out[0]["tool_calls"]]
    assert ids == ["functions.read:0", "functions.write:1"]


def test_missing_name_falls_back_to_original_id():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "", "arguments": "{}"},
                }
            ],
        }
    ]
    out = _canonicalize_kimi_tool_call_ids(messages)
    assert out[0]["tool_calls"][0]["id"] == "call_x"


def test_tool_result_without_matching_call_untouched():
    messages = [{"role": "tool", "tool_call_id": "call_orphan", "content": "ok"}]
    out = _canonicalize_kimi_tool_call_ids(messages)
    assert out[0]["tool_call_id"] == "call_orphan"


def test_kimi_tool_call_start_marker_constant():
    # Guard the detection value against mlx_lm drift.
    assert _KIMI_TOOL_CALL_START == "<|tool_calls_section_begin|>"
