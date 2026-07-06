# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

openai_harmony = pytest.importorskip("openai_harmony")

from openai_harmony import (  # noqa: E402
    HarmonyEncodingName,
    load_harmony_encoding,
)

from fusion_mlx.output_router import Channel, TokenMap  # noqa: E402
from fusion_mlx.output_router_harmony import HarmonyStreamingRouter  # noqa: E402

from ._harmony_markers import HARMONY_LEAK_MARKERS  # noqa: E402


@pytest.fixture(scope="module")
def encoding():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


class _HarmonyDecodeAdapter:
    def __init__(self, enc):
        self._enc = enc

    def decode(self, ids):
        return self._enc.decode(ids)

    def get_vocab(self):
        return {}


@pytest.fixture
def router(encoding):
    tm = TokenMap(format_tag="harmony")
    return HarmonyStreamingRouter(tm, _HarmonyDecodeAdapter(encoding))


def _encode(encoding, text: str) -> list[int]:
    return encoding.encode(text, allowed_special="all")


def test_issue_444_480_commentary_tool_call_no_marker_leak(router, encoding):
    text = (
        "<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>"
        '{"city":"NYC"}<|call|>'
    )
    tokens = _encode(encoding, text)
    result = router.feed_sequence(tokens)

    assert result["content"] is None, (
        f"#444/#480: tool-call commentary stream must NOT leak into "
        f"content; got content={result['content']!r}"
    )
    assert result["reasoning"] is None, (
        f"#444/#480: no analysis channel in this sequence — reasoning "
        f"must stay empty; got reasoning={result['reasoning']!r}"
    )
    assert result["tool_calls"] is not None and len(result["tool_calls"]) == 1, (
        f"#444/#480: tool call must surface; got tool_calls={result['tool_calls']!r}"
    )
    tc = result["tool_calls"][0]
    assert isinstance(tc, dict), f"#444/#480: tool_call must be dict; got {type(tc)}"
    assert tc["name"] == "get_weather", (
        f"#444/#480: tool call must carry recipient name; got {tc!r}"
    )
    assert tc["arguments"] == '{"city":"NYC"}', (
        f"#444/#480: tool call arguments must be verbatim body bytes; got {tc!r}"
    )
    assert "<|channel|>" not in tc["arguments"]
    assert "<|message|>" not in tc["arguments"]
    assert "<|call|>" not in tc["arguments"]


def test_issue_455_analysis_then_commentary_separates_channels(router, encoding):
    text = (
        "<|channel|>analysis<|message|>"
        "I'll fetch the weather.<|end|>"
        "<|start|>assistant<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>"
        '{"city":"Paris"}<|call|>'
    )
    tokens = _encode(encoding, text)
    result = router.feed_sequence(tokens)

    assert result["content"] is None, (
        f"#455: content must stay empty; got {result['content']!r}"
    )
    assert result["reasoning"] == "I'll fetch the weather.", (
        f"#455: reasoning must carry analysis body; got {result['reasoning']!r}"
    )
    assert result["tool_calls"] is not None and len(result["tool_calls"]) == 1, (
        f"#455: tool call must surface; got tool_calls={result['tool_calls']!r}"
    )
    tc = result["tool_calls"][0]
    assert tc == {"name": "get_weather", "arguments": '{"city":"Paris"}'}, (
        f"#455: structured payload mismatch; got {tc!r}"
    )

    for ch_name in ("content", "reasoning"):
        val = result.get(ch_name) or ""
        for marker in HARMONY_LEAK_MARKERS:
            assert marker not in val, (
                f"#455: marker {marker!r} leaked into {ch_name}; got {val!r}"
            )


def test_issue_468_compound_analysis_commentary_final_separates(router, encoding):
    text = (
        "<|channel|>analysis<|message|>"
        "Need to compute the sum.<|end|>"
        "<|start|>assistant<|channel|>commentary "
        "to=functions.add <|constrain|>json<|message|>"
        '{"a":1,"b":2}<|call|>'
        "<|start|>assistant<|channel|>final<|message|>"
        "The answer is 3.<|return|>"
    )
    tokens = _encode(encoding, text)
    result = router.feed_sequence(tokens)

    assert result["reasoning"] == "Need to compute the sum.", (
        f"#468: reasoning must carry analysis; got {result['reasoning']!r}"
    )
    assert result["content"] == "The answer is 3.", (
        f"#468: content must carry final body; got {result['content']!r}"
    )
    assert result["tool_calls"] is not None and len(result["tool_calls"]) == 1, (
        f"#468: one tool call must surface; got {result['tool_calls']!r}"
    )

    tc = result["tool_calls"][0]
    assert tc == {"name": "add", "arguments": '{"a":1,"b":2}'}, (
        f"#468: structured payload mismatch; got {tc!r}"
    )

    for ch_name in ("content", "reasoning"):
        val = result.get(ch_name) or ""
        for marker in HARMONY_LEAK_MARKERS:
            assert marker not in val, (
                f"#468: marker {marker!r} leaked into {ch_name}; got {val!r}"
            )


def test_structured_payload_preserves_body_with_harmony_sentinels(router, encoding):
    sentinel_bearing_bodies = (
        '{"text":"use <|call|>"}',
        '{"text":"<|message|>injected"}',
        '{"x":"<|channel|>commentary"}',
        '{"y":"<|end|>"}',
        '{"z":"<|return|>"}',
        '{"a":"<|start|>"}',
        '{"b":"<|constrain|>json"}',
    )
    prefix_ids = encoding.encode(
        "<|channel|>commentary to=functions.echo <|constrain|>json<|message|>",
        allowed_special="all",
    )
    suffix_ids = encoding.encode("<|call|>", allowed_special="all")
    for body in sentinel_bearing_bodies:
        body_ids = encoding.encode(body, allowed_special="none", disallowed_special=())
        tokens = prefix_ids + body_ids + suffix_ids
        router.reset()
        result = router.feed_sequence(tokens)

        assert result["tool_calls"] is not None, (
            f"body {body!r}: tool call MUST surface (round-15 "
            f"bytes-faithful refactor); got tool_calls=None"
        )
        assert len(result["tool_calls"]) == 1, (
            f"body {body!r}: exactly one tool call expected; "
            f"got {result['tool_calls']!r}"
        )
        tc = result["tool_calls"][0]
        assert tc == {"name": "echo", "arguments": body}, (
            f"body {body!r}: structured payload must preserve bytes; got {tc!r}"
        )
        assert result["content"] is None, (
            f"body {body!r}: must NOT leak into content under "
            f"bytes-faithful refactor; got {result['content']!r}"
        )


def test_structured_payload_carries_truncated_recipient_namespace(router):
    class _C:
        def __init__(self, t):
            self.text = t

    class _Msg:
        recipient = "functions.my-tool-with-dash_and_under"
        content = [_C('{"k":"v"}')]
        content_type = None

    out = router._extract_structured_tool_call(_Msg())
    assert out is not None
    assert out["name"] == "my-tool-with-dash_and_under"
    assert out["arguments"] == '{"k":"v"}'


def test_structured_payload_drops_malformed_recipient(router):
    class _C:
        def __init__(self, t):
            self.text = t

    class _Msg:
        def __init__(self, recipient):
            self.recipient = recipient
            self.content = [_C('{"x":1}')]
            self.content_type = None

    bad_recipients = (
        "functions.bad name",
        "functions.<|call|>",
        "not_functions.x",
        "functions.",
    )
    for bad in bad_recipients:
        out = router._extract_structured_tool_call(_Msg(bad))
        assert out is None, (
            f"malformed recipient {bad!r} must drop (return None); got {out!r}"
        )


def test_structured_payload_drops_empty_recipient(router):
    class _C:
        def __init__(self, t):
            self.text = t

    class _Msg:
        recipient = None
        content = [_C("x")]
        content_type = None

    assert router._extract_structured_tool_call(_Msg()) is None


def test_per_token_streaming_routes_one_event_per_body_token(router, encoding):
    text = "<|channel|>final<|message|>Hi there.<|return|>"
    tokens = _encode(encoding, text)
    message_id = encoding.encode("<|message|>", allowed_special="all")[0]
    return_id = encoding.encode("<|return|>", allowed_special="all")[0]
    body_start = tokens.index(message_id) + 1
    body_end = tokens.index(return_id)
    body_token_count = body_end - body_start

    router.reset()
    events_per_channel: dict[Channel, list[str]] = {
        Channel.CONTENT: [],
        Channel.REASONING: [],
        Channel.TOOL_CALL: [],
    }
    for tid in tokens:
        ev = router.feed(tid)
        if ev is None:
            continue
        events_per_channel[ev.channel].append(ev.text)

    assert events_per_channel[Channel.TOOL_CALL] == []
    assert events_per_channel[Channel.REASONING] == []
    assert "".join(events_per_channel[Channel.CONTENT]) == "Hi there."
    assert len(events_per_channel[Channel.CONTENT]) == body_token_count, (
        f"per-token body deltas must be 1-to-1 with body tokens; "
        f"got {len(events_per_channel[Channel.CONTENT])} events for "
        f"{body_token_count} body tokens: "
        f"{events_per_channel[Channel.CONTENT]!r}"
    )


def test_tool_call_event_carries_structured_payload(router, encoding):
    text = (
        "<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>"
        '{"city":"NYC"}<|call|>'
    )
    tokens = _encode(encoding, text)
    router.reset()
    tool_call_events = []
    for tid in tokens:
        ev = router.feed(tid)
        if ev is not None and ev.channel == Channel.TOOL_CALL:
            tool_call_events.append(ev)
    assert len(tool_call_events) == 1
    ev = tool_call_events[0]
    assert ev.tool_call is not None
    assert ev.tool_call["name"] == "get_weather"
    assert ev.tool_call["arguments"] == '{"city":"NYC"}'


def test_feed_sequence_preserves_leading_and_trailing_whitespace(router, encoding):
    text = "<|channel|>final<|message|>\n```py\nprint('hi')\n```  <|return|>"
    tokens = _encode(encoding, text)
    router.reset()
    result = router.feed_sequence(tokens)

    assert result["content"] == "\n```py\nprint('hi')\n```  ", (
        f"feed_sequence must preserve surrounding whitespace; got {result['content']!r}"
    )


def test_recipient_shape_accepts_digit_start_names(router):
    class _C:
        def __init__(self, t):
            self.text = t

    class _Msg:
        def __init__(self, recipient):
            self.recipient = recipient
            self.content = [_C('{"x":1}')]
            self.content_type = None

    for good in (
        "functions.2fa_lookup",
        "functions.0_index",
        "functions.123",
        "functions.tool-with-dash",
    ):
        out = router._extract_structured_tool_call(_Msg(good))
        assert out is not None, f"good recipient {good!r} dropped; got None"
        assert out["name"] == good.split(".", 1)[1]


def test_compat_gate_rejects_unknown_tokenizer_identity():
    from fusion_mlx.output_router import TokenMap
    from fusion_mlx.output_router_harmony import is_openai_harmony_compatible

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    class _NotGptOssTokenizer:
        name_or_path = "mistralai/Mistral-7B-Instruct-v0.3"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return [99001, 99002]

    assert is_openai_harmony_compatible(tm, _NotGptOssTokenizer()) is False


def test_compat_gate_accepts_gpt_oss_quant_suffix_variants():
    from fusion_mlx.output_router import TokenMap
    from fusion_mlx.output_router_harmony import is_openai_harmony_compatible

    enc = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    def _id(s):
        return enc.encode(s, allowed_special="all")[0]

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=_id("<|channel|>"),
        harmony_message=_id("<|message|>"),
        harmony_call=_id("<|call|>"),
        harmony_end=_id("<|end|>"),
        harmony_return=_id("<|return|>"),
        harmony_start=_id("<|start|>"),
        harmony_constrain=_id("<|constrain|>"),
    )

    class _GptOssLike:
        def __init__(self, name):
            self.name_or_path = name
            self._enc = enc

        def encode(self, text, add_special_tokens=False):
            return self._enc.encode(text, allowed_special="none")

        def decode(self, ids):
            return self._enc.decode(ids)

        def get_vocab(self):
            return {}

    for name in (
        "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "openai/gpt-oss-120b",
        "mlx-community/GPT-OSS-20b",
    ):
        assert is_openai_harmony_compatible(tm, _GptOssLike(name)) is True, (
            f"identity {name!r} must be accepted"
        )

    for name in (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen3-0.6B",
        "",
    ):
        assert is_openai_harmony_compatible(tm, _GptOssLike(name)) is False, (
            f"identity {name!r} must be rejected"
        )


def test_compat_gate_rejects_tokenizer_without_encode():
    from fusion_mlx.output_router import TokenMap
    from fusion_mlx.output_router_harmony import is_openai_harmony_compatible

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    class _NoEncodeTokenizer:
        name_or_path = "mlx-community/gpt-oss-NO-ENCODE-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

    assert is_openai_harmony_compatible(tm, _NoEncodeTokenizer()) is False


def test_compat_gate_rejects_mismatched_body_vocab():
    from fusion_mlx.output_router import TokenMap
    from fusion_mlx.output_router_harmony import is_openai_harmony_compatible

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    class _MismatchedBodyVocabTokenizer:
        name_or_path = "mlx-community/gpt-oss-MISMATCHED-BODY-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return [99001, 99002]

    assert is_openai_harmony_compatible(tm, _MismatchedBodyVocabTokenizer()) is False


def test_compat_gate_anchored_allowlist_rejects_tail_substring_fake():
    from fusion_mlx.output_router import TokenMap
    from fusion_mlx.output_router_harmony import is_openai_harmony_compatible

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    enc = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    class _CompatTokenizerBase:
        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return enc.encode(text, allowed_special="none")

    rejected_names = (
        "my-not-gpt-oss/whatever",
        "foo/bar-gpt-oss-malicious",
        "my-not-gpt-oss-20b",
        "notgpt-oss-fake",
        "some-user/gpt-oss-remapped",
        "evil-org/gpt-oss-20b",
        "anonymous/gpt-oss",
    )
    for name in rejected_names:

        class _Fake(_CompatTokenizerBase):
            pass

        _Fake.name_or_path = name
        assert is_openai_harmony_compatible(tm, _Fake()) is False, (
            f"tail-substring identity {name!r} must be rejected"
        )

    accepted_names = (
        "openai/gpt-oss-20b",
        "openai/gpt-oss-20b-mxfp4-q8",
        "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "unsloth/gpt-oss-20b-MLX-8bit",
        "gpt-oss-20b-mxfp4-q8",
        "gpt-oss",
        "/models/gpt-oss-20b",
        "~/lmstudio-models/gpt-oss-20b",
        "./gpt-oss-20b-quantized",
        "../models/gpt-oss-20b",
    )
    for name in accepted_names:

        class _Real(_CompatTokenizerBase):
            pass

        _Real.name_or_path = name
        assert is_openai_harmony_compatible(tm, _Real()) is True, (
            f"canonical identity {name!r} must pass"
        )


def test_compat_gate_accepts_hf_cache_snapshot_path():
    from fusion_mlx.output_router import TokenMap
    from fusion_mlx.output_router_harmony import is_openai_harmony_compatible

    enc = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    def _id(s):
        return enc.encode(s, allowed_special="all")[0]

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=_id("<|channel|>"),
        harmony_message=_id("<|message|>"),
        harmony_call=_id("<|call|>"),
        harmony_end=_id("<|end|>"),
        harmony_return=_id("<|return|>"),
        harmony_start=_id("<|start|>"),
        harmony_constrain=_id("<|constrain|>"),
    )

    class _CompatTokenizerBase:
        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return enc.encode(text, allowed_special="none")

    accepted_cache_paths = (
        "/Users/me/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/abc123def/",
        "/home/user/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/0f1e2d3c/",
        "/var/lib/hf-cache/models--unsloth--gpt-oss-20b-MLX-8bit/snapshots/fedcba9876/",
    )
    for path in accepted_cache_paths:

        class _CachePath(_CompatTokenizerBase):
            pass

        _CachePath.name_or_path = path
        assert is_openai_harmony_compatible(tm, _CachePath()) is True, (
            f"HF cache snapshot {path!r} must pass (round-14 BLOCKING fix)"
        )

    class _BadCachePath(_CompatTokenizerBase):
        pass

    _BadCachePath.name_or_path = (
        "/home/x/.cache/huggingface/hub/models--evil-org--gpt-oss-20b/snapshots/xyz/"
    )
    assert is_openai_harmony_compatible(tm, _BadCachePath()) is False, (
        "HF cache path with non-allowlisted owner must NOT pass"
    )

    class _NonGptOssCachePath(_CompatTokenizerBase):
        pass

    _NonGptOssCachePath.name_or_path = (
        "/home/x/.cache/huggingface/hub/models--openai--whisper-large/snapshots/xyz/"
    )
    assert is_openai_harmony_compatible(tm, _NonGptOssCachePath()) is False, (
        "HF cache path for non-gpt-oss model must NOT pass"
    )


def test_compat_cache_key_segregates_by_tokenizer_instance():
    from fusion_mlx.output_router import TokenMap
    from fusion_mlx.output_router_harmony import is_openai_harmony_compatible

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    enc = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    class _RealHarmony:
        name_or_path = "mlx-community/gpt-oss-CACHE-INSTANCE-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return enc.encode(text, allowed_special="none")

    assert is_openai_harmony_compatible(tm, _RealHarmony()) is True

    class _RemappedBody:
        name_or_path = "mlx-community/gpt-oss-CACHE-INSTANCE-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return [99001, 99002]

    assert is_openai_harmony_compatible(tm, _RemappedBody()) is False, (
        "cache must segregate by tokenizer instance; "
        "remapped tokenizer shared instance #1's stale True decision"
    )


def test_compat_gate_rejects_non_int_encode_result():
    from fusion_mlx.output_router import TokenMap
    from fusion_mlx.output_router_harmony import is_openai_harmony_compatible

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    class _BatchEncodingShape:
        def __iter__(self):
            return iter(("input_ids", "attention_mask"))

    class _BatchEncodingTokenizer:
        name_or_path = "mlx-community/gpt-oss-BATCH-ENCODING-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return _BatchEncodingShape()

    assert is_openai_harmony_compatible(tm, _BatchEncodingTokenizer()) is False

    class _StringTokenIdsTokenizer:
        name_or_path = "mlx-community/gpt-oss-STRING-IDS-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return ("Hello", "world")

    assert is_openai_harmony_compatible(tm, _StringTokenIdsTokenizer()) is False


def test_compat_gate_rejects_missing_marker_ids():
    from fusion_mlx.output_router import TokenMap
    from fusion_mlx.output_router_harmony import is_openai_harmony_compatible

    enc = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    def _id(s):
        return enc.encode(s, allowed_special="all")[0]

    tm_missing_call = TokenMap(
        format_tag="harmony",
        harmony_channel=_id("<|channel|>"),
        harmony_message=_id("<|message|>"),
        harmony_end=_id("<|end|>"),
        harmony_return=_id("<|return|>"),
        harmony_start=_id("<|start|>"),
        harmony_constrain=_id("<|constrain|>"),
    )

    class _GptOssTokenizer:
        name_or_path = "mlx-community/gpt-oss-MISSING-CALL-VARIANT"

        def encode(self, text, add_special_tokens=False):
            return enc.encode(text, allowed_special="none")

        def decode(self, ids):
            return enc.decode(ids)

        def get_vocab(self):
            return {}

    assert is_openai_harmony_compatible(tm_missing_call, _GptOssTokenizer()) is False


def test_compat_gate_cache_segregates_by_marker_ids():
    from fusion_mlx.output_router import TokenMap
    from fusion_mlx.output_router_harmony import (
        _COMPAT_RESULT_CACHE,
        is_openai_harmony_compatible,
    )

    class _T:
        name_or_path = "mlx-community/gpt-oss-CACHE-KEY-SEGREGATION"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return [0]

    tm_a = TokenMap(format_tag="harmony", harmony_channel=200005)
    tm_b = TokenMap(format_tag="harmony", harmony_channel=999999)

    t = _T()
    is_openai_harmony_compatible(tm_a, t)
    is_openai_harmony_compatible(tm_b, t)

    inner = _COMPAT_RESULT_CACHE.get(t)
    assert inner is not None, "tokenizer must have a cache slot after probes"
    assert len(inner) == 2, (
        f"cache must keep separate inner entries per marker-ID tuple; "
        f"got {len(inner)} entries: {list(inner.keys())!r}"
    )


def test_compat_gate_caches_per_tokenizer_identity():
    from fusion_mlx.output_router import TokenMap
    from fusion_mlx.output_router_harmony import is_openai_harmony_compatible

    enc = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    def _id(s):
        return enc.encode(s, allowed_special="all")[0]

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=_id("<|channel|>"),
        harmony_message=_id("<|message|>"),
        harmony_call=_id("<|call|>"),
        harmony_end=_id("<|end|>"),
        harmony_return=_id("<|return|>"),
        harmony_start=_id("<|start|>"),
        harmony_constrain=_id("<|constrain|>"),
    )

    call_count = {"n": 0}

    class _CountingTokenizer:
        name_or_path = "mlx-community/gpt-oss-CACHE-PROBE-INVOCATION"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            call_count["n"] += 1
            return [0]

    t = _CountingTokenizer()
    result1 = is_openai_harmony_compatible(tm, t)
    assert result1 is False
    first_call_count = call_count["n"]
    assert first_call_count > 0

    result2 = is_openai_harmony_compatible(tm, t)
    assert result2 is False
    assert call_count["n"] == first_call_count, (
        "compat gate must cache False results per identity"
    )


def test_finalize_never_synthesizes_truncated_commentary(router, encoding):
    text = (
        "<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>"
        '{"city":"NYC"}'
    )
    tokens = _encode(encoding, text)
    router.reset()
    routed_during_stream = []
    for tid in tokens:
        ev = router.feed(tid)
        if ev is not None:
            routed_during_stream.append(ev)
    drained = router.finalize()

    assert all(ev.channel != Channel.TOOL_CALL for ev in routed_during_stream)
    assert drained is None, (
        f"truncated commentary must NEVER surface as a tool call; got {drained!r} "
        f"(stream events: {len(routed_during_stream)})"
    )


def test_finalize_drops_mid_json_truncation(router, encoding):
    text = (
        "<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>"
        '{"city":"NY'
    )
    tokens = _encode(encoding, text)
    router.reset()
    for tid in tokens:
        router.feed(tid)
    drained = router.finalize()

    assert drained is None


def test_finalize_drops_truncated_commentary_with_empty_body(router, encoding):
    text = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>"
    tokens = _encode(encoding, text)
    router.reset()
    for tid in tokens:
        router.feed(tid)
    drained = router.finalize()

    assert drained is None


def test_finalize_does_not_double_emit_completed_final(router, encoding):
    text = "<|channel|>final<|message|>Hi there.<|return|>"
    tokens = _encode(encoding, text)
    router.reset()
    streamed = []
    for tid in tokens:
        ev = router.feed(tid)
        if ev is not None:
            streamed.append(ev)
    drained = router.finalize()

    assert drained is None
    assert (
        "".join(e.text for e in streamed if e.channel == Channel.CONTENT) == "Hi there."
    )


def test_finalize_does_not_drain_post_eos_buffered_delta(router, encoding):
    class _StubMessage:
        def __init__(self, channel):
            self.channel = channel
            self.recipient = None
            self.content = []
            self.content_type = None

    class _StubParser:
        def __init__(self, channel, delta):
            self._channel = channel
            self._delta = delta
            self.tokens = [1]
            self.messages: list[_StubMessage] = []

        @property
        def current_channel(self):
            return self._channel

        @property
        def last_content_delta(self):
            return self._delta

        def process_eos(self):
            self.messages.append(_StubMessage(self._channel))
            self._delta = " tail"

    router.reset()
    router._parser = _StubParser("final", "")
    assert router.finalize() is None

    router.reset()
    router._parser = _StubParser("analysis", "")
    assert router.finalize() is None
