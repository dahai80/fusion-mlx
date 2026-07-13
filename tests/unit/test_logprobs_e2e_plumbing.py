# SPDX-License-Identifier: Apache-2.0
# E2E plumbing tests for #99: logprobs copy-through
# Response -> RequestOutput -> GenerationOutput -> routes extraction.

import pytest

from fusion_mlx.engine.base import GenerationOutput
from fusion_mlx.output_collector import RequestOutputCollector
from fusion_mlx.request import RequestOutput
from fusion_mlx.service.helpers import _extract_streaming_token_logprobs


class _FakeTokenizer:
    def decode(self, ids):
        return "".join(f"<t{i}>" for i in ids)


class TestGenerationOutputLogprobsFields:
    def test_fields_default_none(self):
        out = GenerationOutput(text="x")
        assert out.logprobs is None
        assert out.new_token_ids == []

    def test_fields_populated(self):
        out = GenerationOutput(text="x", logprobs="arr", new_token_ids=[42])
        assert out.logprobs == "arr"
        assert out.new_token_ids == [42]

    def test_extract_skips_when_logprobs_none(self):
        chunk = GenerationOutput(text="x", new_text="x", new_token_ids=[1])
        assert _extract_streaming_token_logprobs(chunk, _FakeTokenizer(), 1) == []


class TestExtractStreamingLogprobsPlumbing:
    def test_extracts_token_logprob_from_generation_output(self):
        # Real-mlx gate: the CI stub provides ``mlx`` but not ``mlx.utils``,
        # so importorskip on ``mlx.utils`` skips under the stub and runs
        # locally where a real vocab vector exercises the astype/argpartition
        # path end-to-end.
        pytest.importorskip("mlx.utils")
        import mlx.core as mx

        vocab = mx.array([-3.0, -2.0, -0.1, -5.0])
        chunk = GenerationOutput(
            text="<t2>",
            new_text="<t2>",
            logprobs=vocab,
            new_token_ids=[2],
        )
        out = _extract_streaming_token_logprobs(chunk, _FakeTokenizer(), top_k=2)
        assert len(out) == 1
        tlp = out[0]
        assert tlp.token == "<t2>"
        assert abs(tlp.logprob - (-0.1)) < 1e-5
        assert len(tlp.top_logprobs) == 2
        assert tlp.top_logprobs[0].logprob >= tlp.top_logprobs[1].logprob


class TestCollectorMergesLogprobArrays:
    def test_merge_accumulates_per_token_arrays(self):
        collector = RequestOutputCollector()
        lp_a = object()
        lp_b = object()
        existing = RequestOutput(
            request_id="test-001",
            new_token_ids=[100],
            new_text="a",
            logprobs=lp_a,
        )
        new = RequestOutput(
            request_id="test-001",
            new_token_ids=[101],
            new_text="b",
            logprobs=lp_b,
        )
        result = collector._merge_outputs(existing, new)
        assert isinstance(result.logprobs, list)
        assert result.logprobs == [lp_a, lp_b]

    def test_merge_keeps_none_when_no_logprobs(self):
        collector = RequestOutputCollector()
        existing = RequestOutput(request_id="t", new_token_ids=[1], new_text="a")
        new = RequestOutput(request_id="t", new_token_ids=[2], new_text="b")
        result = collector._merge_outputs(existing, new)
        assert result.logprobs is None
