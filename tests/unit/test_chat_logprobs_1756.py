# SPDX-License-Identifier: Apache-2.0
# #1756: logprobs/top_logprobs on /v1/chat/completions were accepted then
# silently dropped (response logprobs always null). Covers the plumbing:
# sampling-param pass-through, the ChoiceLogProbs builders, and the
# adapter response gate. All vectors are numpy; no model load.

import numpy as np
import pytest

from fusion_mlx.api.adapters.base import InternalResponse
from fusion_mlx.api.adapters.openai import OpenAIAdapter
from fusion_mlx.api.models import ChoiceLogProbs
from fusion_mlx.service.helpers import (
    _build_choice_logprobs,
    _extract_streaming_token_logprobs,
)

_VOCAB = ["▁Hello", " world", "!", "foo", "bar", "baz", "qux", "zap"]


class _FakeTokenizer:
    def decode(self, ids):
        return "".join(_VOCAB[i] for i in ids)


def _vec(vocab_size=8, hot=None):
    v = np.full(vocab_size, -10.0)
    if hot is not None:
        for idx, lp in hot.items():
            v[idx] = lp
    return v


class _Gen:
    def __init__(self, logprobs, token_ids, new_text="x"):
        self.logprobs = logprobs
        self.new_token_ids = token_ids
        self.new_text = new_text


class TestBuildChoiceLogprobs:
    def test_none_engine_logprobs_returns_none(self):
        assert _build_choice_logprobs(_Gen(None, [1]), _FakeTokenizer(), 3) is None

    def test_no_token_ids_returns_none(self):
        assert _build_choice_logprobs(_Gen([_vec()], []), _FakeTokenizer(), 3) is None

    def test_single_vector_wrapped_as_list(self):
        out = _build_choice_logprobs(
            _Gen(_vec(hot={2: -0.5}), [2]), _FakeTokenizer(), 0
        )
        assert out is not None and out.content is not None
        assert len(out.content) == 1
        entry = out.content[0]
        assert entry.token == "!"
        assert entry.logprob == pytest.approx(-0.5)
        assert entry.top_logprobs == []

    def test_top_k_alternatives_sorted_desc(self):
        vec = _vec(hot={1: -0.1, 4: -1.2, 6: -0.7})
        out = _build_choice_logprobs(_Gen([vec], [4]), _FakeTokenizer(), 2)
        entry = out.content[0]
        assert [t.token for t in entry.top_logprobs] == [" world", "qux"]
        assert entry.logprob == pytest.approx(-1.2)
        assert entry.top_logprobs[0].logprob == pytest.approx(-0.1)

    def test_token_count_mismatch_truncates(self):
        out = _build_choice_logprobs(
            _Gen([_vec(), _vec(), _vec()], [1]), _FakeTokenizer(), 0
        )
        assert len(out.content) == 1


class TestStreamingExtraction:
    def test_single_step_vector(self):
        gen = _Gen(_vec(hot={3: -0.9}), [3])
        out = _extract_streaming_token_logprobs(gen, _FakeTokenizer(), 0)
        assert len(out) == 1
        assert out[0].token == "foo"
        assert out[0].logprob == pytest.approx(-0.9)

    def test_none_logprobs_returns_empty(self):
        assert (
            _extract_streaming_token_logprobs(_Gen(None, [1]), _FakeTokenizer(), 3)
            == []
        )


class TestSamplingParamsPassThrough:
    def test_request_flags_reach_sampling_params(self):
        from fusion_mlx.api.openai._common import _build_sampling_params
        from fusion_mlx.api.openai_models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            logprobs=True,
            top_logprobs=3,
        )
        sampling = _build_sampling_params(req)
        assert sampling.logprobs is True
        assert sampling.top_logprobs == 3

    def test_default_off(self):
        from fusion_mlx.api.openai._common import _build_sampling_params
        from fusion_mlx.api.openai_models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        sampling = _build_sampling_params(req)
        assert not sampling.logprobs


class TestAdapterResponseGate:
    def _request(self, logprobs=None, top_logprobs=None):
        from fusion_mlx.api.openai_models import ChatCompletionRequest

        return ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )

    def _internal(self, logprobs):
        return InternalResponse(
            text="hello",
            finish_reason="stop",
            prompt_tokens=2,
            completion_tokens=2,
            request_id="chatcmpl-x",
            logprobs=logprobs,
        )

    def test_populated_when_requested(self):
        clp = _build_choice_logprobs(
            _Gen([_vec(hot={1: -0.3})], [1]), _FakeTokenizer(), 2
        )
        resp = OpenAIAdapter().format_response(
            self._internal(clp), self._request(True, 2)
        )
        assert resp.choices[0].logprobs is clp
        assert resp.choices[0].logprobs.content[0].token == " world"

    def test_null_when_not_requested(self):
        clp = ChoiceLogProbs(content=[])
        resp = OpenAIAdapter().format_response(self._internal(clp), self._request())
        assert resp.choices[0].logprobs is None

    def test_null_when_requested_but_engine_returned_none(self):
        # Spec-decode paths leave the engine field None; the wire shape must
        # stay null rather than fabricate an empty block.
        resp = OpenAIAdapter().format_response(
            self._internal(None), self._request(True, 2)
        )
        assert resp.choices[0].logprobs is None
