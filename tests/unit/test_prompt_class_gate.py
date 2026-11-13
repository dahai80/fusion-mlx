# SPDX-License-Identifier: Apache-2.0
"""P2-2 prompt-class gate: classifier + spec gating rules."""

from fusion_mlx.speculative.prompt_class import (
    CLASS_CHAT,
    CLASS_CODE,
    CLASS_MATH,
    CLASS_OTHER,
    classify_prompt,
    extract_last_user_turn,
    spec_gate_allows,
)

_RENDERED_CHAT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nHey, I had a long day at work and just want to chat "
    "about movies and music. What did you watch lately?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class TestClassifyPrompt:
    def test_code_markers(self):
        assert classify_prompt("Write a function:\n```python\ndef f(x):\n    return x\n```") == CLASS_CODE
        assert classify_prompt("import os\nfrom typing import Any\n" + "x" * 30) == CLASS_CODE

    def test_code_hints(self):
        assert classify_prompt("Please write a python function that merges two lists " * 2) == CLASS_CODE

    def test_math_operators(self):
        assert classify_prompt("Solve step by step: if 3*x + 7 = 22, what is x? Show work " * 2) == CLASS_MATH

    def test_math_two_hints(self):
        assert classify_prompt("Compute the integral and then compute the probability of the outcome " * 2) == CLASS_MATH

    def test_open_chat(self):
        assert classify_prompt("Hey, I had a long day at work and just want to chat about movies and music " * 2) == CLASS_CHAT

    def test_short_returns_other(self):
        assert classify_prompt("hi there") == CLASS_OTHER

    def test_token_ids_return_other(self):
        assert classify_prompt([1, 2, 3, 4, 5]) == CLASS_OTHER

    def test_none_returns_other(self):
        assert classify_prompt(None) == CLASS_OTHER

    def test_extract_last_user_turn(self):
        assert extract_last_user_turn(_RENDERED_CHAT) == (
            "Hey, I had a long day at work and just want to chat about "
            "movies and music. What did you watch lately?"
        )

    def test_extract_strips_special_tokens_when_no_user_tag(self):
        out = extract_last_user_turn("<|system|>be helpful<|eot|>plain text here")
        assert out == "be helpful plain text here"

    def test_rendered_chat_template_classifies_chat(self):
        # The gate pipeline (spec_decode._classify_request_prompt) must
        # extract the user turn first — the raw render classifies "other"
        # because special-token markers inflate operator density.
        from types import SimpleNamespace

        from fusion_mlx.scheduler.spec_decode import _classify_request_prompt

        req = SimpleNamespace(
            prompt=_RENDERED_CHAT,
            sampling_params=SimpleNamespace(temperature=0.7),
        )
        assert _classify_request_prompt(req) == CLASS_CHAT
        assert classify_prompt(_RENDERED_CHAT) == CLASS_OTHER


class TestSpecGateAllows:
    def test_math_always(self):
        assert spec_gate_allows(CLASS_MATH, 0.7)
        assert spec_gate_allows(CLASS_MATH, 0.0)

    def test_code_always(self):
        assert spec_gate_allows(CLASS_CODE, 0.7)

    def test_chat_greedy_only(self):
        assert spec_gate_allows(CLASS_CHAT, 0.0)
        assert not spec_gate_allows(CLASS_CHAT, 0.7)

    def test_other_ungated(self):
        assert spec_gate_allows(CLASS_OTHER, 0.7)
