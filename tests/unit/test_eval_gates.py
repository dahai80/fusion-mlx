# SPDX-License-Identifier: Apache-2.0
"""G1/S1: eval gate pure-logic tests (no server needed)."""

import json

from fusion_mlx.eval.coherence_gate import _check_response, _has_repetition_collapse
from fusion_mlx.eval.tool_result_grader import _grade


class TestRepetitionCollapse:
    def test_short_text_not_collapse(self):
        assert _has_repetition_collapse("hello world") is False

    def test_repeating_tail_detected(self):
        text = "once upon a time " + "the the " * 10
        assert _has_repetition_collapse(text) is True

    def test_varied_long_text_not_collapse(self):
        text = " ".join(str(i) for i in range(50))
        assert _has_repetition_collapse(text) is False


class TestCheckResponse:
    def test_empty_fails(self):
        passed, reason = _check_response("q", "", lambda r: True)
        assert passed is False
        assert "empty" in reason

    def test_too_short_fails(self):
        passed, _ = _check_response("q", "ab", lambda r: True)
        assert passed is False

    def _checker(self, r):
        return "4" in r

    def test_valid_passes(self):
        passed, reason = _check_response("2+2?", "The answer is 4.", self._checker)
        assert passed is True
        assert reason == ""

    def test_checker_predicate_fail(self):
        passed, reason = _check_response("2+2?", "I don't know.", self._checker)
        assert passed is False
        assert "predicate" in reason

    def test_repetition_collapse_fails(self):
        text = "answer " + "no no " * 10
        passed, reason = _check_response("q", text, lambda r: True)
        assert passed is False
        assert "repetition" in reason


class TestToolGrade:
    PROBE = {
        "prompt": "test",
        "expected_tool": "get_weather",
        "expected_keys": ["location"],
        "expected_values": {"location": "Paris"},
    }

    def test_no_tool_calls_fails(self):
        s = _grade([], self.PROBE)
        assert s.passed is False
        assert "no tool_calls" in s.reason

    def test_wrong_name_fails(self):
        tc = {"function": {"name": "other", "arguments": '{"location": "Paris"}'}}
        s = _grade([tc], self.PROBE)
        assert s.passed is False
        assert s.tool_name_match is False

    def test_invalid_args_fails(self):
        tc = {"function": {"name": "get_weather", "arguments": "{broken"}}
        s = _grade([tc], self.PROBE)
        assert s.passed is False
        assert s.args_valid_json is False

    def test_missing_key_fails(self):
        tc = {"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
        s = _grade([tc], self.PROBE)
        assert s.passed is False
        assert s.keys_present is False

    def test_correct_passes(self):
        tc = {
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"location": "Paris"}),
            }
        }
        s = _grade([tc], self.PROBE)
        assert s.passed is True
        assert s.tool_name_match is True
        assert s.args_valid_json is True
        assert s.keys_present is True
        assert s.values_match is True
