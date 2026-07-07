# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.api.constants — rescue payload helper."""

from __future__ import annotations

from fusion_mlx.api.constants import (
    REASONING_CUTOFF_SENTINEL,
    RESCUE_TAIL_LENGTH,
    is_rescue_payload,
)


class TestConstants:
    def test_sentinel_value(self):
        assert (
            REASONING_CUTOFF_SENTINEL
            == "[truncated — reasoning incomplete; raise max_tokens]"
        )

    def test_tail_length(self):
        assert RESCUE_TAIL_LENGTH == 200


class TestIsRescuePayload:
    def test_none_is_not_rescue(self):
        assert is_rescue_payload(None) is False

    def test_empty_is_not_rescue(self):
        assert is_rescue_payload("") is False

    def test_exact_sentinel_is_rescue(self):
        assert is_rescue_payload(REASONING_CUTOFF_SENTINEL) is True

    def test_sentinel_with_body_is_rescue(self):
        payload = REASONING_CUTOFF_SENTINEL + "\n\n" + "tail content"
        assert is_rescue_payload(payload) is True

    def test_sentinel_prefix_only_not_rescue(self):
        # prefix alone (no body after \n\n) is NOT rescue — len == prefix len
        prefix = REASONING_CUTOFF_SENTINEL + "\n\n"
        assert is_rescue_payload(prefix) is False

    def test_unrelated_content_not_rescue(self):
        assert is_rescue_payload("just some reasoning text") is False

    def test_sentinel_substring_not_rescue(self):
        # content starting with sentinel but shorter than prefix+body not rescue
        assert is_rescue_payload(REASONING_CUTOFF_SENTINEL + "\n") is False
