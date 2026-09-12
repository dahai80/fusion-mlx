# SPDX-License-Identifier: Apache-2.0
"""S1.5-ext: error-code solutions registry coverage."""

from fusion_mlx.error_solutions import ERROR_SOLUTIONS_MAP, get_solutions


class TestErrorSolutionsRegistry:
    def test_registry_covers_twelve_common_codes(self):
        expected = {400, 401, 403, 404, 408, 413, 429, 500, 502, 503, 504, 507}
        assert expected.issubset(set(ERROR_SOLUTIONS_MAP.keys()))

    def test_every_code_has_nonempty_solutions(self):
        for code, sols in ERROR_SOLUTIONS_MAP.items():
            assert isinstance(sols, list), f"{code} solutions not a list"
            assert len(sols) >= 2, f"{code} has fewer than 2 suggestions"

    def test_get_solutions_returns_list_for_known(self):
        sols = get_solutions(413)
        assert isinstance(sols, list) and len(sols) >= 2

    def test_get_solutions_empty_for_unknown(self):
        assert get_solutions(999) == []

    def test_get_solutions_safe_on_bad_input(self):
        assert get_solutions("not-a-code") == []

    def test_400_mentions_validation(self):
        assert any(
            "schema" in s.lower() or "validation" in s.lower() or "field" in s.lower()
            for s in ERROR_SOLUTIONS_MAP[400]
        )

    def test_401_mentions_api_key(self):
        assert any("key" in s.lower() for s in ERROR_SOLUTIONS_MAP[401])

    def test_404_mentions_model_or_pull(self):
        joined = " ".join(ERROR_SOLUTIONS_MAP[404]).lower()
        assert "model" in joined or "pull" in joined

    def test_408_mentions_timeout_or_streaming(self):
        joined = " ".join(ERROR_SOLUTIONS_MAP[408]).lower()
        assert "timeout" in joined or "stream" in joined

    def test_502_mentions_gateway_or_retry(self):
        joined = " ".join(ERROR_SOLUTIONS_MAP[502]).lower()
        assert "gateway" in joined or "retry" in joined or "transient" in joined
