import time
from unittest.mock import patch

from fusion_mlx.api.utils import SSEKeepalive, cap_max_tokens_to_context


class TestSSEKeepalive:
    def test_ping_after_interval(self):
        ka = SSEKeepalive(interval_seconds=0.05)
        ka.reset()
        time.sleep(0.06)
        ping = ka.maybe_ping()
        assert ping == ": keepalive\n\n"

    def test_no_ping_before_interval(self):
        ka = SSEKeepalive(interval_seconds=10.0)
        ka.reset()
        assert ka.maybe_ping() is None

    def test_no_double_ping(self):
        ka = SSEKeepalive(interval_seconds=0.05)
        ka.reset()
        time.sleep(0.06)
        assert ka.maybe_ping() == ": keepalive\n\n"
        assert ka.maybe_ping() is None

    def test_reset_prevents_ping(self):
        ka = SSEKeepalive(interval_seconds=0.05)
        ka.reset()
        time.sleep(0.06)
        ka.reset()
        assert ka.maybe_ping() is None

    def test_zero_interval_disables(self):
        ka = SSEKeepalive(interval_seconds=0)
        ka.reset()
        time.sleep(0.01)
        assert ka.maybe_ping() is None

    def test_negative_interval_disables(self):
        ka = SSEKeepalive(interval_seconds=-1.0)
        ka.reset()
        assert ka.maybe_ping() is None


class TestContextScaling:
    @patch("fusion_mlx.server.get_max_context_window", return_value=8192)
    def test_fits_within_context(self, _mock):
        assert cap_max_tokens_to_context(4096, "m", 0) == 4096

    @patch("fusion_mlx.server.get_max_context_window", return_value=8192)
    def test_caps_at_context_minus_margin(self, _mock):
        assert cap_max_tokens_to_context(8192, "m", 0) == 8128

    @patch("fusion_mlx.server.get_max_context_window", return_value=8192)
    def test_caps_with_prompt_estimate(self, _mock):
        assert cap_max_tokens_to_context(8192, "m", 4000) == 4128

    @patch("fusion_mlx.server.get_max_context_window", return_value=None)
    def test_no_cap_when_no_context_window(self, _mock):
        assert cap_max_tokens_to_context(99999, "m", 0) == 99999

    @patch("fusion_mlx.server.get_max_context_window", return_value=128)
    def test_minimum_cap(self, _mock):
        result = cap_max_tokens_to_context(8192, "m", 100)
        assert result == 64
