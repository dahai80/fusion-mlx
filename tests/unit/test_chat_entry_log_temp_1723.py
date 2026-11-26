import logging

from fusion_mlx.api.openai.chat import _log_chat_entry


class _Req:
    model = "m"
    stream = False
    max_tokens = 8
    messages = []


def _capture(caplog):
    return caplog.records[-1].getMessage()


class TestChatEntryLogTemp:
    def test_explicit_zero_logged_as_zero(self, caplog):
        req = _Req()
        req.temperature = 0
        with caplog.at_level(logging.INFO, logger="fusion_mlx.api.openai._common"):
            _log_chat_entry(req, "hello")
        assert "temp=0" in _capture(caplog)

    def test_explicit_fractional_logged(self, caplog):
        req = _Req()
        req.temperature = 0.3
        with caplog.at_level(logging.INFO, logger="fusion_mlx.api.openai._common"):
            _log_chat_entry(req, "hello")
        assert "temp=0.3" in _capture(caplog)

    def test_missing_falls_back_to_default(self, caplog):
        req = _Req()
        with caplog.at_level(logging.INFO, logger="fusion_mlx.api.openai._common"):
            _log_chat_entry(req, "hello")
        assert "temp=0.7" in _capture(caplog)

    def test_none_falls_back_to_default(self, caplog):
        req = _Req()
        req.temperature = None
        with caplog.at_level(logging.INFO, logger="fusion_mlx.api.openai._common"):
            _log_chat_entry(req, "hello")
        assert "temp=0.7" in _capture(caplog)
