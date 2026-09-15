# SPDX-License-Identifier: Apache-2.0
"""Tests for logging configuration filters."""

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from fusion_mlx.logging_config import AdminStatsAccessFilter, configure_file_logging


class TestAdminStatsAccessFilter:
    """Tests for the admin polling access log filter."""

    def setup_method(self):
        self.filter = AdminStatsAccessFilter()

    def _make_record(self, msg: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_suppresses_admin_stats(self):
        record = self._make_record('127.0.0.1 - "GET /admin/api/stats HTTP/1.1" 200')
        assert self.filter.filter(record) is False

    def test_suppresses_admin_stats_with_params(self):
        record = self._make_record(
            '127.0.0.1 - "GET /admin/api/stats?scope=alltime HTTP/1.1" 200'
        )
        assert self.filter.filter(record) is False

    def test_suppresses_admin_login(self):
        record = self._make_record('127.0.0.1 - "POST /admin/api/login HTTP/1.1" 200')
        assert self.filter.filter(record) is False

    def test_allows_other_requests(self):
        record = self._make_record('127.0.0.1 - "GET /v1/models HTTP/1.1" 200')
        assert self.filter.filter(record) is True

    def test_allows_health_check(self):
        record = self._make_record('127.0.0.1 - "GET /health HTTP/1.1" 200')
        assert self.filter.filter(record) is True

    def test_allows_chat_completions(self):
        record = self._make_record(
            '127.0.0.1 - "POST /v1/chat/completions HTTP/1.1" 200'
        )
        assert self.filter.filter(record) is True


class TestConfigureFileLogging:
    def test_adds_handler_on_first_call(self, tmp_path):
        root = logging.getLogger()
        handlers_before = len(root.handlers)
        configure_file_logging(tmp_path)
        handlers_after = len(root.handlers)
        assert handlers_after == handlers_before + 1
        added = root.handlers[-1]
        assert isinstance(added, TimedRotatingFileHandler)
        assert Path(added.baseFilename).name == "server.log"

    def test_second_call_does_not_duplicate(self, tmp_path):
        root = logging.getLogger()
        handlers_before = len(root.handlers)
        configure_file_logging(tmp_path)
        configure_file_logging(tmp_path)
        assert len(root.handlers) == handlers_before + 1
        assert (
            sum(
                1
                for h in root.handlers
                if isinstance(h, TimedRotatingFileHandler)
                and Path(h.baseFilename).resolve()
                == (tmp_path / "server.log").resolve()
            )
            == 1
        )

    def test_second_call_updates_level(self, tmp_path):
        root = logging.getLogger()
        configure_file_logging(tmp_path, level="INFO")
        configure_file_logging(tmp_path, level="DEBUG")
        handler = next(
            h
            for h in root.handlers
            if isinstance(h, TimedRotatingFileHandler)
            and Path(h.baseFilename).resolve() == (tmp_path / "server.log").resolve()
        )
        assert handler.level == logging.DEBUG
