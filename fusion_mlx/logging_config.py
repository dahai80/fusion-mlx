# SPDX-License-Identifier: Apache-2.0
"""Centralized logging configuration for fusion-mlx.

Provides standard + structured JSON logging, request-context tracking via a
ContextVar, daily-rotated file logging, and suppression of repetitive admin
polling access logs. Ported from omlx with fusion-mlx logger names.
"""

import json
import logging
import sys
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# Context variable for request ID tracking — set per HTTP request by the
# request_id ASGI middleware so log records emitted during handling can be
# correlated back to the request that produced them.
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Custom TRACE level (5) — below DEBUG, includes full message content.
TRACE = 5


class RequestContextFilter(logging.Filter):
    """Add the current request_id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get() or "-"
        return True


class AdminStatsAccessFilter(logging.Filter):
    """Suppress repetitive uvicorn access logs for admin polling endpoints."""

    _SUPPRESSED = (
        "/admin/api/stats",
        "/admin/api/login",
        "/admin/api/hf/tasks",
        "/admin/api/oq/tasks",
        "/admin/api/ms/tasks",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for path in self._SUPPRESSED:
            if path in msg:
                return False
        return True


class ColoredFormatter(logging.Formatter):
    """ANSI-colored log formatter for terminal output."""

    COLORS = {
        TRACE: "\033[90m",                 # Gray (TRACE)
        logging.DEBUG: "\033[36m",         # Cyan
        logging.INFO: "\033[32m",          # Green
        logging.WARNING: "\033[33m",       # Yellow
        logging.ERROR: "\033[31m",         # Red
        logging.CRITICAL: "\033[35m",      # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging / log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None)
        if request_id and request_id != "-":
            log_data["request_id"] = request_id

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        for key in ("request_id", "model", "tokens", "latency_ms"):
            if hasattr(record, key) and key not in log_data:
                log_data[key] = getattr(record, key)

        return json.dumps(log_data)


def get_request_id() -> str | None:
    """Get the current request ID from context."""
    return _request_id.get()


def set_request_id(request_id: str | None) -> None:
    """Set the current request ID in context."""
    _request_id.set(request_id)


def configure_logging(
    level: str = "INFO",
    format_style: str = "standard",
    include_request_id: bool = True,
    colored: bool = True,
) -> int:
    """Configure console (stderr) logging on the root logger.

    Returns the resolved numeric log level so callers (e.g. cli_serve wiring
    uvicorn) can reuse it. Clears existing root handlers to avoid duplicates.
    """
    level_name = level.upper()
    log_level = TRACE if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)

    if include_request_id:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
    else:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)

    if format_style == "json":
        formatter = JsonFormatter(format_str)
    elif colored and sys.stderr.isatty():
        formatter = ColoredFormatter(format_str)
    else:
        formatter = logging.Formatter(format_str)

    handler.setFormatter(formatter)

    if include_request_id:
        handler.addFilter(RequestContextFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # fusion-mlx logger hierarchy
    logging.getLogger("fusion_mlx").setLevel(log_level)
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").addFilter(AdminStatsAccessFilter())

    # Suppress noisy third-party loggers unless trace level
    third_party_level = log_level if log_level <= TRACE else logging.INFO
    logging.getLogger("httpx").setLevel(third_party_level)
    logging.getLogger("httpcore").setLevel(third_party_level)

    return log_level


def get_logger(name: str, request_id: str | None = None) -> logging.Logger:
    """Get a logger, optionally seeding the request-id context."""
    if request_id:
        set_request_id(request_id)
    return logging.getLogger(name)


class RequestLogContext:
    """Context manager for request-scoped logging.

    Usage:
        with RequestLogContext(request_id="abc123"):
            logger.info("Processing request")
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.previous_id: str | None = None

    def __enter__(self) -> "RequestLogContext":
        self.previous_id = _request_id.get()
        _request_id.set(self.request_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _request_id.set(self.previous_id)


def configure_file_logging(
    log_dir: Path,
    level: str = "INFO",
    include_request_id: bool = True,
    retention_days: int = 7,
) -> Path:
    """Add a daily-rotated file handler writing to {log_dir}/server.log.

    Returns the resolved log directory path. Old rotated files are deleted
    after ``retention_days``. Safe to call after ``configure_logging`` —
    appends to the root logger without disturbing the console handler.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    level_name = level.upper()
    log_level = TRACE if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)

    if include_request_id:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
    else:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    log_file = log_dir / "server.log"

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(log_level)

    formatter = logging.Formatter(format_str)
    file_handler.setFormatter(formatter)

    if include_request_id:
        file_handler.addFilter(RequestContextFilter())

    logging.getLogger().addHandler(file_handler)
    return log_dir
