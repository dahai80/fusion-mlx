# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for Fusion-MLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import logging
from collections import deque
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from .auth import (
    require_admin,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "https://bench.dpdns.org/assets/fusionmlx_preset.json"


from .helpers import (
    _get_rich_global_settings,
)

_router = APIRouter()

# =============================================================================
# Logs API Routes
# =============================================================================


def _tail_file(file_path: Path, num_lines: int) -> tuple[str, int]:
    """
    Read the last N lines of a file efficiently.

    Uses a deque to efficiently keep only the last N lines in memory.

    Args:
        file_path: Path to the log file.
        num_lines: Number of lines to return.

    Returns:
        Tuple of (content_string, total_line_count)
    """
    if not file_path.exists():
        return "", 0

    # Use deque for efficient tail operation
    lines = deque(maxlen=num_lines)
    total_lines = 0

    with open(file_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line)
            total_lines += 1

    return "".join(lines), total_lines


def _get_available_log_files(log_dir: Path) -> list[str]:
    """
    Get list of available log files sorted by modification time.

    Args:
        log_dir: Directory containing log files.

    Returns:
        List of log file names, newest first.
    """
    if not log_dir.exists():
        return []

    files = []
    for f in log_dir.iterdir():
        if not f.is_file() or f.name.startswith("."):
            continue
        # Match server.log, server.log.YYYY-MM-DD, fusion-mlx-YYYYMMDD.log
        if f.suffix == ".log" or ".log." in f.name:
            files.append(f.name)

    # Sort by modification time (newest first)
    files.sort(key=lambda x: (log_dir / x).stat().st_mtime, reverse=True)
    return files


@_router.get("/api/logs")
async def get_logs(
    lines: int = 100,
    file: str | None = None,
    is_admin: bool = Depends(require_admin),
):
    """
    Get server logs.

    Returns the last N lines of the specified log file (or current log).
    Supports viewing historical rotated log files.

    Args:
        lines: Number of lines to return (default: 100, max: 10000).
        file: Optional specific log file name. If not specified, uses current log.

    Returns:
        JSON response with log content and metadata:
        - logs: The log content string
        - total_lines: Total number of lines in the file
        - log_file: Name of the log file being read
        - available_files: List of available log files

    Raises:
        HTTPException: 401 if not authenticated, 503 if server not initialized,
                        400 if invalid file name, 404 if log file not found.
    """
    global_settings = _get_rich_global_settings()

    if global_settings is not None:
        log_dir = global_settings.logging.get_log_dir(global_settings.base_path)
    else:
        log_dir = Path.home() / ".fusion-mlx" / "logs"

    # Fallback: macOS app writes logs under ~/Library/Application Support/FusionMLX/logs/
    app_support_dir = (
        Path.home() / "Library" / "Application Support" / "FusionMLX" / "logs"
    )
    if not log_dir.exists() and app_support_dir.exists():
        log_dir = app_support_dir

    # Limit lines to prevent memory issues
    lines = min(max(1, lines), 10000)

    # Get available log files
    available_files = _get_available_log_files(log_dir)

    # Determine which file to read
    if file:
        # Validate file name (prevent path traversal)
        if "/" in file or "\\" in file or ".." in file:
            raise HTTPException(status_code=400, detail="Invalid file name")
        log_file = log_dir / file
        if not log_file.exists():
            raise HTTPException(status_code=404, detail=f"Log file not found: {file}")
    else:
        # Default: server.log if present, otherwise newest available
        log_file = log_dir / "server.log"
        if not log_file.exists() and available_files:
            log_file = log_dir / available_files[0]

    # Read log content
    if log_file.exists():
        content, total_lines = _tail_file(log_file, lines)
    else:
        content = ""
        total_lines = 0

    return {
        "logs": content,
        "total_lines": total_lines,
        "log_file": log_file.name,
        "available_files": available_files,
    }


router = _router


# =============================================================================
# Log Stream (SSE) — #1220 union-web monitor module
# =============================================================================

_STREAM_POLL_SECONDS = 1.0
_STREAM_HEARTBEAT_SECONDS = 15.0


def _resolve_current_log_file() -> Path | None:
    """Same file resolution as GET /api/logs (no request params)."""
    global_settings = _get_rich_global_settings()

    if global_settings is not None:
        log_dir = global_settings.logging.get_log_dir(global_settings.base_path)
    else:
        log_dir = Path.home() / ".fusion-mlx" / "logs"

    app_support_dir = Path.home() / "Library/Application Support/FusionMLX/logs"
    if not log_dir.exists() and app_support_dir.exists():
        log_dir = app_support_dir

    log_file = log_dir / "server.log"
    if not log_file.exists():
        available = _get_available_log_files(log_dir)
        if available:
            log_file = log_dir / available[0]
        else:
            return None
    return log_file


@_router.get("/api/logs/stream")
async def stream_logs(request: Request, is_admin: bool = Depends(require_admin)):
    """Server-Sent Events tail of the current log file (#1220).

    Emits one ``data: {"line": "..."}\\n\\n`` event per new log line,
    starting from the current end of file (fetch /api/logs for history),
    plus ``: ping`` keepalive comments every 15s. Daily rotation (file
    replaced/shrunk) is detected and the tail restarts from the new file
    head. Ends when the client disconnects.
    """
    import asyncio
    import json as _json

    from fastapi.responses import StreamingResponse

    log_file = _resolve_current_log_file()
    if log_file is None or not log_file.exists():
        raise HTTPException(status_code=404, detail="No log file available")

    async def event_stream():
        offset = log_file.stat().st_size
        tracked = log_file
        heartbeat = 0.0
        logger.info("log stream: client attached file=%s offset=%d", tracked, offset)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    size = tracked.stat().st_size
                except FileNotFoundError:
                    # Rotated away entirely — fall back to dir resolution.
                    tracked = _resolve_current_log_file() or tracked
                    size = tracked.stat().st_size if tracked.exists() else 0
                    offset = 0
                if size < offset:
                    # Truncated/rotated: restart from head of the new file.
                    logger.info(
                        "log stream: rotation detected file=%s old_offset=%d size=%d",
                        tracked,
                        offset,
                        size,
                    )
                    offset = 0
                if size > offset:
                    with open(tracked, encoding="utf-8", errors="replace") as f:
                        f.seek(offset)
                        chunk = f.read()
                        offset = f.tell()
                    for line in chunk.splitlines():
                        if line.strip():
                            payload = _json.dumps({"line": line}, ensure_ascii=False)
                            yield f"data: {payload}\n\n"
                    heartbeat = 0.0
                else:
                    await asyncio.sleep(_STREAM_POLL_SECONDS)
                    heartbeat += _STREAM_POLL_SECONDS
                    if heartbeat >= _STREAM_HEARTBEAT_SECONDS:
                        heartbeat = 0.0
                        yield ": ping\n\n"
        finally:
            logger.info("log stream: client detached file=%s", tracked)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
