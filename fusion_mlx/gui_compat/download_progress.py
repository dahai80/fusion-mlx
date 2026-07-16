"""
Real download progress tracking for MLX-GUI.
This module provides actual progress tracking for HuggingFace downloads.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class DownloadProgress:
    """Represents the current download progress."""
    status: str = "not_started"
    progress: float = 0.0
    downloaded_mb: float = 0.0
    total_mb: float = 0.0
    speed_mbps: float = 0.0
    eta_seconds: float = 0.0
    stage: str = "Starting download..."
    error: Optional[str] = None


class DownloadProgressTracker:
    """Tracks real download progress from HuggingFace downloads."""

    def __init__(self):
        self._downloads: Dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._running = False

    def start_tracking(self, model_id: str) -> None:
        """Start tracking download progress for a model."""
        with self._lock:
            self._downloads[model_id] = DownloadProgress(
                status="downloading",
                progress=0.0,
                stage="Starting download..."
            )

        # Start cleanup thread if not running
        if not self._running:
            self._start_cleanup_thread()

    def update_progress(self, model_id: str, **kwargs) -> None:
        """Update progress for a model."""
        with self._lock:
            if model_id in self._downloads:
                for key, value in kwargs.items():
                    if hasattr(self._downloads[model_id], key):
                        setattr(self._downloads[model_id], key, value)

    def get_progress(self, model_id: str) -> Dict[str, Any]:
        """Get current download progress for a model."""
        with self._lock:
            if model_id in self._downloads:
                progress = self._downloads[model_id]
                return {
                    "status": progress.status,
                    "progress": progress.progress,
                    "downloaded_mb": progress.downloaded_mb,
                    "total_mb": progress.total_mb,
                    "speed_mbps": progress.speed_mbps,
                    "eta_seconds": progress.eta_seconds,
                    "stage": progress.stage
                }
            else:
                return {
                    "status": "not_downloading",
                    "progress": 0,
                    "downloaded_mb": 0,
                    "total_mb": 0,
                    "speed_mbps": 0,
                    "eta_seconds": 0,
                    "stage": "No download in progress"
                }

    def complete_download(self, model_id: str) -> None:
        """Mark download as complete."""
        with self._lock:
            if model_id in self._downloads:
                self._downloads[model_id].status = "completed"
                self._downloads[model_id].progress = 100.0
                self._downloads[model_id].stage = "Download complete"

    def fail_download(self, model_id: str, error: str) -> None:
        """Mark download as failed."""
        with self._lock:
            if model_id in self._downloads:
                self._downloads[model_id].status = "failed"
                self._downloads[model_id].error = error
                self._downloads[model_id].stage = f"Download failed: {error}"

    def parse_huggingface_progress(self, log_line: str, model_id: str) -> bool:
        """Parse HuggingFace download progress from log line."""
        # Pattern: "  136M/5.31G [00:59<26:37, 3.24MB/s]"
        pattern = r'(\d+(?:\.\d+)?[KMGT]?B?)/(\d+(?:\.\d+)?[KMGT]?B?) \[.*?, (\d+(?:\.\d+)?[KMGT]?B?/s)\]'
        match = re.search(pattern, log_line)

        if match:
            downloaded_str = match.group(1)
            total_str = match.group(2)
            speed_str = match.group(3)

            # Convert to MB
            downloaded_mb = self._convert_to_mb(downloaded_str)
            total_mb = self._convert_to_mb(total_str)
            speed_mbps = self._convert_to_mb(speed_str)

            # Calculate progress
            progress = (downloaded_mb / total_mb) * 100 if total_mb > 0 else 0

            # Calculate ETA
            if speed_mbps > 0:
                remaining_mb = total_mb - downloaded_mb
                eta_seconds = remaining_mb / speed_mbps
            else:
                eta_seconds = 0

            self.update_progress(
                model_id,
                progress=progress,
                downloaded_mb=downloaded_mb,
                total_mb=total_mb,
                speed_mbps=speed_mbps,
                eta_seconds=eta_seconds,
                stage=f"Downloading... {downloaded_str}/{total_str}"
            )
            return True

        return False

    def _convert_to_mb(self, size_str: str) -> float:
        """Convert size string to MB."""
        size_str = size_str.upper()

        # Remove 'B' suffix if present
        if size_str.endswith('B'):
            size_str = size_str[:-1]

        # Extract number and unit
        match = re.match(r'(\d+(?:\.\d+)?)\s*([KMGT]?)', size_str)
        if not match:
            return 0.0

        number = float(match.group(1))
        unit = match.group(2)

        # Convert to MB
        multipliers = {
            '': 1/1024/1024,  # Assume bytes if no unit
            'K': 1/1024,      # KB to MB
            'M': 1,           # Already MB
            'G': 1024,        # GB to MB
            'T': 1024*1024    # TB to MB
        }

        return number * multipliers.get(unit, 1)

    def _start_cleanup_thread(self) -> None:
        """Start background thread to clean up old downloads."""
        if self._running:
            return

        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_old_downloads,
            daemon=True
        )
        self._cleanup_thread.start()

    def _cleanup_old_downloads(self) -> None:
        """Clean up downloads that are older than 1 hour."""
        while self._running:
            try:
                time.sleep(300)  # Check every 5 minutes

                with self._lock:
                    current_time = time.time()
                    to_remove = []

                    for model_id, progress in self._downloads.items():
                        # Remove completed/failed downloads after 1 hour
                        if progress.status in ["completed", "failed"]:
                            to_remove.append(model_id)

                    for model_id in to_remove:
                        del self._downloads[model_id]

                    if not self._downloads:
                        self._running = False
                        break

            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")

    def stop_tracking(self, model_id: str) -> None:
        """Stop tracking download for a model."""
        with self._lock:
            if model_id in self._downloads:
                del self._downloads[model_id]

    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get progress for all tracked downloads."""
        with self._lock:
            return {
                model_id: self.get_progress(model_id)
                for model_id in self._downloads.keys()
            }


# Global tracker instance
_tracker = None


def get_progress_tracker() -> DownloadProgressTracker:
    """Get the global progress tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = DownloadProgressTracker()
    return _tracker


# Hook into logging to capture HuggingFace download progress
class HuggingFaceDownloadHandler(logging.Handler):
    """Custom logging handler to capture HuggingFace download progress."""

    def __init__(self):
        super().__init__()
        self.tracker = get_progress_tracker()

    def emit(self, record):
        """Process log records for download progress."""
        if record.name.startswith('huggingface_hub'):
            message = str(record.getMessage())

            # Extract model ID from log context if available
            model_id = getattr(record, 'model_id', None)
            if not model_id:
                # Try to extract from message
                match = re.search(r'Repository\s+([^\s]+)', message)
                if match:
                    model_id = match.group(1)

            if model_id:
                self.tracker.parse_huggingface_progress(message, model_id)


# Install the handler
def install_progress_handler():
    """Install the HuggingFace download progress handler."""
    handler = HuggingFaceDownloadHandler()
    handler.setLevel(logging.INFO)

    # Add to huggingface_hub logger
    hf_logger = logging.getLogger('huggingface_hub')
    hf_logger.addHandler(handler)

    # Also add to general loggers that might contain download info
    for logger_name in ['huggingface_hub.file_download', 'huggingface_hub.snapshot_download']:
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)


# Initialize the handler
install_progress_handler()
