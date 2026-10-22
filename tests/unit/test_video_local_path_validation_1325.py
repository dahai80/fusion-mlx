# SPDX-License-Identifier: Apache-2.0
# #1325: local media paths are existence-checked at request validation time —
# a missing file must 400 before admission/engine load, not surface as a raw
# FileNotFoundError after the engine and its media reservation are committed.

import asyncio

import pytest
from fastapi import HTTPException

from fusion_mlx.api import _url_safety as _us
from fusion_mlx.api.videos_routes import (
    _ensure_local_file_exists,
    _resolve_image_to_path,
    _validate_path_param,
)


@pytest.fixture
def allowed_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(_us, "_ALLOWED_READ_DIRS", [str(tmp_path)])
    return tmp_path


def test_missing_file_raises_400(allowed_dir):
    missing = allowed_dir / "nope.jpg"
    with pytest.raises(HTTPException) as exc:
        _ensure_local_file_exists(str(missing), "Image path")
    assert exc.value.status_code == 400
    assert "file not found" in exc.value.detail
    assert str(missing) in exc.value.detail


def test_existing_file_passes(allowed_dir):
    target = allowed_dir / "frame.jpg"
    target.write_bytes(b"jpeg")
    _ensure_local_file_exists(str(target), "Image path")


def test_resolve_image_missing_file_400(allowed_dir):
    missing = allowed_dir / "verify-frame.jpg"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_resolve_image_to_path(str(missing)))
    assert exc.value.status_code == 400
    assert "Image path file not found" in exc.value.detail


def test_validate_path_param_missing_file_400(allowed_dir):
    missing = allowed_dir / "control.mp4"
    with pytest.raises(HTTPException) as exc:
        _validate_path_param(str(missing), "Control video")
    assert exc.value.status_code == 400
    assert "Control video path file not found" in exc.value.detail


def test_validate_path_param_existing_ok(allowed_dir):
    target = allowed_dir / "control.mp4"
    target.write_bytes(b"mp4")
    assert _validate_path_param(str(target), "Control video") == str(target)
