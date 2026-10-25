# SPDX-License-Identifier: Apache-2.0
"""Regression tests for audit 1019 L3: the dead sync batch path is gone.

generate_batch_sync drove Scheduler.step() directly, bypassing the executor,
streaming, abort-on-disconnect and the error handlers, and had zero callers.
Batches must go through generate_batch_async.
"""

import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[2]
_ENGINE = _REPO / "fusion_mlx/engine_core.py"


def test_generate_batch_sync_removed():
    source = _ENGINE.read_text()
    assert "def generate_batch_sync" not in source


def test_no_callers_anywhere():
    offenders = []
    for path in list((_REPO / "fusion_mlx").rglob("*.py")) + list(
        (_REPO / "tests").rglob("*.py")
    ):
        if "__pycache__" in path.parts or path.name == pathlib.Path(__file__).name:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "generate_batch_sync" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{path.relative_to(_REPO)}:{lineno}: {line.strip()}")
    assert offenders == []


def test_async_batch_path_still_present():
    from fusion_mlx.engine_core import EngineCore

    assert hasattr(EngineCore, "generate_batch_async")
