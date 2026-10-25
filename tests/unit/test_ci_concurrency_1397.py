# SPDX-License-Identifier: Apache-2.0

import pathlib

import yaml

_WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
)


def test_ci_cancels_superseded_runs_per_ref():
    # #1397: without this, a force-push after a rebase leaves the previous
    # run holding macOS slots for hours while the new run waits in the queue.
    workflow = yaml.safe_load(_WORKFLOW.read_text())

    concurrency = workflow.get("concurrency")

    assert concurrency is not None, "ci.yml has no concurrency block"
    assert concurrency["cancel-in-progress"] is True
    assert "github.ref" in concurrency["group"]
