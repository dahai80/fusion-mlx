# SPDX-License-Identifier: Apache-2.0
"""Unit tests for /distributed/decode_step + /distributed/reset_cache (#630).

Fast validation/route tests — no model load. Real-model bit-exact coverage
lives in test_distributed_decode_step_e2e.py (this file's sibling convention
follows test_distributed_pipeline.py, but e2e is split out for clarity)."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")


def test_shard_info_exposes_kv_offset_zero_on_fresh_shard():
    """A freshly loaded shard has kv_cache=None → list_shards reports
    kv_offset=0."""
    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    # Use a dummy shard entry to avoid a real model load in this unit test.
    mgr._shards["shard-fresh"] = {
        "shard_id": "shard-fresh",
        "model_id": "dummy",
        "shard_index": 0,
        "layer_range": [0, 4],
        "dtype": None,
        "num_layers": 16,
        "kv_cache": None,
    }
    from fusion_mlx.api.distributed_routes import ShardInfo

    info = ShardInfo(**mgr.list_shards()[0])
    assert info.kv_offset == 0
