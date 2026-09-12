# SPDX-License-Identifier: Apache-2.0
"""Tests for PR-D11: cluster weighted router + ablations + CI sharding."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass

import pytest


def test_swrr_weight_distribution():
    # nginx smooth weighted round-robin: 5:1:1 over 7 picks = 5A 1B 1C.
    from fusion_mlx.cluster.router import Backend, ClusterRouter

    backs = [
        Backend("A", "http://a:1", 5),
        Backend("B", "http://b:1", 1),
        Backend("C", "http://c:1", 1),
    ]
    r = ClusterRouter(backs)
    picks = [asyncio.run(r.select()).name for _ in range(7)]
    c = Counter(picks)
    assert c["A"] == 5
    assert c["B"] == 1
    assert c["C"] == 1


def test_swrr_smooth_not_clustered():
    # SWRR spreads picks — A should NOT appear 5 times in a row.
    from fusion_mlx.cluster.router import Backend, ClusterRouter

    backs = [Backend("A", "http://a:1", 5), Backend("B", "http://b:1", 1)]
    r = ClusterRouter(backs)
    picks = [asyncio.run(r.select()).name for _ in range(6)]
    # A weight 5 of 6 total — but smooth means no run of 5 A's.
    max_run = 0
    run = 0
    for p in picks:
        run = run + 1 if p == "A" else 0
        max_run = max(max_run, run)
    assert max_run < 5, f"picks clustered: {picks}"


def test_dead_backend_skipped():
    from fusion_mlx.cluster.router import Backend, ClusterRouter

    r = ClusterRouter([Backend("A", "http://a:1", 1), Backend("B", "http://b:1", 1)])
    r.get_backend("A").mark_dead()
    sel = asyncio.run(r.select())
    assert sel is not None
    assert sel.name == "B"


def test_all_dead_returns_none():
    from fusion_mlx.cluster.router import Backend, ClusterRouter

    r = ClusterRouter([Backend("A", "http://a:1", 1)])
    r.get_backend("A").mark_dead()
    # within cooldown → not eligible
    assert asyncio.run(r.select()) is None


def test_dead_cooldown_revive():
    from fusion_mlx.cluster.router import _DEAD_COOLDOWN, Backend, ClusterRouter

    r = ClusterRouter([Backend("A", "http://a:1", 1)])
    b = r.get_backend("A")
    b.mark_dead()
    assert not b.alive
    # simulate cooldown expiry
    b.last_failure_ts -= _DEAD_COOLDOWN + 1
    sel = asyncio.run(r.select())
    assert sel is not None
    assert sel.name == "A"


def test_record_failure_marks_dead_at_threshold():
    from fusion_mlx.cluster.router import _MAX_FAILURES, Backend

    b = Backend("A", "http://a:1", 1)
    for _ in range(_MAX_FAILURES):
        b.record_failure()
    assert not b.alive


def test_record_success_resets():
    from fusion_mlx.cluster.router import Backend

    b = Backend("A", "http://a:1", 1)
    b.record_failure()
    b.record_failure()
    b.record_success()
    assert b.alive
    assert b.failures == 0


def test_weight_must_be_positive():
    from fusion_mlx.cluster.router import Backend

    with pytest.raises(ValueError):
        Backend("A", "http://a:1", 0)
    with pytest.raises(ValueError):
        Backend("A", "http://a:1", -1)


def test_snapshot_frozen():
    from fusion_mlx.cluster.router import Backend, ClusterRouter

    r = ClusterRouter([Backend("A", "http://a:1", 3)])
    snaps = asyncio.run(r.snapshot())
    assert len(snaps) == 1
    s = snaps[0]
    assert s.name == "A"
    assert s.weight == 3
    assert s.alive is True
    # mutating backend does not change the frozen snapshot
    r.get_backend("A").mark_dead()
    assert s.alive is True


def test_add_remove_backend():
    from fusion_mlx.cluster.router import Backend, ClusterRouter

    r = ClusterRouter()
    r.add_backend(Backend("A", "http://a:1", 2))
    r.add_backend(Backend("B", "http://b:1", 1))
    assert len(r.backends()) == 2
    assert r.remove_backend("A") is True
    assert r.remove_backend("nope") is False
    assert len(r.backends()) == 1


def test_build_backends_from_config():
    from fusion_mlx.cluster.router import build_backends_from_config

    @dataclass
    class Cfg:
        cluster_peers: list
        cluster_weights: dict

    cfg = Cfg(
        cluster_peers=["127.0.0.1:11435", "127.0.0.1:11436"],
        cluster_weights={"127.0.0.1:11435": 3, "127.0.0.1:11436": 1},
    )
    built = build_backends_from_config(cfg)
    assert len(built) == 2
    w = {b.name: b.weight for b in built}
    assert w["127.0.0.1:11435"] == 3
    assert w["127.0.0.1:11436"] == 1


def test_build_backends_default_weight_one():
    from fusion_mlx.cluster.router import build_backends_from_config

    @dataclass
    class Cfg:
        cluster_peers: list
        cluster_weights: dict

    cfg = Cfg(cluster_peers=["127.0.0.1:11435"], cluster_weights={})
    built = build_backends_from_config(cfg)
    assert built[0].weight == 1


def test_bootstrap_weighted_empty_weights_inactive():
    from fusion_mlx.cluster.router import bootstrap_weighted, get_router, set_router

    set_router(None)

    @dataclass
    class Cfg:
        cluster_peers: list
        cluster_weights: dict

    cfg = Cfg(cluster_peers=["127.0.0.1:11435"], cluster_weights={})
    n = asyncio.run(bootstrap_weighted(cfg))
    assert n == 0
    assert get_router() is None


def test_bootstrap_weighted_activates():
    from fusion_mlx.cluster.router import bootstrap_weighted, get_router, set_router

    set_router(None)

    @dataclass
    class Cfg:
        cluster_peers: list
        cluster_weights: dict

    cfg = Cfg(
        cluster_peers=["127.0.0.1:11435", "127.0.0.1:11436"],
        cluster_weights={"127.0.0.1:11435": 2, "127.0.0.1:11436": 1},
    )
    n = asyncio.run(bootstrap_weighted(cfg))
    assert n == 2
    assert get_router() is not None
    set_router(None)


def test_backend_node_adapter_parses_url():
    from fusion_mlx.cluster.router import Backend, _BackendNodeAdapter

    b = Backend("x", "http://127.0.0.1:11435", 1)
    a = _BackendNodeAdapter(b)
    assert a.node_id == "x"
    assert a.base_url == "http://127.0.0.1:11435"
    assert a.host == "127.0.0.1"
    assert a.port == 11435


def test_ablation_gated_off_by_default():
    from fusion_mlx.ablations import load_ablation

    assert load_ablation("nonexistent_xyz") is None


def test_ablation_loads_when_env_set(monkeypatch):
    # create a throwaway ablation module via importlib
    import sys
    import types

    from fusion_mlx.ablations import load_ablation

    mod = types.ModuleType("fusion_mlx.ablations._test_abl")
    mod.MARK = "loaded"
    sys.modules["fusion_mlx.ablations._test_abl"] = mod
    monkeypatch.setenv("FUSION_ABLATION__TEST_ABL", "1")
    loaded = load_ablation("_test_abl")
    assert loaded is not None
    assert loaded.MARK == "loaded"
    del sys.modules["fusion_mlx.ablations._test_abl"]


def test_shard_script_known_shards():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "shard_tests", Path("scripts/shard_tests.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(mod.KNOWN_SHARDS) == {"core", "llm", "modal", "infra", "exp", "rest"}


def test_shard_script_full_coverage():
    # every test_*.py file must be claimed by exactly one shard — no
    # test silently dropped from CI.
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "shard_tests", Path("scripts/shard_tests.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    all_files = {p.name for p in mod.TESTS_DIR.glob("test_*.py")}
    sharded: set[str] = set()
    for shard in mod.KNOWN_SHARDS:
        for f in mod.files_for(shard):
            sharded.add(Path(f).name)
    missing = all_files - sharded
    assert (
        not missing
    ), f"{len(missing)} files dropped from shards: {sorted(missing)[:10]}"


def test_shard_for_classification():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "shard_tests", Path("scripts/shard_tests.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._shard_for("test_audio_routes.py") == "modal"
    assert mod._shard_for("test_server.py") == "core"
    assert mod._shard_for("test_batched_engine.py") == "llm"
    assert mod._shard_for("test_model_auto_config.py") == "infra"
    assert mod._shard_for("test_dflash2.py") == "exp"
    assert mod._shard_for("test_unknown_thing.py") == "rest"
