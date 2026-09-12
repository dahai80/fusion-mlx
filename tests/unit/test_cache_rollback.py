# SPDX-License-Identifier: Apache-2.0
"""D2.8/G13: multi-cache atomic rollback tests.

Verifies: snapshot+rollback evicts keys from ALL registered leaves;
concurrent rollback doesn't double-evict or miss leaves; context manager
rolls back on exception, commits on success; re-raise preserves original.
"""

import threading

import pytest

from fusion_mlx.cache.cache_rollback import CacheRollbackManager


class _FakeLeaf:
    def __init__(self, name: str):
        self.name = name
        self.store: set = set()
        self.evict_calls = 0

    def evict(self, key) -> bool:
        self.evict_calls += 1
        if key in self.store:
            self.store.discard(key)
            return True
        return False


def _manager_with_leaves(*leaves):
    mgr = CacheRollbackManager()
    for leaf in leaves:
        mgr.register_leaf(leaf.name, leaf.evict)
    return mgr


def test_rollback_evicts_from_all_leaves():
    hot = _FakeLeaf("hot")
    cold = _FakeLeaf("cold")
    prefix = _FakeLeaf("prefix")
    for leaf in (hot, cold, prefix):
        leaf.store.add("k1")
        leaf.store.add("k2")
    mgr = _manager_with_leaves(hot, cold, prefix)
    evicted = mgr.rollback_keys(["k1", "k2"])
    assert evicted == 6
    for leaf in (hot, cold, prefix):
        assert leaf.store == set()
        assert leaf.evict_calls == 2


def test_rollback_skips_missing_keys():
    hot = _FakeLeaf("hot")
    hot.store.add("present")
    mgr = _manager_with_leaves(hot)
    evicted = mgr.rollback_keys(["present", "absent"])
    assert evicted == 1
    assert hot.store == set()


def test_double_rollback_is_noop():
    hot = _FakeLeaf("hot")
    hot.store.add("k")
    mgr = _manager_with_leaves(hot)
    token = mgr.snapshot(["k"])
    assert mgr.rollback(token) == 1
    assert mgr.rollback(token) == 0


def test_context_manager_rolls_back_on_exception():
    hot = _FakeLeaf("hot")
    cold = _FakeLeaf("cold")
    hot.store.add("k1")
    cold.store.add("k1")
    mgr = _manager_with_leaves(hot, cold)
    with pytest.raises(RuntimeError, match="boom"):
        with mgr.rollback_context(["k1"]):
            raise RuntimeError("boom")
    assert hot.store == set()
    assert cold.store == set()


def test_context_manager_commits_on_success():
    hot = _FakeLeaf("hot")
    hot.store.add("k1")
    mgr = _manager_with_leaves(hot)
    with mgr.rollback_context(["k1"]) as token:
        token.keys.append("k1")
    assert hot.store == {"k1"}
    assert not token.rolled_back


def test_concurrent_rollback_no_tear():
    hot = _FakeLeaf("hot")
    keys = [f"k{i}" for i in range(100)]
    for k in keys:
        hot.store.add(k)
    mgr = _manager_with_leaves(hot)
    barrier = threading.Barrier(4)
    errors = []

    def worker():
        barrier.wait()
        try:
            mgr.rollback_keys(keys)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert hot.store == set()


def test_evict_failure_does_not_propagate():
    def bad_evict(key):
        raise OSError("disk dead")

    mgr = CacheRollbackManager()
    mgr.register_leaf("broken", bad_evict)
    good = _FakeLeaf("good")
    good.store.add("k")
    mgr.register_leaf("good", good.evict)
    evicted = mgr.rollback_keys(["k"])
    assert evicted == 1
    assert good.store == set()


def test_register_leaf_replaces_existing():
    mgr = CacheRollbackManager()
    leaf1 = _FakeLeaf("hot")
    leaf2 = _FakeLeaf("hot")
    mgr.register_leaf("hot", leaf1.evict)
    mgr.register_leaf("hot", leaf2.evict)
    assert len(mgr.leaves()) == 1
    leaf2.store.add("k")
    mgr.rollback_keys(["k"])
    assert leaf1.evict_calls == 0
    assert leaf2.evict_calls == 1
