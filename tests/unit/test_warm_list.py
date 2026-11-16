"""WarmList store behavior: record/evict/flush, kill switch, fail-safe.

Covers the Lever-3 warm-list contract: bounded LRU by last_seen, head
text kept local under caps, FUSION_WARMLIST_DISABLE=1 stops all
recording, and no exception ever escapes record().
"""

import json

import pytest

from fusion_mlx.cache.warm_list import WarmList


@pytest.fixture(autouse=True)
def _no_min_tokens(monkeypatch):
    # Default suite exercises store mechanics with tiny prompts; the
    # size filter itself is pinned by the dedicated tests below.
    monkeypatch.setenv("FUSION_WARMLIST_MIN_TOKENS", "0")


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "warm_list.json")


def test_record_and_reload(path):
    wl = WarmList(path)
    wl.record("aaaa", "system prompt head", 128)
    wl.flush()
    wl2 = WarmList(path)
    assert wl2.head_for("aaaa") == "system prompt head"
    assert wl2.top(1)[0]["head_hash"] == "aaaa"
    assert wl2.top(1)[0]["hits"] == 1


def test_hits_increment_and_head_truncated(path):
    wl = WarmList(path)
    wl.record("bbbb", "x" * 2000, 512)
    wl.record("bbbb", "x" * 2000, 512)
    assert wl.head_for("bbbb") == "x" * 1024
    assert wl.top(1)[0]["hits"] == 2


def test_lru_eviction_by_entries(path, monkeypatch):
    monkeypatch.setenv("FUSION_WARMLIST_MAX_ENTRIES", "2")
    wl = WarmList(path)
    wl.record("k1", "a", 1)
    wl.record("k2", "b", 1)
    wl.record("k3", "c", 1)
    wl.flush()
    assert wl.head_for("k1") is None
    assert wl.head_for("k3") == "c"


def test_kill_switch(path, monkeypatch):
    monkeypatch.setenv("FUSION_WARMLIST_DISABLE", "1")
    wl = WarmList(path)
    wl.record("kkkk", "secret", 1)
    wl.flush()
    assert wl.head_for("kkkk") is None
    assert wl.stats()["disabled"] is True


def test_non_string_prompt_ignored(path):
    wl = WarmList(path)
    wl.record("cccc", [1, 2, 3], 3)
    wl.record("dddd", None, 0)
    wl.record("eeee", "", 0)
    assert wl.stats()["entries"] == 0


def test_corrupt_file_starts_empty(path):
    with open(path, "w") as f:
        f.write("{not json")
    wl = WarmList(path)
    wl.record("ffff", "ok", 1)
    assert wl.head_for("ffff") == "ok"


def test_flush_writes_valid_json(path):
    wl = WarmList(path)
    wl.record("gggg", "hello", 5)
    wl.flush()
    with open(path) as f:
        data = json.load(f)
    assert data["gggg"]["head"] == "hello"


def test_size_filter_skips_short_prompts(path, monkeypatch):
    monkeypatch.setenv("FUSION_WARMLIST_MIN_TOKENS", "256")
    wl = WarmList(path)
    wl.record("ssss", "warmup:", 4)
    wl.record("tttt", "below threshold", 255)
    assert wl.head_for("ssss") is None
    assert wl.head_for("tttt") is None
    assert wl.stats()["entries"] == 0


def test_size_filter_none_tokens_skipped(path, monkeypatch):
    monkeypatch.setenv("FUSION_WARMLIST_MIN_TOKENS", "256")
    wl = WarmList(path)
    wl.record("nnnn", "unknown size", None)
    assert wl.head_for("nnnn") is None


def test_default_min_tokens_is_256(monkeypatch):
    monkeypatch.delenv("FUSION_WARMLIST_MIN_TOKENS", raising=False)
    from fusion_mlx.cache.warm_list import warm_min_tokens

    assert warm_min_tokens() == 256


def test_size_filter_allows_at_threshold(path, monkeypatch):
    monkeypatch.setenv("FUSION_WARMLIST_MIN_TOKENS", "256")
    wl = WarmList(path)
    wl.record("uuuu", "long enough head", 256)
    assert wl.head_for("uuuu") == "long enough head"


def test_top_heads_skips_legacy_short_entries(path, monkeypatch):
    # Entry written before the size filter existed: stored on disk with
    # a tiny prompt_tokens. Replay path must skip it without deleting
    # the store; fresh long entries still surface.
    with open(path, "w") as f:
        json.dump(
            {
                "old1": {
                    "head": "tiny warmup prompt",
                    "prompt_tokens": 6,
                    "last_seen": 2.0,
                    "hits": 9,
                },
                "new1": {
                    "head": "a genuine system prompt head",
                    "prompt_tokens": 2048,
                    "last_seen": 1.0,
                    "hits": 1,
                },
            },
            f,
        )
    wl = WarmList(path)
    monkeypatch.setenv("FUSION_WARMLIST_MIN_TOKENS", "256")
    assert wl.top_heads(3) == [("new1", "a genuine system prompt head")]
