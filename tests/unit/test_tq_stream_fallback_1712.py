import logging

import pytest

from fusion_mlx import turboquant_kv as tq


class _StubCache:
    def __init__(self, bits=None, seed=None):
        self.bits = bits
        self.seed = seed
        self.calls = 0

    def update_and_fetch(self, keys, values):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("There is no Stream(gpu, 0) in current thread.")
        self.last_keys = keys
        self.last_values = values


class _FailingCache(_StubCache):
    def update_and_fetch(self, keys, values):
        self.calls += 1
        raise RuntimeError("boom")


class _FakeSource:
    def keys_and_values(self):
        return ("KEYS", "VALUES")


@pytest.fixture
def stub_cls(monkeypatch):
    monkeypatch.setattr(tq, "TurboQuantKVCache", _StubCache)
    monkeypatch.setattr(tq, "K8V4TurboQuantKVCache", _StubCache)
    return _StubCache


def test_source_quantize_retries_on_default_stream(stub_cls, caplog):
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.turboquant_kv"):
        cache = tq.make_turboquant_kv_cache(4.0, mode="k8v4", source=_FakeSource())
    assert cache.calls == 2
    assert cache.last_keys == "KEYS"
    assert "#1712" in caplog.text
    assert "default GPU stream" in caplog.text


def test_non_stream_runtime_error_propagates(monkeypatch):
    monkeypatch.setattr(tq, "TurboQuantKVCache", _FailingCache)
    monkeypatch.setattr(tq, "K8V4TurboQuantKVCache", _FailingCache)
    with pytest.raises(RuntimeError, match="boom"):
        tq.make_turboquant_kv_cache(4.0, mode="k8v4", source=_FakeSource())


def test_source_none_skips_quantize_in(stub_cls):
    cache = tq.make_turboquant_kv_cache(4.0, mode="v4")
    assert cache.calls == 0
    assert cache.bits == 4.0
