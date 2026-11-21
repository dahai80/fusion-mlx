import pytest
from mlx_lm.generate import BatchGenerator

from fusion_mlx.scheduler import monkeypatches as mp

# Importing the monkeypatches module installs the close() patch.
assert getattr(
    BatchGenerator.close, "_fusion_patched", False
), "BatchGenerator.close patch not installed on import"


class _Gen:
    def __init__(self, wired_limit=4):
        self._old_wired_limit = wired_limit


def test_close_swallows_foreign_thread_stream_error(monkeypatch):
    # CI failure mode (#1678): close() -> mx.synchronize raises on a
    # thread other than the stream's creator. Must not propagate, and
    # must still restore the wired limit.
    gen = _Gen(wired_limit=8)

    def raising_close(self):
        raise RuntimeError("There is no Stream(gpu, 70) in current thread.")

    monkeypatch.setattr(mp, "_original_bg_close", raising_close)
    mp._patched_bg_close(gen)
    assert gen._old_wired_limit is None


def test_close_swallows_variant_message(monkeypatch):
    gen = _Gen(wired_limit=2)

    def raising_close(self):
        raise RuntimeError("Stream(gpu, 3) not exist")

    monkeypatch.setattr(mp, "_original_bg_close", raising_close)
    mp._patched_bg_close(gen)
    assert gen._old_wired_limit is None


def test_close_noop_without_wired_limit(monkeypatch):
    gen = _Gen(wired_limit=None)

    def raising_close(self):
        raise RuntimeError("There is no Stream(gpu, 64) in current thread.")

    monkeypatch.setattr(mp, "_original_bg_close", raising_close)
    mp._patched_bg_close(gen)
    assert gen._old_wired_limit is None


def test_close_reraises_unrelated_runtime_error(monkeypatch):
    def boom(self):
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(mp, "_original_bg_close", boom)
    with pytest.raises(RuntimeError, match="unrelated"):
        mp._patched_bg_close(_Gen())


def test_close_passthrough_success(monkeypatch):
    gen = _Gen(wired_limit=6)

    def good_close(self):
        self._old_wired_limit = None
        self.closed = True

    monkeypatch.setattr(mp, "_original_bg_close", good_close)
    mp._patched_bg_close(gen)
    assert getattr(gen, "closed", False) is True
