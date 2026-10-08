# SPDX-License-Identifier: Apache-2.0
"""#1051: HiddenStateCapture must be uninstalled on engine deep_reset,
and Eagle3Speculator.reset() must clear captured/prefill tensors so
abort/timeout doesn't strand hidden_size x seq_len Metal arrays."""

from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.scheduler import sched_step
from fusion_mlx.scheduler.spec_decode import SpecDecodeState
from fusion_mlx.speculative.eagle3 import speculator as e3
from fusion_mlx.speculative.eagle3.speculator import Eagle3Speculator
from fusion_mlx.speculative.hidden_capture import HiddenStateCapture


class _StubLayer(nn.Module):
    def __call__(self, *args, **kwargs):
        return mx.zeros((1, 1, 4))

    def some_attr(self):
        return "original"


class _StubInnerModel:
    def __init__(self, n_layers=5):
        self.layers = [_StubLayer() for _ in range(n_layers)]


class _StubModel:
    def __init__(self, n_layers=5):
        self.model = _StubInnerModel(n_layers)


class _StubDraftModel:
    def forward_standalone(self, input_ids, cache=None, hidden_state=None):
        return mx.zeros((1, 1, 32000))


def _make_capture(layer_ids=(0, 2, 4)):
    model = _StubModel()
    cap = HiddenStateCapture(model, layer_ids=list(layer_ids))
    cap.install()
    return cap, model


def _populate_capture(cap, seq_len=128):
    cap._captured[0] = mx.zeros((1, seq_len, 4096))
    cap._captured[2] = mx.zeros((1, seq_len, 4096))
    cap._prefill_captured[0] = mx.zeros((1, seq_len, 4096))
    cap._prefill_captured[2] = mx.zeros((1, seq_len, 4096))


def test_reset_clears_hidden_capture_tensors():
    """Eagle3Speculator.reset() must call clear_captured +
    clear_prefill_captured on its _hidden_capture. Before #1051, reset
    only cleared _draft_cache/_prefill_hidden, leaving captured Metal
    tensors stranded until the next successful prefill build."""
    cap, _ = _make_capture()
    _populate_capture(cap, seq_len=256)
    assert len(cap._captured) == 2
    assert len(cap._prefill_captured) == 2

    cfg = e3.Eagle3DraftConfig(num_draft=3, temperature=0.0)
    spec = Eagle3Speculator(config=cfg)
    spec.model = _StubDraftModel()
    spec._loaded = True
    spec._draft_cache = type("C", (), {"offset": 0})()
    spec.set_hidden_capture(cap)
    spec._prefill_hidden = mx.zeros((1, 1, 4096))

    spec.reset()

    assert len(cap._captured) == 0
    assert len(cap._prefill_captured) == 0
    assert spec._prefill_hidden is None
    assert cap.installed is True


def test_reset_with_no_hidden_capture_is_safe():
    """reset() must not crash when _hidden_capture is None (non-eagle3
    or pre-install path)."""
    cfg = e3.Eagle3DraftConfig(num_draft=3, temperature=0.0)
    spec = Eagle3Speculator(config=cfg)
    spec.model = _StubDraftModel()
    spec._loaded = True
    spec._draft_cache = type("C", (), {"offset": 0})()
    spec._prefill_hidden = mx.zeros((1, 1, 4096))
    spec.reset()
    assert spec._prefill_hidden is None


def test_deep_reset_uninstalls_hidden_capture():
    """scheduler.deep_reset() must call hidden_capture.uninstall() BEFORE
    dropping _spec_decode_state. Before #1051, setting _spec_decode_state
    =None orphaned the capture (wrappers stayed in model.model.layers,
    captured tensors leaked on Metal) with no reachable uninstall path."""
    cap, model = _make_capture()
    _populate_capture(cap, seq_len=512)
    assert cap.installed is True
    inner = model.model
    assert isinstance(inner.layers[0], type(cap._original_layers[0])) is False

    spec_state = SpecDecodeState(draft_model_decoder=MagicMock(), hidden_capture=cap)

    sched = MagicMock()
    sched._spec_decode_state = spec_state
    sched.model = model
    sched.tokenizer = None
    sched.paged_cache_manager = None
    sched.block_aware_cache = None
    sched.memory_monitor = None
    sched._boundary_snapshot_store = None
    sched._specprefill_draft_model = None
    sched._vlm_mtp_drafter = None
    sched._dflash_runtime = None
    sched._draft_prefix_cache = None
    sched.reset = MagicMock()

    sched_step.deep_reset(sched)

    assert cap.installed is False
    assert len(cap._original_layers) == 0
    assert len(cap._captured) == 0
    assert len(cap._prefill_captured) == 0
    assert isinstance(inner.layers[0], _StubLayer)
    assert isinstance(inner.layers[2], _StubLayer)
    assert isinstance(inner.layers[4], _StubLayer)
    assert sched._spec_decode_state is None


def test_deep_reset_no_spec_state_is_safe():
    """deep_reset() must not crash when _spec_decode_state is None
    (non-spec-decode engine)."""
    model = _StubModel()
    sched = MagicMock()
    sched._spec_decode_state = None
    sched.model = model
    sched.tokenizer = None
    sched.paged_cache_manager = None
    sched.block_aware_cache = None
    sched.memory_monitor = None
    sched._boundary_snapshot_store = None
    sched._specprefill_draft_model = None
    sched._vlm_mtp_drafter = None
    sched._dflash_runtime = None
    sched._draft_prefix_cache = None
    sched.reset = MagicMock()

    sched_step.deep_reset(sched)
    assert sched._spec_decode_state is None
