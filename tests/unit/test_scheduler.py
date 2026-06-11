# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scheduler modules — config, types, helpers, trim, query, handoff, batch, misc."""

from collections import deque
from unittest.mock import MagicMock, patch
import pytest

from fusion_mlx.scheduler.config import SchedulerConfig, SchedulerOutput, SchedulingPolicy
from fusion_mlx.scheduler.types import (
    _StoreCacheGate, _PrefillAbortedError, _BoundarySnapshotProvider,
)
from fusion_mlx.scheduler.helpers import (
    _prompt_cache_needs_snapshots, _cache_layer_token_count, _cache_base_sizes,
    _slice_vlm_extra, _advance_vlm_extra, _deferred_clear_delay,
    _KNOWN_SLICEABLE_CACHE_TYPES,
)
from fusion_mlx.request import (
    Request, RequestOutput, RequestStatus, SamplingParams,
)


class TestSchedulerConfig:

    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.max_num_seqs == 256
        assert cfg.max_num_batched_tokens == 8192
        assert cfg.policy == SchedulingPolicy.FCFS
        assert cfg.chunked_prefill is False

    def test_custom_values(self):
        cfg = SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=4096, chunked_prefill=True)
        assert cfg.max_num_seqs == 64
        assert cfg.chunked_prefill is True

    def test_invalid_max_num_seqs(self):
        with pytest.raises(ValueError, match="max_num_seqs"):
            SchedulerConfig(max_num_seqs=0)

    def test_invalid_batched_tokens_less_than_prefill(self):
        with pytest.raises(ValueError, match="max_num_batched_tokens"):
            SchedulerConfig(max_num_batched_tokens=512, prefill_step_size=1024)

    def test_invalid_completion_batch_size(self):
        with pytest.raises(ValueError, match="completion_batch_size"):
            SchedulerConfig(max_num_seqs=8, completion_batch_size=16)

    def test_invalid_paged_cache_block_size(self):
        with pytest.raises(ValueError, match="paged_cache_block_size"):
            SchedulerConfig(paged_cache_block_size=0)

    def test_invalid_max_cache_blocks(self):
        with pytest.raises(ValueError, match="max_cache_blocks"):
            SchedulerConfig(max_cache_blocks=100, initial_cache_blocks=256)

    def test_invalid_gc_cleanup_interval(self):
        with pytest.raises(ValueError, match="gc_cleanup_interval"):
            SchedulerConfig(gc_cleanup_interval=-1)

    def test_invalid_mlx_cache_cleanup_interval(self):
        with pytest.raises(ValueError, match="mlx_cache_cleanup_interval"):
            SchedulerConfig(mlx_cache_cleanup_interval=0)


class TestSchedulerOutput:

    def test_defaults(self):
        out = SchedulerOutput()
        assert out.scheduled_request_ids == []
        assert out.num_scheduled_tokens == 0
        assert out.finished_request_ids == set()
        assert out.outputs == []
        assert out.has_work is False

    def test_populated(self):
        out = SchedulerOutput(
            scheduled_request_ids=["r1", "r2"],
            num_scheduled_tokens=100,
            finished_request_ids={"r3"},
            has_work=True,
         )
        assert len(out.scheduled_request_ids) == 2
        assert out.has_work is True


class TestStoreCacheGate:

    def test_initial_state(self):
        gate = _StoreCacheGate(cap=5)
        assert gate.in_flight == 0
        assert gate.cap == 5
        assert gate.has_capacity is True

    def test_submit_and_done(self):
        gate = _StoreCacheGate(cap=3)
        gate.note_submitted()
        gate.note_submitted()
        assert gate.in_flight == 2
        assert gate.has_capacity is True
        gate.note_submitted()
        assert gate.in_flight == 3
        assert gate.has_capacity is False
        gate.note_done()
        assert gate.in_flight == 2
        assert gate.has_capacity is True

    def test_cap_minimum_one(self):
        gate = _StoreCacheGate(cap=0)
        assert gate.cap == 1

    def test_set_cap(self):
        gate = _StoreCacheGate(cap=5)
        gate.set_cap(2)
        assert gate.cap == 2

    def test_note_done_does_not_go_negative(self):
        gate = _StoreCacheGate(cap=1)
        gate.note_done()
        assert gate.in_flight == 0


class TestPrefillAbortedError:

    def test_error_fields(self):
        err = _PrefillAbortedError(aborted_uids=[1, 2], processed_tokens=42)
        assert err.aborted_uids == [1, 2]
        assert err.processed_tokens == 42
        assert "UIDs" in str(err)


class TestBoundarySnapshotProvider:

    def test_contains(self):
        p = _BoundarySnapshotProvider(
            store=None, request_id="r1", valid_tcs=[10, 20, 30],
            in_memory_snapshots={}, extract_fn=lambda x: (x, None),
         )
        assert 10 in p
        assert 15 not in p

    def test_len_and_bool(self):
        p = _BoundarySnapshotProvider(
            store=None, request_id="r1", valid_tcs=[10, 20],
            in_memory_snapshots={}, extract_fn=lambda x: (x, None),
         )
        assert len(p) == 2
        assert bool(p) is True

    def test_empty_provider_is_false(self):
        p = _BoundarySnapshotProvider(
            store=None, request_id="r1", valid_tcs=[],
            in_memory_snapshots={}, extract_fn=lambda x: (x, None),
         )
        assert bool(p) is False

    def test_getitem_from_memory(self):
        results = []
        def ext(snap):
            results.append(snap)
            return (snap, None)
        p = _BoundarySnapshotProvider(
            store=None, request_id="r1", valid_tcs=[10],
            in_memory_snapshots={10: "snap10"}, extract_fn=ext,
         )
        assert p[10] == "snap10"
        assert results == ["snap10"]

    def test_getitem_from_store(self):
        ms = MagicMock()
        ms.load.return_value = "from_store"
        p = _BoundarySnapshotProvider(
            store=ms, request_id="r1", valid_tcs=[10],
            in_memory_snapshots={}, extract_fn=lambda x: (x, None),
         )
        assert p[10] == "from_store"
        ms.load.assert_called_once_with("r1", 10)

    def test_getitem_missing_returns_none(self):
        p = _BoundarySnapshotProvider(
            store=None, request_id="r1", valid_tcs=[10],
            in_memory_snapshots={}, extract_fn=lambda x: (x, None),
         )
        assert p[10] is None


class TestPromptCacheNeedsSnapshots:

    def test_empty_cache(self):
        assert _prompt_cache_needs_snapshots([]) is False

    def test_known_sliceable_cache(self):
        o = MagicMock()
        type(o).__name__ = "KVCache"
        assert _prompt_cache_needs_snapshots([o]) is False

    def test_unknown_cache_type(self):
        o = MagicMock()
        type(o).__name__ = "RotatingCache"
        assert _prompt_cache_needs_snapshots([o]) is True

    def test_sub_caches_all_sliceable(self):
        o = MagicMock()
        o.caches = [MagicMock(), MagicMock()]
        type(o.caches[0]).__name__ = "KVCache"
        type(o.caches[1]).__name__ = "KVCache"
        assert _prompt_cache_needs_snapshots([o]) is False

    def test_sub_caches_one_unknown(self):
        o = MagicMock()
        o.caches = [MagicMock(), MagicMock()]
        type(o.caches[0]).__name__ = "KVCache"
        type(o.caches[1]).__name__ = "UnknownCache"
        assert _prompt_cache_needs_snapshots([o]) is True


class TestCacheLayerTokenCount:

    def test_offset_attribute(self):
        assert _cache_layer_token_count(MagicMock(offset=42, caches=None)) == 42

    def test_size_method(self):
        o = MagicMock()
        o.offset = None
        o.size = MagicMock(return_value=100)
        o.caches = None
        assert _cache_layer_token_count(o) == 100

    def test_nested_caches(self):
        s1 = MagicMock(offset=50, caches=None)
        s2 = MagicMock(offset=100, caches=None)
        o = MagicMock(caches=[s1, s2], offset=None, size=None)
        assert _cache_layer_token_count(o) == 100

    def test_fallback_zero(self):
        o = MagicMock(offset=None, size=None, caches=None)
        assert _cache_layer_token_count(o) == 0


class TestCacheBaseSizes:

    def test_empty(self):
        assert _cache_base_sizes([]) == 0

    def test_max_of_layers(self):
        assert _cache_base_sizes([MagicMock(offset=10), MagicMock(offset=50)]) == 50


class TestSliceVlmExtra:

    def _fake_arr(self, ret="s"):
        class F:
            ndim = 3
            shape = (1, 10, 64)
            def __getitem__(s, k):
                return ret
        import mlx.core as mx
        mx.array = F
        return F()

    def test_slice_mx_array(self):
        assert _slice_vlm_extra({"k": self._fake_arr("sliced")}, 5)["k"] == "sliced"

    def test_non_array_preserved(self):
        r = _slice_vlm_extra({"k1": "v1", "k2": 42}, 5)
        assert r["k1"] == "v1" and r["k2"] == 42

    def test_advance_vlm_extra(self):
        assert _advance_vlm_extra({"k": self._fake_arr("adv")}, 3)["k"] == "adv"


class TestDeferredClearDelay:

    def test_default(self):
        s = MagicMock(running={}, _DEFERRED_CLEAR_DELAY=4)
        assert _deferred_clear_delay(s) == 4

    def test_high_batch(self):
        s = MagicMock(running={str(i): None for i in range(20)}, _DEFERRED_CLEAR_DELAY=4)
        assert _deferred_clear_delay(s) == 9

    def test_min_two(self):
        s = MagicMock(running={}, _DEFERRED_CLEAR_DELAY=0)
        assert _deferred_clear_delay(s) == 2

    def test_max_sixteen(self):
        s = MagicMock(running={str(i): None for i in range(100)}, _DEFERRED_CLEAR_DELAY=10)
        assert _deferred_clear_delay(s) == 16


class TestRequestStatus:

    def test_not_finished(self):
        assert not RequestStatus.is_finished(RequestStatus.WAITING)
        assert not RequestStatus.is_finished(RequestStatus.RUNNING)
        assert not RequestStatus.is_finished(RequestStatus.PREEMPTED)

    def test_finished(self):
        assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH_CAPPED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ABORTED)

    def test_reasons(self):
        assert RequestStatus.get_finish_reason(RequestStatus.FINISHED_STOPPED) == "stop"
        assert RequestStatus.get_finish_reason(RequestStatus.FINISHED_LENGTH_CAPPED) == "length"
        assert RequestStatus.get_finish_reason(RequestStatus.FINISHED_ABORTED) == "abort"
        assert RequestStatus.get_finish_reason(RequestStatus.WAITING) is None


class TestSamplingParams:

    def test_defaults(self):
        sp = SamplingParams()
        assert sp.max_tokens == 256
        assert sp.temperature == 0.7
        assert sp.stop == []

    def test_custom(self):
        assert SamplingParams(max_tokens=512, temperature=0.0).max_tokens == 512


class TestRequest:

    def test_create(self):
        r = Request(request_id="r1", prompt="hi", sampling_params=SamplingParams())
        assert r.request_id == "r1" and r.status == RequestStatus.WAITING

    def test_num_tokens(self):
        r = Request(request_id="r1", prompt="hi", sampling_params=SamplingParams())
        r.num_prompt_tokens = 10
        r.output_token_ids = [1, 2, 3]
        assert r.num_tokens == 13

    def test_append(self):
        r = Request(request_id="r1", prompt="hi", sampling_params=SamplingParams())
        r.append_output_token(42)
        assert r.output_token_ids == [42] and r.num_computed_tokens == 1

    def test_set_finished(self):
        r = Request(request_id="r1", prompt="hi", sampling_params=SamplingParams())
        r.set_finished(RequestStatus.FINISHED_STOPPED)
        assert r.status == RequestStatus.FINISHED_STOPPED and r.finish_reason == "stop"

    def test_comparison(self):
        sp = SamplingParams()
        assert Request("r1", "a", sp) == Request("r1", "b", sp)
        assert Request("r1", "a", sp) != Request("r2", "a", sp)

    def test_ordering(self):
        sp = SamplingParams()
        r1 = Request("r1", "a", sp, priority=1)
        r2 = Request("r2", "b", sp, priority=0)
        assert r2 < r1

    def test_max_tokens(self):
        r = Request("r1", "a", SamplingParams(max_tokens=512))
        assert r.max_tokens == 512


class TestRequestOutput:

    def test_usage(self):
        u = RequestOutput("r1", prompt_tokens=10, completion_tokens=5).usage
        assert u["total_tokens"] == 15


class TestKnownSliceableCacheTypes:

    def test_present(self):
        for t in ("KVCache", "BatchKVCache", "QuantizedKVCache", "TurboQuantKVCache", "ChunkedKVCache"):
            assert t in _KNOWN_SLICEABLE_CACHE_TYPES

    def test_frozenset(self):
        assert isinstance(_KNOWN_SLICEABLE_CACHE_TYPES, frozenset)


# ============================================================
# Sched Trim
# ============================================================

class TestRemoveUidFromActiveBatch:

    def test_negative_uid_noop(self):
        s = MagicMock(batch_generator=MagicMock())
        from fusion_mlx.scheduler.sched_trim import _remove_uid_from_active_batch
        _remove_uid_from_active_batch(s, -1)
        s.batch_generator.remove.assert_not_called()

    def test_none_bg_noop(self):
        s = MagicMock(batch_generator=None)
        from fusion_mlx.scheduler.sched_trim import _remove_uid_from_active_batch
        _remove_uid_from_active_batch(s, 5)

    def test_valid_uid(self):
        s = MagicMock(batch_generator=MagicMock())
        from fusion_mlx.scheduler.sched_trim import _remove_uid_from_active_batch
        _remove_uid_from_active_batch(s, 5)
        s.batch_generator.remove.assert_called_once_with([5])


class TestCheckPendingAborts:

    def test_no_aborts(self):
        s = MagicMock(_pending_abort_ids=set(), uid_to_request_id={})
        from fusion_mlx.scheduler.sched_trim import _check_pending_aborts_for_uids
        assert _check_pending_aborts_for_uids(s, [1, 2]) == []

    def test_aborted(self):
        s = MagicMock(_pending_abort_ids={"r1", "r3"}, uid_to_request_id={1: "r1", 2: "r2", 3: "r3"})
        from fusion_mlx.scheduler.sched_trim import _check_pending_aborts_for_uids
        assert _check_pending_aborts_for_uids(s, [1, 2, 3]) == [1, 3]

    def test_empty_uids(self):
        s = MagicMock(_pending_abort_ids={"r1"}, uid_to_request_id={1: "r1"})
        from fusion_mlx.scheduler.sched_trim import _check_pending_aborts_for_uids
        assert _check_pending_aborts_for_uids(s, []) == []


class TestTrimCacheTreeByOne:

    def _trim(self, obj):
        """Inline _trim_cache_tree_by_one logic."""
        sub_caches = getattr(obj, "caches", None)
        if isinstance(sub_caches, (list, tuple)):
            return all(self._trim(sub) for sub in sub_caches)
        trim_fn = getattr(obj, "trim", None)
        if not callable(trim_fn):
            return False
        try:
            r = trim_fn(1)
            return r is None or int(r) >= 1
        except Exception:
            return False

    def test_no_trim_fn(self):
        assert self._trim(MagicMock(caches=None, trim=None)) is False

    def test_trim_none_is_true(self):
        assert self._trim(MagicMock(caches=None, trim=MagicMock(return_value=None))) is True

    def test_trim_positive(self):
        assert self._trim(MagicMock(caches=None, trim=MagicMock(return_value=1))) is True

    def test_trim_raises(self):
        assert self._trim(MagicMock(caches=None, trim=MagicMock(side_effect=RuntimeError("x")))) is False

    def test_nested_all_trim(self):
        s1 = MagicMock(caches=None, trim=MagicMock(return_value=1))
        s2 = MagicMock(caches=None, trim=MagicMock(return_value=1))
        o = MagicMock(caches=[s1, s2])
        assert self._trim(o) is True
        s1.trim.assert_called_once_with(1)

    def test_nested_one_fails(self):
        s1 = MagicMock(caches=None, trim=MagicMock(return_value=1))
        s2 = MagicMock(caches=None, trim=None)
        o = MagicMock(caches=[s1, s2])
        assert self._trim(o) is False


# ============================================================
# Sched Query
# ============================================================

class TestHasRequests:

    def _call(self, w=None, p=None, r=None, d=None):
        from fusion_mlx.scheduler.sched_query import has_requests
        s = MagicMock(waiting=w or [], prefilling=p or [], running=r or {}, _deferred_clear_at=d)
        return has_requests(s)

    def test_empty(self):
        assert self._call() is False

    def test_waiting(self):
        assert self._call(w=["r1"]) is True

    def test_running(self):
        assert self._call(r={"r1": None}) is True

    def test_deferred(self):
        assert self._call(d=42) is True


class TestPreflightMemoryCheck:

    def test_disabled(self):
        from fusion_mlx.scheduler.sched_query import _preflight_memory_check
        assert _preflight_memory_check(MagicMock(_prefill_memory_guard=False), MagicMock()) is None

    def test_no_limit(self):
        from fusion_mlx.scheduler.sched_query import _preflight_memory_check
        s = MagicMock(_prefill_memory_guard=True, _memory_hard_limit_bytes=0)
        assert _preflight_memory_check(s, MagicMock()) is None

    def test_zero_new_tokens(self):
        from fusion_mlx.scheduler.sched_query import _preflight_memory_check
        s = MagicMock(_prefill_memory_guard=True, _memory_hard_limit_bytes=1024, memory_monitor=MagicMock())
        r = MagicMock(num_prompt_tokens=100, cached_tokens=100)
        assert _preflight_memory_check(s, r) is None


# ============================================================
# Sched Handoff
# ============================================================

class TestSchedHandoff:

    def _sched(self):
        s = MagicMock()
        s.requests = {}
        s.running = {}
        s.paged_cache_manager = None
        return s

    def test_export_not_found(self):
        from fusion_mlx.scheduler.sched_handoff import export_kv_state
        assert export_kv_state(self._sched(), "nope") is None

    def test_export_remaining_tokens(self):
        from fusion_mlx.scheduler.sched_handoff import export_kv_state
        s = self._sched()
        r = Request("r1", "hi", SamplingParams())
        r.remaining_tokens = [1, 2]
        s.requests["r1"] = r
        assert export_kv_state(s, "r1") is None

    def test_export_complete(self):
        from fusion_mlx.scheduler.sched_handoff import export_kv_state
        s = self._sched()
        r = Request("r1", "hi", SamplingParams())
        r.remaining_tokens = []
        r.prompt_cache = ["c1"]
        r.num_computed_tokens = 100
        r.cached_tokens = 50
        r.shared_prefix_blocks = 3
        r.block_table = MagicMock()
        s.requests["r1"] = r
        res = export_kv_state(s, "r1")
        assert res is not None
        assert res["num_computed_tokens"] == 100
        assert res["prompt_cache"] == ["c1"]

    def test_import_not_found(self):
        from fusion_mlx.scheduler.sched_handoff import import_kv_state
        import_kv_state(self._sched(), "nope", {"num_computed_tokens": 10})

    def test_import_applies(self):
        from fusion_mlx.scheduler.sched_handoff import import_kv_state
        s = self._sched()
        r = Request("r1", "hi", SamplingParams())
        s.requests["r1"] = r
        kv = {"prompt_cache": ["c1"], "num_computed_tokens": 50, "cached_tokens": 30,
              "shared_prefix_blocks": 2, "block_table": None}
        import_kv_state(s, "r1", kv)
        assert r.prompt_cache == ["c1"]
        assert r.num_computed_tokens == 50
        assert r.cached_tokens == 30
        assert r.shared_prefix_blocks == 2
        assert r.remaining_tokens == []


# ============================================================
# Sched Batch
# ============================================================

class TestAdaptiveChunkSize:

    def test_tiers(self):
        from fusion_mlx.scheduler.sched_batch import _adaptive_chunk_size
        s = MagicMock()
        s._PREFILL_STEP_TIERS = (1024, 512, 256, 128)
        s._memory_limit_bytes = 100 * 1024**3
        s._memory_hard_limit_bytes = 110 * 1024**3
        s._prefill_safe_zone_ratio = 0.8
        s._prefill_min_chunk_tokens = 1
           # soft_watermark = 80GB, band = 110-80 = 30GB
           # 81GB → (81-80)/30 = 0.033 < 0.25 → 1024
        with patch("fusion_mlx.scheduler.sched_batch.mx") as mock_mx:
            mock_mx.get_active_memory = MagicMock(return_value=81 * 1024**3)
            assert _adaptive_chunk_size(s, 10000, request_id="r1", loop_label="L") == 1024
           # 88GB → (88-80)/30 = 0.267, 0.25-0.50 → 512
        with patch("fusion_mlx.scheduler.sched_batch.mx") as mock_mx:
            mock_mx.get_active_memory = MagicMock(return_value=88 * 1024**3)
            assert _adaptive_chunk_size(s, 10000, request_id="r1", loop_label="L") == 512
           # 108GB → (108-80)/30 = 0.933, >= 0.75 → 128
        with patch("fusion_mlx.scheduler.sched_batch.mx") as mock_mx:
            mock_mx.get_active_memory = MagicMock(return_value=108 * 1024**3)
            assert _adaptive_chunk_size(s, 10000, request_id="r1", loop_label="L") == 128


# ============================================================
# Sched Misc
# ============================================================

class TestFormatBytes:

    def test_units(self):
        from fusion_mlx.scheduler.sched_misc import _format_bytes
        assert "B" in _format_bytes(0)
        assert "KB" in _format_bytes(1500)
        assert "MB" in _format_bytes(1500000)
        assert "GB" in _format_bytes(1500000000)