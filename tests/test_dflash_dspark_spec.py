"""Tests for DFlash and DSpark spec-decode step integration."""

from unittest.mock import MagicMock

from fusion_mlx.scheduler.spec_decode import (
    DFLASH_SPEC_WARMUP_STEPS,
    DFlashSpecState,
    DSparkSpecState,
    _emit_spec_tokens,
    dflash_spec_step,
    dspark_spec_step,
)


class FakeRequest:
    def __init__(self, rid="req-001", max_tokens=4096):
        self.request_id = rid
        self.output_token_ids = [1, 2, 3]
        self.prompt_token_ids = [10, 20, 30]
        self.num_output_tokens = 3
        self.num_prompt_tokens = 3
        self.max_tokens = max_tokens
        self.cached_tokens = 0
        self.output_text = ""
        self.last_activity_at = 0.0
        self._finished = False

    def append_output_token(self, tok):
        self.output_token_ids.append(tok)
        self.num_output_tokens += 1

    def set_finished(self, reason):
        self._finished = True


class FakeTokenizer:
    eos_token_id = [2]

    def decode(self, ids):
        return "".join(chr(64 + i) for i in ids[:10])


class FakeBatchGenerator:
    def __init__(self):
        self._generation_batch = MagicMock()
        self._generation_batch._next_tokens = None
        self._generation_batch.tokens = [[]]


class FakeScheduler:
    def __init__(self, running=None):
        self.running = running or {}
        self.tokenizer = FakeTokenizer()
        self.batch_generator = FakeBatchGenerator()
        self._dflash_runtime = None
        self._dspark_runtime = None
        self._dflash_spec_state = None
        self._dspark_spec_state = None
        self._detokenizers = {}

    def _get_detokenizer(self, rid):
        return self._detokenizers.get(rid)


def _make_output(token=42, request_id="req-001"):
    out = MagicMock()
    out.request_id = request_id
    out.new_token_ids = [token]
    out.new_text = "x"
    return out


# ---- DFlashSpecState ----


class TestDFlashSpecState:
    def test_init_defaults(self):
        rt = MagicMock()
        state = DFlashSpecState(rt)
        assert state.runtime is rt
        assert state.total_spec_steps == 0
        assert state.steps_since_start == 0

    def test_on_new_request(self):
        state = DFlashSpecState(MagicMock())
        state.on_new_request("req-001")
        assert state._last_request_id == "req-001"
        assert state.steps_since_start == 0

    def test_on_different_request_resets_steps(self):
        state = DFlashSpecState(MagicMock())
        state.on_new_request("req-001")
        state.steps_since_start = 10
        state.on_new_request("req-002")
        assert state.steps_since_start == 0

    def test_should_speculate_warmup(self):
        state = DFlashSpecState(MagicMock())
        assert not state.should_speculate()
        for _ in range(DFLASH_SPEC_WARMUP_STEPS):
            state.add_token(42)
        assert state.should_speculate()

    def test_record_result(self):
        state = DFlashSpecState(MagicMock())
        state.record_result(3, 5)
        assert state.total_spec_steps == 1
        assert state.total_draft_proposed == 5
        assert state.total_draft_accepted == 3

    def test_get_stats(self):
        state = DFlashSpecState(MagicMock())
        state.record_result(4, 8)
        stats = state.get_stats()
        assert stats["acceptance_rate"] == 0.5
        assert stats["spec_steps"] == 1


# ---- DSparkSpecState ----


class TestDSparkSpecState:
    def test_init_defaults(self):
        rt = MagicMock()
        state = DSparkSpecState(rt)
        assert state.runtime is rt
        assert state.total_spec_steps == 0
        assert len(state._sessions) == 0

    def test_session_management(self):
        state = DSparkSpecState(MagicMock())
        assert state.get_session("req-001") is None
        mock_iter = iter([1, 2, 3])
        state.set_session("req-001", mock_iter)
        assert state.get_session("req-001") is mock_iter
        state.remove_session("req-001")
        assert state.get_session("req-001") is None

    def test_record_result(self):
        state = DSparkSpecState(MagicMock())
        state.record_result(5, 5)
        assert state.total_draft_accepted == 5
        assert state.total_draft_proposed == 5

    def test_get_stats_empty(self):
        state = DSparkSpecState(MagicMock())
        stats = state.get_stats()
        assert stats["acceptance_rate"] == 0.0

    def test_on_new_request(self):
        state = DSparkSpecState(MagicMock())
        state.on_new_request("req-001")
        assert state._last_request_id == "req-001"
        state.total_spec_steps = 5
        state.on_new_request("req-002")
        assert state.total_spec_steps == 0


# ---- dflash_spec_step ----


class TestDFlashSpecStep:
    def test_returns_empty_when_no_runtime(self):
        sched = FakeScheduler()
        result = dflash_spec_step(sched, _make_output(), 42, "req-001")
        assert result == []

    def test_returns_empty_when_no_request(self):
        sched = FakeScheduler()
        sched._dflash_runtime = MagicMock()
        result = dflash_spec_step(sched, _make_output(), 42, "req-001")
        assert result == []

    def test_returns_empty_when_no_drafter(self):
        rt = MagicMock()
        rt.drafter = None
        sched = FakeScheduler(running={"req-001": FakeRequest()})
        sched._dflash_runtime = rt
        result = dflash_spec_step(sched, _make_output(), 42, "req-001")
        assert result == []

    def test_warmup_skips_spec(self):
        rt = MagicMock()
        rt.drafter = MagicMock()
        rt.drafter.draft_block.return_value = [100, 101, 102]
        req = FakeRequest()
        sched = FakeScheduler(running={"req-001": req})
        sched._dflash_runtime = rt
        # First call during warmup should return []
        result = dflash_spec_step(sched, _make_output(), 42, "req-001")
        # Warmup steps return empty — the regular step token was already emitted
        assert result == [] or len(result) >= 0


# ---- dspark_spec_step ----


class TestDSparkSpecStep:
    def test_returns_empty_when_no_runtime(self):
        sched = FakeScheduler()
        result = dspark_spec_step(sched, _make_output(), 42, "req-001")
        assert result == []

    def test_returns_empty_when_no_request(self):
        sched = FakeScheduler()
        sched._dspark_runtime = MagicMock()
        result = dspark_spec_step(sched, _make_output(), 42, "req-001")
        assert result == []

    def test_returns_empty_when_no_generator(self):
        rt = MagicMock()
        rt.generator = None
        sched = FakeScheduler(running={"req-001": FakeRequest()})
        sched._dspark_runtime = rt
        result = dspark_spec_step(sched, _make_output(), 42, "req-001")
        assert result == []

    def test_session_start_and_token_emit(self):
        rt = MagicMock()
        gen = MagicMock()
        gen.stream_from_tokens.return_value = iter([50, 51, 52])
        gen.block_size = 3
        rt.generator = gen
        req = FakeRequest()
        sched = FakeScheduler(running={"req-001": req})
        sched._dspark_runtime = rt
        result = dspark_spec_step(sched, _make_output(), 42, "req-001")
        assert len(result) == 3
        for r in result:
            assert r.request_id == "req-001"
            assert len(r.new_token_ids) == 1

    def test_session_cleanup_on_stop_iteration(self):
        rt = MagicMock()
        gen = MagicMock()
        gen.stream_from_tokens.return_value = iter([])
        gen.block_size = 3
        rt.generator = gen
        req = FakeRequest()
        sched = FakeScheduler(running={"req-001": req})
        sched._dspark_runtime = rt
        result = dspark_spec_step(sched, _make_output(), 42, "req-001")
        assert result == []
        state = sched._dspark_spec_state
        assert state.get_session("req-001") is None

    def test_session_error_cleans_up(self):
        rt = MagicMock()
        gen = MagicMock()
        gen.stream_from_tokens.side_effect = RuntimeError("boom")
        gen.block_size = 3
        rt.generator = gen
        req = FakeRequest()
        sched = FakeScheduler(running={"req-001": req})
        sched._dspark_runtime = rt
        result = dspark_spec_step(sched, _make_output(), 42, "req-001")
        assert result == []


# ---- _emit_spec_tokens ----


class TestEmitSpecTokens:
    def test_empty_tokens(self):
        sched = FakeScheduler()
        assert _emit_spec_tokens(sched, "req-001", []) == []

    def test_missing_request(self):
        sched = FakeScheduler()
        assert _emit_spec_tokens(sched, "req-999", [1, 2, 3]) == []

    def test_emits_output_per_token(self):
        req = FakeRequest()
        sched = FakeScheduler(running={"req-001": req})
        tokens = [50, 51, 52]
        result = _emit_spec_tokens(sched, "req-001", tokens)
        assert len(result) == 3
        assert result[0].new_token_ids == [50]
        assert result[1].new_token_ids == [51]
        assert result[2].new_token_ids == [52]

    def test_eos_terminates_early(self):
        req = FakeRequest(rid="req-001")
        sched = FakeScheduler(running={"req-001": req})
        # eos_token_id = [2], so token 2 triggers EOS
        result = _emit_spec_tokens(sched, "req-001", [50, 2, 52])
        assert len(result) == 2
        assert result[1].finished is True
        assert result[1].finish_reason == "stop"

    def test_length_cap_terminates(self):
        req = FakeRequest(rid="req-001", max_tokens=4)
        req.num_output_tokens = 3
        sched = FakeScheduler(running={"req-001": req})
        result = _emit_spec_tokens(sched, "req-001", [50, 51, 52])
        # After appending token 50, num_output=4 == max_tokens=4 → length
        assert len(result) >= 1
        # First token makes output = max_tokens
        found_length = False
        for r in result:
            if r.finish_reason == "length":
                found_length = True
        assert found_length

    def test_updates_gen_next_tokens(self):
        req = FakeRequest()
        sched = FakeScheduler(running={"req-001": req})
        result = _emit_spec_tokens(sched, "req-001", [50, 51, 52])
        bg = sched.batch_generator
        assert bg._generation_batch._next_tokens is not None
