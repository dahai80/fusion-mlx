# SPDX-License-Identifier: Apache-2.0
"""Tests for fusion_mlx.request module."""

from fusion_mlx.request import (
    Request,
    RequestOutput,
    RequestStatus,
    SamplingParams,
)


class TestRequestStatus:
    """Test cases for RequestStatus enum."""

    def test_is_finished_waiting(self):
        assert RequestStatus.is_finished(RequestStatus.WAITING) is False

    def test_is_finished_running(self):
        assert RequestStatus.is_finished(RequestStatus.RUNNING) is False

    def test_is_finished_preempted(self):
        assert RequestStatus.is_finished(RequestStatus.PREEMPTED) is False

    def test_is_finished_stopped(self):
        assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED) is True

    def test_is_finished_length_capped(self):
        assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH_CAPPED) is True

    def test_is_finished_aborted(self):
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ABORTED) is True

    def test_get_finish_reason_waiting(self):
        assert RequestStatus.get_finish_reason(RequestStatus.WAITING) is None

    def test_get_finish_reason_running(self):
        assert RequestStatus.get_finish_reason(RequestStatus.RUNNING) is None

    def test_get_finish_reason_preempted(self):
        assert RequestStatus.get_finish_reason(RequestStatus.PREEMPTED) is None

    def test_get_finish_reason_stopped(self):
        assert RequestStatus.get_finish_reason(RequestStatus.FINISHED_STOPPED) == "stop"

    def test_get_finish_reason_length_capped(self):
        assert (
            RequestStatus.get_finish_reason(RequestStatus.FINISHED_LENGTH_CAPPED)
            == "length"
        )

    def test_get_finish_reason_aborted(self):
        assert (
            RequestStatus.get_finish_reason(RequestStatus.FINISHED_ABORTED) == "abort"
        )

    def test_status_ordering(self):
        assert RequestStatus.WAITING < RequestStatus.RUNNING
        assert RequestStatus.RUNNING < RequestStatus.PREEMPTED
        assert RequestStatus.PREEMPTED < RequestStatus.FINISHED_STOPPED
        assert RequestStatus.FINISHED_STOPPED < RequestStatus.FINISHED_LENGTH_CAPPED
        assert RequestStatus.FINISHED_LENGTH_CAPPED < RequestStatus.FINISHED_ABORTED


class TestSamplingParams:
    """Test cases for SamplingParams dataclass."""

    def test_default_values(self):
        params = SamplingParams()
        assert params.max_tokens == 65536
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 0
        assert params.min_p == 0.0
        assert params.repetition_penalty == 1.0
        assert params.presence_penalty == 0.0
        assert params.frequency_penalty == 0.0
        assert params.stop == []
        assert params.stop_token_ids == []
        assert params.logprobs is False
        assert params.top_logprobs is None
        assert params.seed is None

    def test_custom_values(self):
        params = SamplingParams(
            max_tokens=100,
            temperature=0.5,
            top_p=0.95,
            stop=["\n"],
        )
        assert params.max_tokens == 100
        assert params.temperature == 0.5
        assert params.top_p == 0.95
        assert params.stop == ["\n"]

    def test_stop_list(self):
        params = SamplingParams(stop=["\n", "END"])
        assert params.stop == ["\n", "END"]

    def test_stop_token_ids(self):
        params = SamplingParams(stop_token_ids=[1, 2, 3])
        assert params.stop_token_ids == [1, 2, 3]

    def test_logprobs_params(self):
        params = SamplingParams(logprobs=True, top_logprobs=5)
        assert params.logprobs is True
        assert params.top_logprobs == 5


class TestRequest:
    """Test cases for Request dataclass."""

    def test_basic_request(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert req.request_id == "test-1"
        assert req.prompt == "Hello"
        assert req.status == RequestStatus.WAITING
        assert req.num_computed_tokens == 0
        assert req.output_token_ids == []
        assert req.output_text == ""
        assert req.finish_reason is None

    def test_request_with_token_ids(self):
        req = Request(
            request_id="test-2",
            prompt=[1, 2, 3, 4],
            sampling_params=SamplingParams(),
            prompt_token_ids=[1, 2, 3, 4],
            num_prompt_tokens=4,
        )
        assert req.prompt == [1, 2, 3, 4]
        assert req.prompt_token_ids == [1, 2, 3, 4]
        assert req.num_prompt_tokens == 4

    def test_request_with_priority(self):
        req1 = Request(
            request_id="test-3",
            prompt="Hi",
            sampling_params=SamplingParams(),
            priority=1,
        )
        req2 = Request(
            request_id="test-4",
            prompt="Hi",
            sampling_params=SamplingParams(),
            priority=2,
        )
        assert req1 < req2

    def test_request_output_token_count(self):
        req = Request(
            request_id="test-5",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req.append_output_token(100)
        req.append_output_token(200)
        assert req.num_output_tokens == 2

    def test_request_total_token_count(self):
        req = Request(
            request_id="test-6",
            prompt="Hello",
            sampling_params=SamplingParams(),
            num_prompt_tokens=5,
        )
        req.append_output_token(100)
        req.append_output_token(200)
        req.append_output_token(300)
        assert req.num_tokens == 8

    def test_request_max_tokens(self):
        params = SamplingParams(max_tokens=50)
        req = Request(
            request_id="test-7",
            prompt="Hello",
            sampling_params=params,
        )
        assert req.max_tokens == 50

    def test_request_is_finished(self):
        req = Request(
            request_id="test-8",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert req.is_finished() is False
        req.status = RequestStatus.FINISHED_STOPPED
        assert req.is_finished() is True

    def test_request_get_finish_reason(self):
        req = Request(
            request_id="test-9",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert req.get_finish_reason() is None
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        assert req.get_finish_reason() == "length"

    def test_request_set_finished(self):
        req = Request(
            request_id="test-10",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req.set_finished(RequestStatus.FINISHED_ABORTED)
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.finish_reason == "abort"

    def test_request_set_finished_with_custom_reason(self):
        req = Request(
            request_id="test-11",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req.set_finished(RequestStatus.FINISHED_STOPPED, reason="custom_reason")
        assert req.finish_reason == "custom_reason"

    def test_request_comparison_by_priority(self):
        req1 = Request(
            request_id="test-12",
            prompt="Hi",
            sampling_params=SamplingParams(),
            priority=1,
            arrival_time=100.0,
        )
        req2 = Request(
            request_id="test-13",
            prompt="Hi",
            sampling_params=SamplingParams(),
            priority=2,
            arrival_time=50.0,
        )
        assert req1 < req2

    def test_request_comparison_by_arrival_time(self):
        req1 = Request(
            request_id="test-14",
            prompt="Hi",
            sampling_params=SamplingParams(),
            priority=1,
            arrival_time=50.0,
        )
        req2 = Request(
            request_id="test-15",
            prompt="Hi",
            sampling_params=SamplingParams(),
            priority=1,
            arrival_time=100.0,
        )
        assert req1 < req2

    def test_request_hash(self):
        req = Request(
            request_id="test-16",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert hash(req) == hash("test-16")

    def test_request_equality(self):
        req1 = Request(
            request_id="test-17",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req2 = Request(
            request_id="test-17",
            prompt="World",
            sampling_params=SamplingParams(max_tokens=100),
        )
        assert req1 == req2

    def test_request_inequality(self):
        req1 = Request(
            request_id="test-18",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req2 = Request(
            request_id="test-19",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert req1 != req2


class TestRequestOutput:
    """Test cases for RequestOutput dataclass."""

    def test_basic_output(self):
        output = RequestOutput(request_id="test-20")
        assert output.request_id == "test-20"
        assert output.new_token_ids == []
        assert output.new_text == ""
        assert output.output_token_ids == []
        assert output.output_text == ""
        assert output.finished is False
        assert output.finish_reason is None
        assert output.prompt_tokens == 0
        assert output.completion_tokens == 0
        assert output.error is None

    def test_output_with_tokens(self):
        output = RequestOutput(
            request_id="test-21",
            new_token_ids=[1, 2, 3],
            new_text="Hello",
            output_token_ids=[1, 2, 3, 4, 5],
            output_text="Hello World",
            completion_tokens=5,
        )
        assert output.new_token_ids == [1, 2, 3]
        assert output.new_text == "Hello"
        assert output.output_token_ids == [1, 2, 3, 4, 5]
        assert output.output_text == "Hello World"
        assert output.completion_tokens == 5

    def test_output_finished(self):
        output = RequestOutput(
            request_id="test-22",
            finished=True,
            finish_reason="stop",
        )
        assert output.finished is True
        assert output.finish_reason == "stop"

    def test_output_with_error(self):
        output = RequestOutput(
            request_id="test-23",
            error="Out of memory",
        )
        assert output.error == "Out of memory"

    def test_output_usage(self):
        output = RequestOutput(
            request_id="test-24",
            prompt_tokens=10,
            completion_tokens=5,
        )
        usage = output.usage
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_output_with_logprobs(self):
        logprobs_data = [{"token_id": 1, "logprob": -0.5}]
        output = RequestOutput(
            request_id="test-25",
            logprobs=logprobs_data,
        )
        assert output.logprobs == logprobs_data
