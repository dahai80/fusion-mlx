# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from fusion_mlx.request import (
    RequestStatus,
    SamplingParams,
    Request,
    RequestOutput,
)


class TestRequestStatus:

    def test_status_values(self):
        assert RequestStatus.WAITING is not None
        assert RequestStatus.RUNNING is not None
        assert RequestStatus.PREEMPTED is not None
        assert RequestStatus.FINISHED_STOPPED is not None
        assert RequestStatus.FINISHED_LENGTH_CAPPED is not None
        assert RequestStatus.FINISHED_ABORTED is not None

    def test_status_ordering(self):
        assert RequestStatus.WAITING < RequestStatus.FINISHED_STOPPED
        assert RequestStatus.RUNNING < RequestStatus.FINISHED_STOPPED
        assert RequestStatus.PREEMPTED < RequestStatus.FINISHED_STOPPED

    def test_is_finished_active_states(self):
        assert RequestStatus.is_finished(RequestStatus.WAITING) is False
        assert RequestStatus.is_finished(RequestStatus.RUNNING) is False
        assert RequestStatus.is_finished(RequestStatus.PREEMPTED) is False

    def test_is_finished_finished_states(self):
        assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED) is True
        assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH_CAPPED) is True
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ABORTED) is True

    def test_get_finish_reason_stopped(self):
        assert RequestStatus.get_finish_reason(RequestStatus.FINISHED_STOPPED) == "stop"

    def test_get_finish_reason_length_capped(self):
        assert RequestStatus.get_finish_reason(RequestStatus.FINISHED_LENGTH_CAPPED) == "length"

    def test_get_finish_reason_aborted(self):
        assert RequestStatus.get_finish_reason(RequestStatus.FINISHED_ABORTED) == "abort"

    def test_get_finish_reason_active_states(self):
        assert RequestStatus.get_finish_reason(RequestStatus.WAITING) is None
        assert RequestStatus.get_finish_reason(RequestStatus.RUNNING) is None
        assert RequestStatus.get_finish_reason(RequestStatus.PREEMPTED) is None


class TestSamplingParams:

    def test_default_values(self):
        params = SamplingParams()
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 0
        assert params.min_p == 0.0
        assert params.xtc_probability == 0.0
        assert params.xtc_threshold == 0.1
        assert params.repetition_penalty == 1.0
        assert params.presence_penalty == 0.0
        assert params.stop == []
        assert params.stop_token_ids == []
        assert params.logprobs is False
        assert params.top_logprobs is None

    def test_custom_values(self):
        params = SamplingParams(
            max_tokens=1024,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            min_p=0.05,
            xtc_probability=0.5,
            xtc_threshold=0.1,
            repetition_penalty=1.1,
            presence_penalty=0.5,
            stop=["###", "END"],
            stop_token_ids=[2, 100],
            logprobs=True,
            top_logprobs=5,
        )
        assert params.max_tokens == 1024
        assert params.temperature == 0.5
        assert params.top_p == 0.95
        assert params.top_k == 40
        assert params.min_p == 0.05
        assert params.xtc_probability == 0.5
        assert params.xtc_threshold == 0.1
        assert params.repetition_penalty == 1.1
        assert params.presence_penalty == 0.5
        assert params.stop == ["###", "END"]
        assert params.stop_token_ids == [2, 100]
        assert params.logprobs is True
        assert params.top_logprobs == 5

    def test_post_init_none_stop(self):
        params = SamplingParams(stop=None, stop_token_ids=None)
        assert params.stop == []
        assert params.stop_token_ids == []

    def test_greedy_sampling(self):
        params = SamplingParams(temperature=0.0, top_k=1)
        assert params.temperature == 0.0
        assert params.top_k == 1


class TestRequest:

    def test_basic_creation(self):
        request = Request(
            request_id="test-001",
            prompt="Hello, world!",
            sampling_params=SamplingParams(),
        )
        assert request.request_id == "test-001"
        assert request.prompt == "Hello, world!"
        assert request.status == RequestStatus.WAITING
        assert request.output_token_ids == []
        assert request.output_text == ""

    def test_creation_with_token_ids(self):
        request = Request(
            request_id="test-002",
            prompt=[1, 2, 3, 4, 5],
            sampling_params=SamplingParams(),
        )
        assert request.prompt == [1, 2, 3, 4, 5]

    def test_arrival_time_auto_set(self):
        before = time.monotonic()
        request = Request(
            request_id="test-003",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        after = time.monotonic()
        assert before <= request.arrival_time <= after

    def test_num_output_tokens_property(self):
        request = Request(
            request_id="test-004",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        assert request.num_output_tokens == 0

        request.output_token_ids = [100, 200, 300]
        assert request.num_output_tokens == 3

    def test_num_tokens_property(self):
        request = Request(
            request_id="test-005",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request.num_prompt_tokens = 10
        request.output_token_ids = [100, 200, 300]
        assert request.num_tokens == 13

    def test_max_tokens_property(self):
        request = Request(
            request_id="test-006",
            prompt="Test",
            sampling_params=SamplingParams(max_tokens=512),
        )
        assert request.max_tokens == 512

    def test_is_finished_method(self):
        request = Request(
            request_id="test-007",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        assert request.is_finished() is False

        request.status = RequestStatus.FINISHED_STOPPED
        assert request.is_finished() is True

    def test_get_finish_reason_method(self):
        request = Request(
            request_id="test-008",
            prompt="Test",
            sampling_params=SamplingParams(),
        )

        assert request.get_finish_reason() is None

        request.status = RequestStatus.FINISHED_STOPPED
        assert request.get_finish_reason() == "stop"

        request.finish_reason = "custom_reason"
        assert request.get_finish_reason() == "custom_reason"

    def test_append_output_token(self):
        request = Request(
            request_id="test-009",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request.append_output_token(100)
        request.append_output_token(200)

        assert request.output_token_ids == [100, 200]
        assert request.num_computed_tokens == 2

    def test_set_finished(self):
        request = Request(
            request_id="test-010",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request.set_finished(RequestStatus.FINISHED_STOPPED)

        assert request.status == RequestStatus.FINISHED_STOPPED
        assert request.finish_reason == "stop"

    def test_set_finished_with_reason(self):
        request = Request(
            request_id="test-011",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request.set_finished(RequestStatus.FINISHED_ABORTED, reason="user_cancelled")

        assert request.status == RequestStatus.FINISHED_ABORTED
        assert request.finish_reason == "user_cancelled"

    def test_comparison_by_priority(self):
        request1 = Request(
            request_id="test-012",
            prompt="Test",
            sampling_params=SamplingParams(),
            priority=1,
        )
        request2 = Request(
            request_id="test-013",
            prompt="Test",
            sampling_params=SamplingParams(),
            priority=2,
        )
        assert request1 < request2

    def test_comparison_by_arrival_time(self):
        request1 = Request(
            request_id="test-014",
            prompt="Test",
            sampling_params=SamplingParams(),
            arrival_time=100.0,
        )
        request2 = Request(
            request_id="test-015",
            prompt="Test",
            sampling_params=SamplingParams(),
            arrival_time=200.0,
        )
        assert request1 < request2

    def test_hash(self):
        request1 = Request(
            request_id="test-016",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request2 = Request(
            request_id="test-016",
            prompt="Different prompt",
            sampling_params=SamplingParams(),
        )
        assert hash(request1) == hash(request2)

    def test_equality(self):
        request1 = Request(
            request_id="test-017",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request2 = Request(
            request_id="test-017",
            prompt="Different",
            sampling_params=SamplingParams(),
        )
        request3 = Request(
            request_id="test-018",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        assert request1 == request2
        assert request1 != request3

    def test_equality_with_non_request(self):
        request = Request(
            request_id="test-019",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        assert request != "test-019"
        assert request != 123
        assert request != None

    def test_reasoning_model_fields(self):
        request = Request(
            request_id="test-020",
            prompt="Test",
            sampling_params=SamplingParams(),
            needs_think_prefix=True,
        )
        assert request.needs_think_prefix is True
        assert request.think_prefix_sent is False

    def test_harmony_model_field(self):
        request = Request(
            request_id="test-021",
            prompt="Test",
            sampling_params=SamplingParams(),
            is_harmony_model=True,
        )
        assert request.is_harmony_model is True

    def test_multimodal_fields(self):
        request = Request(
            request_id="test-022",
            prompt="Describe this image",
            sampling_params=SamplingParams(),
            images=["image_data_1", "image_data_2"],
            videos=["video_data_1"],
        )
        assert request.images == ["image_data_1", "image_data_2"]
        assert request.videos == ["video_data_1"]


class TestRequestOutput:

    def test_basic_creation(self):
        output = RequestOutput(request_id="test-001")
        assert output.request_id == "test-001"
        assert output.new_token_ids == []
        assert output.new_text == ""
        assert output.output_token_ids == []
        assert output.output_text == ""
        assert output.finished is False
        assert output.finish_reason is None

    def test_with_tokens(self):
        output = RequestOutput(
            request_id="test-002",
            new_token_ids=[100, 200],
            new_text="Hello",
            output_token_ids=[100, 200, 300, 400],
            output_text="Hello world",
        )
        assert output.new_token_ids == [100, 200]
        assert output.new_text == "Hello"
        assert output.output_token_ids == [100, 200, 300, 400]
        assert output.output_text == "Hello world"

    def test_finished_output(self):
        output = RequestOutput(
            request_id="test-003",
            finished=True,
            finish_reason="stop",
        )
        assert output.finished is True
        assert output.finish_reason == "stop"

    def test_usage_property(self):
        output = RequestOutput(
            request_id="test-004",
            prompt_tokens=10,
            completion_tokens=20,
        )
        usage = output.usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_usage_property_zero(self):
        output = RequestOutput(request_id="test-005")
        usage = output.usage
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_tool_calls(self):
        tool_calls = [
            {"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "search", "arguments": "{}"}},
        ]
        output = RequestOutput(
            request_id="test-006",
            tool_calls=tool_calls,
        )
        assert output.tool_calls == tool_calls
        assert len(output.tool_calls) == 2
