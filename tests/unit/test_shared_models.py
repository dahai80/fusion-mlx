import pytest
import time
from unittest.mock import patch
from fusion_mlx.api.shared_models import (
    IDPrefix,
    generate_id,
    get_unix_timestamp,
    BaseUsage,
)


class TestIDPrefix:
    def test_prefix_values(self):
        assert IDPrefix.CHAT_COMPLETION == "chatcmpl"
        assert IDPrefix.COMPLETION == "cmpl"
        assert IDPrefix.EMBEDDING == "emb"
        assert IDPrefix.RERANK == "rerank"
        assert IDPrefix.RESPONSE == "resp"
        assert IDPrefix.MESSAGE == "msg"
        assert IDPrefix.FUNCTION_CALL == "fc"
        assert IDPrefix.REASONING == "rs"

    def test_prefix_string_conversion(self):
        assert IDPrefix.CHAT_COMPLETION.value == "chatcmpl"
        assert IDPrefix.COMPLETION.value == "cmpl"

    def test_prefix_equality(self):
        assert IDPrefix.CHAT_COMPLETION == "chatcmpl"
        assert IDPrefix.COMPLETION == "cmpl"
        assert IDPrefix.CHAT_COMPLETION != IDPrefix.COMPLETION


class TestGenerateId:
    def test_generate_id_format(self):
        result = generate_id(IDPrefix.CHAT_COMPLETION)
        assert result.startswith("chatcmpl-")
        suffix = result[len("chatcmpl-"):]
        assert len(suffix) == 8
        assert all(c.isalnum() for c in suffix)

    def test_generate_id_with_completion_prefix(self):
        result = generate_id(IDPrefix.COMPLETION)
        assert result.startswith("cmpl-")
        suffix = result[len("cmpl-"):]
        assert len(suffix) == 8

    def test_generate_id_with_embedding_prefix(self):
        result = generate_id(IDPrefix.EMBEDDING)
        assert result.startswith("emb-")

    def test_generate_id_with_rerank_prefix(self):
        result = generate_id(IDPrefix.RERANK)
        assert result.startswith("rerank-")

    def test_generate_id_with_message_prefix(self):
        result = generate_id(IDPrefix.MESSAGE)
        assert result.startswith("msg_")
        suffix = result[len("msg_"):]
        assert len(suffix) == 24

    def test_generate_id_with_response_prefix(self):
        result = generate_id(IDPrefix.RESPONSE)
        assert result.startswith("resp_")
        suffix = result[len("resp_"):]
        assert len(suffix) == 24

    def test_generate_id_with_function_call_prefix(self):
        result = generate_id(IDPrefix.FUNCTION_CALL)
        assert result.startswith("fc_")
        suffix = result[len("fc_"):]
        assert len(suffix) == 8

    def test_generate_id_with_reasoning_prefix(self):
        result = generate_id(IDPrefix.REASONING)
        assert result.startswith("rs_")
        suffix = result[len("rs_"):]
        assert len(suffix) == 24

    def test_generate_id_uniqueness(self):
        ids = {generate_id(IDPrefix.CHAT_COMPLETION) for _ in range(100)}
        assert len(ids) == 100

    def test_generate_id_custom_length(self):
        result = generate_id(IDPrefix.CHAT_COMPLETION, length=12)
        suffix = result[len("chatcmpl-"):]
        assert len(suffix) == 12

    def test_generate_id_zero_length(self):
        result = generate_id(IDPrefix.CHAT_COMPLETION, length=0)
        assert result == "chatcmpl-"

    def test_generate_id_large_length(self):
        result = generate_id(IDPrefix.CHAT_COMPLETION, length=32)
        suffix = result[len("chatcmpl-"):]
        assert len(suffix) == 32

    def test_generate_id_alphanumeric_only(self):
        for _ in range(50):
            result = generate_id(IDPrefix.CHAT_COMPLETION)
            suffix = result[len("chatcmpl-"):]
            assert all(c.isalnum() for c in suffix)


class TestGetUnixTimestamp:
    def test_returns_int(self):
        result = get_unix_timestamp()
        assert isinstance(result, int)

    def test_reasonable_value(self):
        result = get_unix_timestamp()
        assert result > 1700000000
        assert result < 2000000000

    def test_monotonic_enough(self):
        t1 = get_unix_timestamp()
        t2 = get_unix_timestamp()
        assert t2 >= t1

    @patch("fusion_mlx.api.shared_models.time.time")
    def test_with_mocked_time(self, mock_time):
        mock_time.return_value = 1234567890.123
        result = get_unix_timestamp()
        assert result == 1234567890

    @patch("fusion_mlx.api.shared_models.time.time")
    def test_zero_time(self, mock_time):
        mock_time.return_value = 0.0
        result = get_unix_timestamp()
        assert result == 0

    @patch("fusion_mlx.api.shared_models.time.time")
    def test_large_timestamp(self, mock_time):
        mock_time.return_value = 9999999999.999
        result = get_unix_timestamp()
        assert result == 9999999999


class TestBaseUsage:
    def test_default_values(self):
        usage = BaseUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_with_values(self):
        usage = BaseUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_partial_values(self):
        usage = BaseUsage(prompt_tokens=100)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 0

    def test_serialization(self):
        usage = BaseUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        data = usage.model_dump()
        assert data["prompt_tokens"] == 100
        assert data["completion_tokens"] == 50
        assert data["total_tokens"] == 150

    def test_deserialization(self):
        data = {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        }
        usage = BaseUsage(**data)
        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 100
        assert usage.total_tokens == 300

    def test_equality(self):
        u1 = BaseUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        u2 = BaseUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u1 == u2

    def test_inequality(self):
        u1 = BaseUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        u2 = BaseUsage(prompt_tokens=11, completion_tokens=20, total_tokens=31)
        assert u1 != u2

    def test_json_schema(self):
        schema = BaseUsage.model_json_schema()
        assert "properties" in schema
        assert "prompt_tokens" in schema["properties"]
        assert "completion_tokens" in schema["properties"]
        assert "total_tokens" in schema["properties"]

    def test_large_values(self):
        usage = BaseUsage(
            prompt_tokens=10**9,
            completion_tokens=10**9,
            total_tokens=2 * 10**9,
        )
        assert usage.prompt_tokens == 10**9
        assert usage.total_tokens == 2 * 10**9

    def test_anthropic_style_aliases(self):
        usage = BaseUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_total_tokens_auto_calculated(self):
        usage = BaseUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150
