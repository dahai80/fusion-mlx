# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

from fusion_mlx.engine.vlm import VLMBatchedEngine
from fusion_mlx.service.helpers import compute_prompt_tokens_for_messages

PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8A"
    "AAwAB/AF+p1K2wAAAAABJRU5ErkJggg=="
)


def _make_engine(tokenizer=None):
    engine = VLMBatchedEngine(model_name="test-vlm")
    engine._vlm_model = MagicMock()
    engine._tokenizer = tokenizer or MagicMock()
    engine._loaded = True
    engine._engine = MagicMock()
    return engine


def _image_messages():
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + PNG_1PX},
                },
                {"type": "text", "text": "Describe"},
            ],
        }
    ]


class TestBuildPrompt:
    def test_merges_system_messages_and_forwards_tools(self):
        processor = MagicMock()
        processor.apply_chat_template.return_value = "rendered"
        engine = _make_engine()
        engine._processor = processor

        prompt = engine.build_prompt(
            [
                {"role": "system", "content": "one"},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "two"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "Bash", "parameters": {}},
                }
            ],
        )

        assert prompt == "rendered"
        rendered = processor.apply_chat_template.call_args[0][0]
        assert rendered[0] == {"role": "system", "content": "one\n\ntwo"}
        assert rendered[1] == {"role": "user", "content": "hi"}
        kwargs = processor.apply_chat_template.call_args[1]
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        assert "tools" in kwargs

    def test_returns_none_when_images_present(self):
        engine = _make_engine()
        engine._processor = MagicMock()

        assert engine.build_prompt(_image_messages()) is None
        engine._processor.apply_chat_template.assert_not_called()

    def test_falls_back_to_processor_tokenizer(self):
        processor = MagicMock(spec=["tokenizer"])
        processor.tokenizer.apply_chat_template.return_value = "rendered"
        engine = _make_engine()
        engine._processor = processor

        assert engine.build_prompt([{"role": "user", "content": "hi"}]) == "rendered"

    def test_adds_default_system_when_none_given(self):
        processor = MagicMock()
        processor.apply_chat_template.return_value = "rendered"
        engine = _make_engine()
        engine._processor = processor

        engine.build_prompt([{"role": "user", "content": "hi"}])

        rendered = processor.apply_chat_template.call_args[0][0]
        assert rendered[0]["role"] == "system"
        assert rendered[0]["content"]


class TestComputePromptTokensOnVlmEngine:
    def test_text_request_counts_nonzero(self):
        tokenizer = MagicMock()
        tokenizer.bos_token = None
        tokenizer.encode.return_value = [1] * 42
        engine = _make_engine(tokenizer=tokenizer)
        engine._processor = MagicMock()
        engine._processor.apply_chat_template.return_value = "prompt"

        assert engine.is_mllm is True
        count = compute_prompt_tokens_for_messages(
            engine, [{"role": "user", "content": "hi"}]
        )
        assert count == 42

    def test_image_request_stays_uncounted(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1] * 42
        engine = _make_engine(tokenizer=tokenizer)
        engine._processor = MagicMock()

        assert compute_prompt_tokens_for_messages(engine, _image_messages()) == 0

    def test_mllm_without_build_prompt_still_returns_zero(self):
        class _Naked:
            is_mllm = True

        assert (
            compute_prompt_tokens_for_messages(
                _Naked(), [{"role": "user", "content": "hi"}]
            )
            == 0
        )
